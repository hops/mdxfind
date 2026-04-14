/* M_copy_bytes, M_set_byte provided by gpu_common.cl */

/* ---- MD5CRYPT kernel (e511): $1$salt$hash, 1000 MD5 iterations ---- */
/* Salt buffer contains "$1$salt$" from Typesalt. Kernel extracts raw salt. */

/* Helper: build MD5 message block from a byte buffer, with padding */
void md5_oneshot(const uchar *data, int len, uint *out) {
    uint M[16];
    out[0] = 0x67452301u; out[1] = 0xEFCDAB89u;
    out[2] = 0x98BADCFEu; out[3] = 0x10325476u;

    int pos = 0;
    /* Process full 64-byte blocks */
    while (pos + 64 <= len) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)data[pos+i*4] | ((uint)data[pos+i*4+1]<<8) |
                   ((uint)data[pos+i*4+2]<<16) | ((uint)data[pos+i*4+3]<<24);
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
        pos += 64;
    }
    /* Final block(s) with padding */
    int rem = len - pos;
    uchar pad[128];
    for (int i = 0; i < 128; i++) pad[i] = 0;
    for (int i = 0; i < rem; i++) pad[i] = data[pos + i];
    pad[rem] = 0x80;
    int blocks = (rem < 56) ? 1 : 2;
    int lenoff = (blocks == 1) ? 56 : 120;
    pad[lenoff] = (len * 8) & 0xff;
    pad[lenoff+1] = ((len * 8) >> 8) & 0xff;
    pad[lenoff+2] = ((len * 8) >> 16) & 0xff;
    pad[lenoff+3] = ((len * 8) >> 24) & 0xff;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)pad[b*64+i*4] | ((uint)pad[b*64+i*4+1]<<8) |
                   ((uint)pad[b*64+i*4+2]<<16) | ((uint)pad[b*64+i*4+3]<<24);
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
    }
}

__kernel void md5crypt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];

    /* Salt buffer has "$1$salt$" — extract raw salt (skip "$1$", stop before trailing "$") */
    __global const uchar *salt_raw = salts + soff + 3; /* skip "$1$" */
    int saltlen = slen_full - 4; /* remove "$1$" prefix and trailing "$" */
    if (saltlen < 0) saltlen = 0;
    if (saltlen > 8) saltlen = 8;

    /* Local buffers */
    uchar buf[256]; /* working buffer for MD5 input */
    uint digest[4]; /* current MD5 state */

    /* Step 1: digest_b = MD5(password + salt + password) */
    int blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i];
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    md5_oneshot(buf, blen, digest);
    uchar digest_b[16];
    for (int i = 0; i < 4; i++) {
        digest_b[i*4]   = digest[i] & 0xff;
        digest_b[i*4+1] = (digest[i] >> 8) & 0xff;
        digest_b[i*4+2] = (digest[i] >> 16) & 0xff;
        digest_b[i*4+3] = (digest[i] >> 24) & 0xff;
    }

    /* Step 2: digest_a = MD5(password + "$1$" + salt + digest_b_chunks + bit_bytes) */
    blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    buf[blen++] = '$'; buf[blen++] = '1'; buf[blen++] = '$';
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i];
    /* Append digest_b for password length bytes */
    for (int x = plen; x > 0; x -= 16) {
        int n = (x > 16) ? 16 : x;
        for (int i = 0; i < n; i++) buf[blen++] = digest_b[i];
    }
    /* Bit-dependent bytes */
    for (int x = plen; x != 0; x >>= 1)
        buf[blen++] = (x & 1) ? 0 : pass[0];
    md5_oneshot(buf, blen, digest);

    /* Step 3: 1000 iterations */
    uchar dig[16];
    for (int i = 0; i < 4; i++) {
        dig[i*4]   = digest[i] & 0xff;
        dig[i*4+1] = (digest[i] >> 8) & 0xff;
        dig[i*4+2] = (digest[i] >> 16) & 0xff;
        dig[i*4+3] = (digest[i] >> 24) & 0xff;
    }
    for (int x = 0; x < 1000; x++) {
        blen = 0;
        if (x & 1) { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        else       { for (int i = 0; i < 16; i++) buf[blen++] = dig[i]; }
        if (x % 3) { for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i]; }
        if (x % 7) { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        if (x & 1) { for (int i = 0; i < 16; i++) buf[blen++] = dig[i]; }
        else       { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        md5_oneshot(buf, blen, digest);
        for (int i = 0; i < 4; i++) {
            dig[i*4]   = digest[i] & 0xff;
            dig[i*4+1] = (digest[i] >> 8) & 0xff;
            dig[i*4+2] = (digest[i] >> 16) & 0xff;
            dig[i*4+3] = (digest[i] >> 24) & 0xff;
        }
    }

    /* Probe compact table with 16-byte binary result */
    uint hx = digest[0], hy = digest[1], hz = digest[2], hw = digest[3];
    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
    }
}
