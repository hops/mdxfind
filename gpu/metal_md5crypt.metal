/* metal_md5crypt.metal — MD5CRYPT ($1$) kernel for Metal
 *
 * Algorithm: $1$salt$hash — 1000 rounds of MD5 with alternating password/salt mixing.
 * Hit stride: 6 (word_idx, salt_idx, hx, hy, hz, hw).
 * Uses md5_block_full from metal_common.metal (included via common_str concatenation).
 */

/* md5_oneshot: compute MD5 of up to 128 bytes (1 or 2 blocks).
 * data[] is a thread-local byte buffer, len is the message length. */
static void md5_oneshot(thread const uint8_t *data, int len, thread uint4 &hash) {
    uint M[16];
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);

    int pos = 0;
    /* Process full 64-byte blocks */
    while (pos + 64 <= len) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)data[pos+i*4] | ((uint)data[pos+i*4+1]<<8) |
                   ((uint)data[pos+i*4+2]<<16) | ((uint)data[pos+i*4+3]<<24);
        md5_block_full(hash, M);
        pos += 64;
    }
    /* Final block(s) with padding */
    int rem = len - pos;
    uint8_t pad[128];
    for (int i = 0; i < 128; i++) pad[i] = 0;
    for (int i = 0; i < rem; i++) pad[i] = data[pos + i];
    pad[rem] = 0x80;
    int blocks = (rem < 56) ? 1 : 2;
    int lenoff = (blocks == 1) ? 56 : 120;
    pad[lenoff]   = (uint8_t)((len * 8) & 0xff);
    pad[lenoff+1] = (uint8_t)(((len * 8) >> 8) & 0xff);
    pad[lenoff+2] = (uint8_t)(((len * 8) >> 16) & 0xff);
    pad[lenoff+3] = (uint8_t)(((len * 8) >> 24) & 0xff);
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)pad[b*64+i*4] | ((uint)pad[b*64+i*4+1]<<8) |
                   ((uint)pad[b*64+i*4+2]<<16) | ((uint)pad[b*64+i*4+3]<<24);
        md5_block_full(hash, M);
    }
}

/* ---- MD5CRYPT kernel (e511): $1$salt$hash, 1000 MD5 iterations ---- */
kernel void md5crypt_batch(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *salts       [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens   [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hex_lens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];

    /* Salt buffer has "$1$salt$" -- extract raw salt (skip "$1$", stop before trailing "$") */
    device const uint8_t *salt_raw = salts + soff + 3; /* skip "$1$" */
    int saltlen = slen_full - 4; /* remove "$1$" prefix and trailing "$" */
    if (saltlen < 0) saltlen = 0;
    if (saltlen > 8) saltlen = 8;

    /* Local buffers */
    uint8_t buf[256];
    uint4 digest;

    /* Step 1: digest_b = MD5(password + salt + password) */
    int blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i];
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    md5_oneshot(buf, blen, digest);
    uint8_t digest_b[16];
    for (int i = 0; i < 4; i++) {
        uint w = ((thread uint *)&digest)[i];
        digest_b[i*4]   = w & 0xff;
        digest_b[i*4+1] = (w >> 8) & 0xff;
        digest_b[i*4+2] = (w >> 16) & 0xff;
        digest_b[i*4+3] = (w >> 24) & 0xff;
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
    uint8_t dig[16];
    for (int i = 0; i < 4; i++) {
        uint w = ((thread uint *)&digest)[i];
        dig[i*4]   = w & 0xff;
        dig[i*4+1] = (w >> 8) & 0xff;
        dig[i*4+2] = (w >> 16) & 0xff;
        dig[i*4+3] = (w >> 24) & 0xff;
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
            uint w = ((thread uint *)&digest)[i];
            dig[i*4]   = w & 0xff;
            dig[i*4+1] = (w >> 8) & 0xff;
            dig[i*4+2] = (w >> 16) & 0xff;
            dig[i*4+3] = (w >> 24) & 0xff;
        }
    }

    PROBE6(digest, word_idx, salt_idx)
}
