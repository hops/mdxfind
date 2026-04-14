/* M_copy_bytes, M_set_byte provided by gpu_common.cl */

/* ---- PHPBB3/phpass kernel (e455): MD5 iterated 2^N times ---- */
/* Salt buffer has "$H$Csalt8chr" (12 bytes). Iteration count from salt[3].
 * Algorithm: MD5(salt + pass), then N iterations of MD5(hash_binary + pass).
 * Password max 39 bytes (single MD5 block for 16 + 39 = 55). */
__constant uchar phpitoa64_k[] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

__kernel void phpbb3_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
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
    __global const uchar *salt_data = salts + soff;

    /* Decode iteration count from salt[3] */
    uchar ic = salt_data[3];
    int log2count = 0;
    for (int k = 0; k < 64; k++) { if (phpitoa64_k[k] == ic) { log2count = k; break; } }
    uint count = 1u << log2count;

    /* Step 1: MD5(salt[4..11] + password) — salt is 8 bytes */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;
    M_copy_bytes(M, 0, salt_data + 4, 8);
    M_copy_bytes(M, 8, pass, plen);
    int total = 8 + plen;
    M_set_byte(M, total, 0x80);
    M[14] = total * 8;
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Step 2: N iterations of MD5(hash_binary[16] + password[plen]) */
    /* Pre-pack M[] with password, padding, length — constant across iterations */
    total = 16 + plen;
    for (int i = 0; i < 16; i++) M[i] = 0;
    M_copy_bytes(M, 16, pass, plen);
    M_set_byte(M, total, 0x80);
    M[14] = total * 8;
    /* Iteration loop: only M[0..3] change (the hash digest) */
    for (uint ic2 = 0; ic2 < count; ic2++) {
        M[0] = hx; M[1] = hy; M[2] = hz; M[3] = hw;
        hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
    }
}
