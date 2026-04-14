/* hex_byte_lc/uc, md5_to_hex_lc/uc provided by gpu_common.cl */

/* ---- Bare MD5 iteration kernel (e1 lowercase, e2 uppercase) ----
 *
 * Input: hex(MD5(password)) pre-packed by CPU into M[] words.
 * GPU iterates: MD5(hex) -> check -> hex(result) -> MD5(hex) -> ...
 * One work item per word (no salt dimension).
 * Hit stride is always 7 (includes iteration number).
 */
__kernel void md5_iter_lc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    /* MD5 padding for 32-byte input -- constant across all iterations */
    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            EMIT_HIT_4(hits, hit_count, params.max_hits, tid, 0, iter + 1, hx, hy, hz, hw)
        }

        if (iter < params.max_iter) {
            md5_to_hex_lc(hx, hy, hz, hw, M);
            /* M[8..15] unchanged: padding for 32-byte input */
        }
    }
}

__kernel void md5_iter_uc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            EMIT_HIT_4(hits, hit_count, params.max_hits, tid, 0, iter + 1, hx, hy, hz, hw)
        }

        if (iter < params.max_iter) {
            md5_to_hex_uc(hx, hy, hz, hw, M);
        }
    }
}
