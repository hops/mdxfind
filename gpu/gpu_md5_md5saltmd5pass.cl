/* ---- MD5(MD5(salt).MD5(pass)) kernel (e367) ---- */
/* Salt buffer has hex(MD5(salt)) [32 bytes], hexhash buffer has hex(MD5(pass)) [32 bytes].
 * Total input is always exactly 64 bytes → deterministic 2-block MD5. */
__kernel void md5_md5saltmd5pass_batch(
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

    /* Load hex(MD5(salt)) [32 bytes] into M[0..7] as LE uint32 words */
    uint M[16];
    { __global const uint *sw = (__global const uint *)(salts + salt_offsets[salt_idx]);
      for (int i = 0; i < 8; i++) M[i] = sw[i];
    }

    /* Load hex(MD5(pass)) [32 bytes] into M[8..15] */
    __global const uint *pass_words = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[8 + i] = pass_words[i];

    /* Block 1: MD5 of the 64 data bytes (no padding in this block) */
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Block 2: padding only — all constants, zero memory access */
    md5_block_pad64(&hx, &hy, &hz, &hw);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
    }
}
