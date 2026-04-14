/* ---- MD5(MD5(salt).MD5(pass)) kernel (e367) ---- */
/* Salt buffer has hex(MD5(salt)) [32 bytes], hexhash has hex(MD5(pass)) [32 bytes].
 * Always 64 bytes → deterministic 2-block MD5. */
kernel void md5_md5saltmd5pass_batch(
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

    uint M[16];
    /* Load hex(MD5(salt)) [32 bytes] into M[0..7] */
    device const uint *sw = (device const uint *)(salts + salt_offsets[salt_idx]);
    for (int i = 0; i < 8; i++) M[i] = sw[i];
    /* Load hex(MD5(pass)) [32 bytes] into M[8..15] */
    device const uint *pw = (device const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[8+i] = pw[i];

    uint4 h = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block_full(h, M);

    md5_block_pad64(h);

    PROBE6(h, word_idx, salt_idx)
}

