/* ---- PHPBB3/phpass kernel (e455): iterated MD5 with 8-byte salt ---- */
/* Salt buffer has "$H$Csalt8chr" (12 bytes). C encodes log2(iteration count).
 * Algorithm: MD5(salt[4..11] + pass), then 2^C iterations of MD5(hash + pass).
 * Password max 39 bytes (16 + 39 = 55, single MD5 block for iteration). */
kernel void phpbb3_batch(
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
    if (plen > 39) return;
    device const uint8_t *pass = hexhashes + word_idx * 256;

    uint soff = salt_offsets[salt_idx];
    device const uint8_t *sd = salts + soff;

    /* Iteration count passed uniformly via params (salts sorted by count on CPU) */
    uint count = params.iter_count;

    /* Step 1: MD5(salt[4..11] + password) — 8-byte salt */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;
    for (int i = 0; i < 8; i++)
        M[i >> 2] |= ((uint)sd[4 + i]) << ((i & 3) << 3);
    for (int i = 0; i < plen; i++) {
        int pos = 8 + i;
        M[pos >> 2] |= ((uint)pass[i]) << ((pos & 3) << 3);
    }
    int total = 8 + plen;
    M[total >> 2] |= 0x80u << ((total & 3) << 3);
    M[14] = total * 8;

    uint4 h = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block_full(h, M);

    /* Step 2: count iterations of MD5(hash[16] + password) */
    /* Pre-pack M[] with password, padding, length — constant across iterations */
    total = 16 + plen;
    for (int i = 0; i < 16; i++) M[i] = 0;
    for (int i = 0; i < plen; i++) {
        int pos = 16 + i;
        M[pos >> 2] |= ((uint)pass[i]) << ((pos & 3) << 3);
    }
    M[total >> 2] |= 0x80u << ((total & 3) << 3);
    M[14] = total * 8;
    /* Iteration loop: only M[0..3] change (the hash digest) */
    for (uint it = 0; it < count; it++) {
        M[0] = h.x; M[1] = h.y; M[2] = h.z; M[3] = h.w;
        h = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
        md5_block_full(h, M);
    }

    PROBE6(h, word_idx, salt_idx)
}
