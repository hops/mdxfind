/* ---- MD5(salt + password) kernel ---- */
kernel void md5saltpass_batch(
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
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    if (total_len <= 55) {
        for (int i = 0; i < slen; i++)
            M[i >> 2] |= ((uint)salts[soff + i]) << ((i & 3) << 3);
        for (int i = 0; i < plen; i++) {
            int pos = slen + i;
            M[pos >> 2] |= ((uint)pass[i]) << ((pos & 3) << 3);
        }
        M[total_len >> 2] |= 0x80u << ((total_len & 3) << 3);
        M[14] = total_len * 8;
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += 0x67452301; hy += 0xEFCDAB89; hz += 0x98BADCFE; hw += 0x10325476;
    } else {
        /* Two blocks */
        /* Fill first 64 bytes from salt+pass */
        int salt_b1 = (slen < 64) ? slen : 64;
        for (int i = 0; i < salt_b1; i++)
            M[i >> 2] |= ((uint)salts[soff + i]) << ((i & 3) << 3);
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        for (int i = 0; i < pass_b1; i++) {
            int pos = salt_b1 + i;
            M[pos >> 2] |= ((uint)pass[i]) << ((pos & 3) << 3);
        }
        if (total_len < 64)
            M[total_len >> 2] |= 0x80u << ((total_len & 3) << 3);
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += 0x67452301; hy += 0xEFCDAB89; hz += 0x98BADCFE; hw += 0x10325476;
        /* Second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        for (int i = 0; i < salt_b2; i++, pos2++)
            M[pos2 >> 2] |= ((uint)salts[soff + salt_b1 + i]) << ((pos2 & 3) << 3);
        int pass_b2 = plen - pass_b1;
        for (int i = 0; i < pass_b2; i++, pos2++)
            M[pos2 >> 2] |= ((uint)pass[pass_b1 + i]) << ((pos2 & 3) << 3);
        if (total_len >= 64)
            M[pos2 >> 2] |= 0x80u << ((pos2 & 3) << 3);
        M[14] = total_len * 8;
        uint sx = hx, sy = hy, sz = hz, sw = hw;
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += sx; hy += sy; hz += sz; hw += sw;
    }

    uint4 hv = uint4(hx, hy, hz, hw);

    PROBE6(hv, word_idx, salt_idx)
}

/* ---- MD5(password + salt) kernel ---- */
kernel void md5passsalt_batch(
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
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    if (total_len <= 55) {
        for (int i = 0; i < plen; i++)
            M[i >> 2] |= ((uint)pass[i]) << ((i & 3) << 3);
        for (int i = 0; i < slen; i++) {
            int pos = plen + i;
            M[pos >> 2] |= ((uint)salts[soff + i]) << ((pos & 3) << 3);
        }
        M[total_len >> 2] |= 0x80u << ((total_len & 3) << 3);
        M[14] = total_len * 8;
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += 0x67452301; hy += 0xEFCDAB89; hz += 0x98BADCFE; hw += 0x10325476;
    } else {
        /* Two blocks: fill first 64 bytes from pass+salt */
        int pass_b1 = (plen < 64) ? plen : 64;
        for (int i = 0; i < pass_b1; i++)
            M[i >> 2] |= ((uint)pass[i]) << ((i & 3) << 3);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        for (int i = 0; i < salt_b1; i++) {
            int pos = pass_b1 + i;
            M[pos >> 2] |= ((uint)salts[soff + i]) << ((pos & 3) << 3);
        }
        if (total_len < 64)
            M[total_len >> 2] |= 0x80u << ((total_len & 3) << 3);
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += 0x67452301; hy += 0xEFCDAB89; hz += 0x98BADCFE; hw += 0x10325476;
        /* Second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        for (int i = 0; i < pass_b2; i++, pos2++)
            M[pos2 >> 2] |= ((uint)pass[pass_b1 + i]) << ((pos2 & 3) << 3);
        int salt_b2 = slen - salt_b1;
        for (int i = 0; i < salt_b2; i++, pos2++)
            M[pos2 >> 2] |= ((uint)salts[soff + salt_b1 + i]) << ((pos2 & 3) << 3);
        if (total_len >= 64)
            M[pos2 >> 2] |= 0x80u << ((pos2 & 3) << 3);
        M[14] = total_len * 8;
        uint sx = hx, sy = hy, sz = hz, sw = hw;
        for (int i = 0; i < 64; i++) {
            uint f, g;
            if (i < 16)      { f = (hy & hz) | (~hy & hw); g = i; }
            else if (i < 32) { f = (hw & hy) | (~hw & hz); g = (5*i+1) & 15; }
            else if (i < 48) { f = hy ^ hz ^ hw;            g = (3*i+5) & 15; }
            else              { f = hz ^ (~hw | hy);         g = (7*i)   & 15; }
            f = f + hx + K[i] + M[g];
            hx = hw; hw = hz; hz = hy;
            hy = hy + ((f << S[i]) | (f >> (32 - S[i])));
        }
        hx += sx; hy += sy; hz += sz; hw += sw;
    }

    uint4 hv = uint4(hx, hy, hz, hw);

    PROBE6(hv, word_idx, salt_idx)
}

