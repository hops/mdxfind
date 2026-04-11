/* metal_sha256unsalted.metal — Pre-padded unsalted SHA256/SHA224 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks (word_stride=64).
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0, h1, h2, h3) or
 *             7 (word_idx, mask_idx, iter, h0, h1, h2, h3) when max_iter > 1
 */

static inline uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) |
           ((x << 8) & 0xff0000u) | ((x << 24) & 0xff000000u);
}

constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

/* Convert one byte to 2 packed hex chars (lowercase, BE word packing) */
static uint hex_byte_be(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Convert 8 SHA256 BE state words to 16 BE M[] words of hex text (lowercase).
 * 8 state words x 8 hex chars each = 64 hex chars = 16 M[] words.
 * Fills the entire M[0..15] block -- padding must go in a second block. */
static void sha256_to_hex_lc(thread uint *state, thread uint *M) {
    for (int i = 0; i < 8; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

/* Convert 7 SHA224 BE state words to 14 BE M[] words of hex text (lowercase).
 * 7 state words x 8 hex chars each = 56 hex chars = 14 M[] words.
 * Fits in one block with padding at M[14] and bit-length at M[15]. */
static void sha224_to_hex_lc(thread uint *state, thread uint *M) {
    for (int i = 0; i < 7; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

static void sha256_compress(thread uint *state, thread uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++) {
        uint s0 = rotate(W[i-15], 25u) ^ rotate(W[i-15], 14u) ^ (W[i-15] >> 3);
        uint s1 = rotate(W[i-2], 15u) ^ rotate(W[i-2], 13u) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotate(e, 26u) ^ rotate(e, 21u) ^ rotate(e, 7u);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotate(a, 30u) ^ rotate(a, 19u) ^ rotate(a, 10u);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

kernel void sha256_unsalted_batch(
    device const uint       *words       [[buffer(0)]],
    device const ushort     *unused_lens [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *mask_desc   [[buffer(3)]],
    device const uint       *unused3     [[buffer(4)]],
    device const ushort     *unused4     [[buffer(5)]],
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
    uint word_idx = tid / params.num_masks;
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }
        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    uint bitlen = M[14];
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);
    M[14] = 0;
    M[15] = bitlen;

    uint state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    sha256_compress(state, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint hx = bswap32(state[0]), hy = bswap32(state[1]);
        uint hz = bswap32(state[2]), hw = bswap32(state[3]);

        uint4 h = uint4(hx, hy, hz, hw);
        ulong key = (ulong(h.y) << 32) | h.x;
        uint fp = uint(key >> 32);
        if (fp == 0) fp = 1;
        ulong pos = (key ^ (key >> 32)) & params.compact_mask;
        bool found = false;
        for (uint p = 0; p < params.max_probe && !found; p++) {
            uint cfp = compact_fp[pos];
            if (cfp == 0) break;
            if (cfp == fp) {
                uint idx = compact_idx[pos];
                if (idx < params.hash_data_count) {
                    ulong off = hash_data_off[idx];
                    device const uint *ref = (device const uint *)(hash_data_buf + off);
                    if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3])
                        found = true;
                }
            }
            pos = (pos + 1) & params.compact_mask;
        }
        if (!found && params.overflow_count > 0) {
            int lo = 0, hi2 = int(params.overflow_count) - 1;
            while (lo <= hi2 && !found) {
                int mid = (lo + hi2) / 2;
                ulong mkey = overflow_keys[mid];
                if (key < mkey) hi2 = mid - 1;
                else if (key > mkey) lo = mid + 1;
                else {
                    for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3])
                            found = true;
                    }
                    for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3])
                            found = true;
                    }
                    break;
                }
            }
        }
        if (found) {
            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
            if (slot < params.max_hits) {
                uint base = slot * hit_stride;
                hits[base] = word_idx; hits[base+1] = mask_idx;
                if (hit_stride == 7) {
                    hits[base+2] = iter;
                    hits[base+3] = h.x; hits[base+4] = h.y;
                    hits[base+5] = h.z; hits[base+6] = h.w;
                } else {
                    hits[base+2] = h.x; hits[base+3] = h.y;
                    hits[base+4] = h.z; hits[base+5] = h.w;
                }
            }
        }
        if (iter < max_iter) {
            /* Hex-encode 8 BE state words into M[0..15] (64 hex chars = full block) */
            sha256_to_hex_lc(state, M);
            /* First compress: data block (no room for padding) */
            state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
            state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
            state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
            state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
            sha256_compress(state, M);
            /* Second compress: padding block */
            M[0] = 0x80000000u;
            for (int i = 1; i < 15; i++) M[i] = 0;
            M[15] = 64 * 8;  /* 64 hex bytes = 512 bits */
            sha256_compress(state, M);
        }
    }
}

/* SHA224 — same compress, different IV */
kernel void sha224_unsalted_batch(
    device const uint       *words       [[buffer(0)]],
    device const ushort     *unused_lens [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *mask_desc   [[buffer(3)]],
    device const uint       *unused3     [[buffer(4)]],
    device const ushort     *unused4     [[buffer(5)]],
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
    uint word_idx = tid / params.num_masks;
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }
        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    uint bitlen = M[14];
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);
    M[14] = 0;
    M[15] = bitlen;

    uint state[8] = {
        0xc1059ed8u, 0x367cd507u, 0x3070dd17u, 0xf70e5939u,
        0xffc00b31u, 0x68581511u, 0x64f98fa7u, 0xbefa4fa4u
    };
    sha256_compress(state, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint hx = bswap32(state[0]), hy = bswap32(state[1]);
        uint hz = bswap32(state[2]), hw = bswap32(state[3]);

        uint4 h = uint4(hx, hy, hz, hw);
        ulong key = (ulong(h.y) << 32) | h.x;
        uint fp = uint(key >> 32);
        if (fp == 0) fp = 1;
        ulong pos = (key ^ (key >> 32)) & params.compact_mask;
        bool found = false;
        for (uint p = 0; p < params.max_probe && !found; p++) {
            uint cfp = compact_fp[pos];
            if (cfp == 0) break;
            if (cfp == fp) {
                uint idx = compact_idx[pos];
                if (idx < params.hash_data_count) {
                    ulong off = hash_data_off[idx];
                    device const uint *ref = (device const uint *)(hash_data_buf + off);
                    if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3])
                        found = true;
                }
            }
            pos = (pos + 1) & params.compact_mask;
        }
        if (!found && params.overflow_count > 0) {
            int lo = 0, hi2 = int(params.overflow_count) - 1;
            while (lo <= hi2 && !found) {
                int mid = (lo + hi2) / 2;
                ulong mkey = overflow_keys[mid];
                if (key < mkey) hi2 = mid - 1;
                else if (key > mkey) lo = mid + 1;
                else {
                    for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3])
                            found = true;
                    }
                    for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3])
                            found = true;
                    }
                    break;
                }
            }
        }
        if (found) {
            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
            if (slot < params.max_hits) {
                uint base = slot * hit_stride;
                hits[base] = word_idx; hits[base+1] = mask_idx;
                if (hit_stride == 7) {
                    hits[base+2] = iter;
                    hits[base+3] = h.x; hits[base+4] = h.y;
                    hits[base+5] = h.z; hits[base+6] = h.w;
                } else {
                    hits[base+2] = h.x; hits[base+3] = h.y;
                    hits[base+4] = h.z; hits[base+5] = h.w;
                }
            }
        }
        if (iter < max_iter) {
            /* SHA224: 7 state words -> 56 hex chars -> 14 M[] words, fits in one block */
            sha224_to_hex_lc(state, M);
            M[14] = 0x80000000u;
            M[15] = 56 * 8;  /* 56 hex bytes = 448 bits */
            state[0] = 0xc1059ed8u; state[1] = 0x367cd507u;
            state[2] = 0x3070dd17u; state[3] = 0xf70e5939u;
            state[4] = 0xffc00b31u; state[5] = 0x68581511u;
            state[6] = 0x64f98fa7u; state[7] = 0xbefa4fa4u;
            sha256_compress(state, M);
        }
    }
}
