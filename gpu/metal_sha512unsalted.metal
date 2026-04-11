/* metal_sha512unsalted.metal — Pre-padded unsalted SHA512/SHA384 with mask expansion
 *
 * Input: 2048 pre-padded 128-byte blocks in passbuf (word_stride=128).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, bitlen as uint32 at byte offset 120.
 *   Kernel byte-swaps 16 x uint64 words to big-endian before SHA512 compress.
 *
 * Dispatch: num_words x num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0_lo, h0_hi, h1_lo, h1_hi) or
 *             7 (word_idx, mask_idx, iter, h0_lo, h0_hi, h1_lo, h1_hi) when max_iter > 1
 */

static inline ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL) | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL) | ((x >> 8) & 0xff000000UL) |
           ((x << 8) & 0xff00000000UL) | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL) | ((x << 56) & 0xff00000000000000UL);
}

static inline ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

constant ulong K512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

/* Convert one byte to 2 packed hex chars (lowercase, BE ulong packing) */
static ulong hex_byte_be64(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((ulong)(hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (ulong)(lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Convert 8 SHA512 BE ulong state words to 16 BE ulong M[] words of hex text.
 * Each state word (8 bytes BE) -> 16 hex chars -> 2 BE ulong M[] words.
 * Fills the entire M[0..15] block -- padding must go in a second block. */
static void sha512_to_hex_lc(thread ulong *state, thread ulong *M) {
    for (int i = 0; i < 8; i++) {
        ulong s = state[i];
        uint b0 = (s >> 56) & 0xff, b1 = (s >> 48) & 0xff;
        uint b2 = (s >> 40) & 0xff, b3 = (s >> 32) & 0xff;
        uint b4 = (s >> 24) & 0xff, b5 = (s >> 16) & 0xff;
        uint b6 = (s >> 8)  & 0xff, b7 = s & 0xff;
        M[i*2]   = (hex_byte_be64(b0) << 48) | (hex_byte_be64(b1) << 32)
                  | (hex_byte_be64(b2) << 16) | hex_byte_be64(b3);
        M[i*2+1] = (hex_byte_be64(b4) << 48) | (hex_byte_be64(b5) << 32)
                  | (hex_byte_be64(b6) << 16) | hex_byte_be64(b7);
    }
}

static void sha512_compress(thread ulong *state, thread ulong *M) {
    ulong W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++) {
        ulong s0 = rotr64(W[i-15], 1) ^ rotr64(W[i-15], 8) ^ (W[i-15] >> 7);
        ulong s1 = rotr64(W[i-2], 19) ^ rotr64(W[i-2], 61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    ulong a = state[0], b = state[1], c = state[2], d = state[3];
    ulong e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 80; i++) {
        ulong S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        ulong ch = (e & f) ^ (~e & g);
        ulong t1 = h + S1 + ch + K512[i] + W[i];
        ulong S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        ulong maj = (a & b) ^ (a & c) ^ (b & c);
        ulong t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

kernel void sha512_unsalted_batch(
    device const uchar      *words       [[buffer(0)]],
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

    /* Load pre-padded block (16 uint64 = 128 bytes, little-endian from host) */
    device const ulong *src = (device const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions -- LE bytes within uint64 words */
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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }

        if (n_app > 0) {
            /* Bit-length stored at byte offset 120 = low 32 bits of M[15] (LE) */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + i;
                int wi = pos >> 3;
                int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    /* Save bit-length before byte-swap.
     * Host stores bitlen as uint32 at byte offset 120 = low 32 bits of M[15] (LE). */
    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);

    /* Convert M[] from little-endian to big-endian for SHA512 */
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);

    /* Fix up padding: SHA512 stores 128-bit big-endian bit count at M[14..15] */
    M[14] = 0;
    M[15] = (ulong)bitlen;

    /* SHA512 compress */
    ulong state[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
        0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
        0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
    };
    sha512_compress(state, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
        /* Byte-swap state to LE for compact table probe */
        ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
        uint hx = (uint)s0, hy = (uint)(s0 >> 32);
        uint hz = (uint)s1, hw = (uint)(s1 >> 32);

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
            /* Hex-encode 8 BE state words into M[0..15] (128 hex chars = full block) */
            sha512_to_hex_lc(state, M);
            /* First compress: data block */
            state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;
            state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;
            state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;
            state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;
            sha512_compress(state, M);
            /* Second compress: padding block */
            M[0] = 0x8000000000000000UL;
            for (int i = 1; i < 15; i++) M[i] = 0;
            M[15] = 128 * 8;  /* 128 hex bytes = 1024 bits */
            sha512_compress(state, M);
        }
    }
}

/* SHA384 -- same compress as SHA512, different IV, truncated output (48 bytes) */
kernel void sha384_unsalted_batch(
    device const uchar      *words       [[buffer(0)]],
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

    device const ulong *src = (device const ulong *)(words + word_idx * 128);
    ulong M[16];
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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
        if (n_app > 0) {
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + i;
                int wi = pos >> 3;
                int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);
    M[14] = 0;
    M[15] = (ulong)bitlen;

    /* SHA384 IV */
    ulong state[8] = {
        0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
        0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
        0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
        0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL
    };
    sha512_compress(state, M);

    ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
    uint hx = (uint)s0, hy = (uint)(s0 >> 32);
    uint hz = (uint)s1, hw = (uint)(s1 >> 32);

    uint4 h = uint4(hx, hy, hz, hw);
    ulong key = (ulong(h.y) << 32) | h.x;
    uint fp = uint(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & params.compact_mask;
    for (uint p = 0; p < params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base] = word_idx; hits[base+1] = mask_idx;
                        hits[base+2] = h.x; hits[base+3] = h.y;
                        hits[base+4] = h.z; hits[base+5] = h.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    if (params.overflow_count > 0) {
        int lo = 0, hi = int(params.overflow_count) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base] = word_idx; hits[base+1] = mask_idx;
                        hits[base+2] = h.x; hits[base+3] = h.y;
                        hits[base+4] = h.z; hits[base+5] = h.w;
                    }
                    return;
                }
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = mask_idx;
                            hits[base+2] = h.x; hits[base+3] = h.y;
                            hits[base+4] = h.z; hits[base+5] = h.w;
                        }
                        return;
                    }
                }
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = mask_idx;
                            hits[base+2] = h.x; hits[base+3] = h.y;
                            hits[base+4] = h.z; hits[base+5] = h.w;
                        }
                        return;
                    }
                }
                break;
            }
        }
    }
}
