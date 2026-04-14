/* metal_md4unsalted.metal — Pre-padded unsalted MD4 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks (word_stride=64).
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw) or
 *             7 (word_idx, mask_idx, iter, hx, hy, hz, hw) when max_iter > 1
 */

/* hex_byte_lc, md5_to_hex_lc provided by metal_common.metal */
/* md4_to_hex_lc is identical to md5_to_hex_lc */
#define md4_to_hex_lc md5_to_hex_lc

/* Fully unrolled MD4 compress — RFC 1320
 * Round 1: F(b,c,d) = (b&c)|(~b&d),       no constant
 * Round 2: G(b,c,d) = (b&c)|(b&d)|(c&d),  constant 0x5A827999
 * Round 3: H(b,c,d) = b^c^d,               constant 0x6ED9EBA1
 */
static void md4_compress(thread uint *hx, thread uint *hy, thread uint *hz, thread uint *hw, thread uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
    /* Round 1: F(b,c,d) = (b&c)|(~b&d), rotations: 3,7,11,19 */
    a += ((b&c)|(~b&d)) + M[ 0]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 1]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[ 2]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[ 3]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[ 4]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 5]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[ 6]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[ 7]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[ 8]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 9]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[10]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[11]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[12]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[13]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[14]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[15]; b = ((b<<19)|(b>>13));
    /* Round 2: G(b,c,d) = (b&c)|(b&d)|(c&d), constant 0x5A827999
     * Message order: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15 */
    a += ((b&c)|(b&d)|(c&d)) + M[ 0] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 4] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[ 8] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[12] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 1] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 5] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[ 9] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[13] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 2] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 6] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[10] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[14] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 3] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 7] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[11] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[15] + 0x5A827999u; b = ((b<<13)|(b>>19));
    /* Round 3: H(b,c,d) = b^c^d, constant 0x6ED9EBA1
     * Message order: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15 */
    a += (b^c^d) + M[ 0] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[ 8] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 4] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[12] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 2] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[10] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 6] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[14] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 1] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[ 9] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 5] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[13] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 3] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[11] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 7] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[15] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    *hx += a; *hy += b; *hz += c; *hw += d;
}

kernel void md4_unsalted_batch(
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
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++)
                append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);

            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
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
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint local_idx = tid % params.num_masks;
            uint aidx = local_idx;
            uint carry = 0;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint local_digit = aidx % sz;
                aidx /= sz;
                uint base_digit = (i < 8)
                    ? (uint)((params.mask_base0 >> (i * 8)) & 0xFF)
                    : (uint)((params.mask_base1 >> ((i - 8) * 8)) & 0xFF);
                uint sum = base_digit + local_digit + carry;
                carry = sum / sz;
                uint final_digit = sum % sz;
                uchar ch = mask_desc[n_total_m + i * 256 + final_digit];
                int pos = app_start + i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    /* MD4 compress (single block, fully unrolled) */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md4_compress(&hx, &hy, &hz, &hw, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
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
            md4_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md4_compress(&hx, &hy, &hz, &hw, M);
        }
    }
}

/* MD4UTF16 (NTLM-style): MD4 of UTF-16LE zero-extended password.
 * Host packs UTF-16LE (each byte → byte,0x00) with doubled offsets.
 * Mask positions are doubled: logical char i → byte 2*i in the block. */
kernel void md4utf16_unsalted_batch(
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
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++)
                append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);

            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int pos = 2 * i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }

            if (n_app > 0) {
                int total_bytes = M[14] >> 3;
                int app_start = total_bytes - 2 * (int)n_app;
                uint aidx = append_idx;
                for (int i = (int)n_app - 1; i >= 0; i--) {
                    int pos_idx = n_pre + i;
                    uint sz = mask_desc[pos_idx];
                    uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                    aidx /= sz;
                    int pos = app_start + 2 * i;
                    M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                                 | ((uint)ch << ((pos & 3) << 3));
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
            int total_bytes = M[14] >> 3;
            int app_start = total_bytes - 2 * (int)n_app;
            uint local_idx = tid % params.num_masks;
            uint aidx = local_idx;
            uint carry = 0;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint local_digit = aidx % sz;
                aidx /= sz;
                uint base_digit = (i < 8)
                    ? (uint)((params.mask_base0 >> (i * 8)) & 0xFF)
                    : (uint)((params.mask_base1 >> ((i - 8) * 8)) & 0xFF);
                uint sum = base_digit + local_digit + carry;
                carry = sum / sz;
                uint final_digit = sum % sz;
                uchar ch = mask_desc[n_total_m + i * 256 + final_digit];
                int pos = app_start + 2 * i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md4_compress(&hx, &hy, &hz, &hw, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
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
            /* Iteration uses plain MD4 on hex text (not UTF16) */
            md4_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md4_compress(&hx, &hy, &hz, &hw, M);
        }
    }
}
