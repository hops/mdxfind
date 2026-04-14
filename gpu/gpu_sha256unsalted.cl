/* gpu_sha256unsalted.cl — Pre-padded unsalted SHA256 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks in passbuf (word_stride=64).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, M[14]=bitlen (LE).
 *   Kernel byte-swaps to big-endian before SHA256 compress.
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 10 (word_idx, mask_idx, h0..h7) or
 *             11 (word_idx, mask_idx, iter, h0..h7) when max_iter > 1
 */


/* bswap32, SHA256_K[], sha256_block(), hex_byte_be64() provided by gpu_common.cl */

/* Convert one byte to 2 packed hex chars (lowercase, BE word packing) */
uint hex_byte_be(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Convert 8 SHA256 BE state words to 16 BE M[] words of hex text (lowercase) */
void sha256_to_hex_lc(uint *state, uint *M) {
    for (int i = 0; i < 8; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

#define sha256_compress sha256_block

__kernel void sha256_unsalted_batch(
    __global const uint *words,
    __global const ushort *unused_lens,
    __global const uchar *mask_desc,
    __global const uint *unused1, __global const ushort *unused2,
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
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions — LE words, same as MD5 kernel */
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
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
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
    /* SHA256: 8 hash words. hit_stride = 2 + 8 = 10, or 3 + 8 = 11 with iter */
    uint hit_stride = (max_iter > 1) ? 11 : 10;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint h[8];
        for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

        if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * hit_stride;
                hits[base]   = word_idx;
                hits[base+1] = mask_idx;
                int offset = 2;
                if (max_iter > 1) { hits[base+2] = iter; offset = 3; }
                for (int i = 0; i < 8; i++) hits[base+offset+i] = h[i];
                mem_fence(CLK_GLOBAL_MEM_FENCE);
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

/* SHA224 — same compress as SHA256, different IV, truncated output (28 bytes) */
__kernel void sha224_unsalted_batch(
    __global const uint *words,
    __global const ushort *unused_lens,
    __global const uchar *mask_desc,
    __global const uint *unused1, __global const ushort *unused2,
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
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
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
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
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

    uint bitlen = M[14];
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);
    M[14] = 0;
    M[15] = bitlen;

    uint state[8] = {
        0xc1059ed8u, 0x367cd507u, 0x3070dd17u, 0xf70e5939u,
        0xffc00b31u, 0x68581511u, 0x64f98fa7u, 0xbefa4fa4u
    };
    sha256_compress(state, M);

    /* SHA224: 7 hash words */
    uint h[7];
    for (int i = 0; i < 7; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 9;  /* 2 + 7 */
            hits[base]   = word_idx;
            hits[base+1] = mask_idx;
            for (int i = 0; i < 7; i++) hits[base+2+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- SHA256RAW: sha256(sha256_binary(pass)) with binary iteration ----
 *
 * For hashcat modes 21400, 30420. With -i N, iterates N times on
 * binary 32-byte output: sha256(sha256(...sha256(pass)...))
 * Second+ iterations always hash exactly 32 bytes (single block, fixed padding).
 * Hit stride: 10 (word_idx, mask_idx, h0..h7) or
 *             11 (word_idx, mask_idx, iter, h0..h7) when max_iter > 1
 */
__kernel void sha256raw_unsalted_batch(
    __global const uint *words,
    __global const ushort *unused_lens,
    __global const uchar *mask_desc,
    __global const uint *unused1, __global const ushort *unused2,
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
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
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
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
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

    /* First: SHA256(password) from pre-padded block */
    uint bitlen_orig = M[14];
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);
    M[14] = 0; M[15] = bitlen_orig;

    uint state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    sha256_compress(state, M);

    /* Binary iteration: sha256(32-byte binary output).
     * Set up fixed padding for 32-byte binary input once. */
    M[8] = 0x80000000u;  /* 0x80 in high byte (BE) */
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = 256;  /* 32 bytes * 8 bits */

    uint max_iter = params.max_iter;
    /* SHA256RAW: 8 hash words. hit_stride = 2 + 8 = 10, or 3 + 8 = 11 with iter */
    uint hit_stride = (max_iter > 1) ? 11 : 10;

    for (uint iter = 1; iter <= max_iter; iter++) {

        uint h[8];
        for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

        if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * hit_stride;
                hits[base]   = word_idx;
                hits[base+1] = mask_idx;
                int offset = 2;
                if (max_iter > 1) { hits[base+2] = iter; offset = 3; }
                for (int i = 0; i < 8; i++) hits[base+offset+i] = h[i];
                mem_fence(CLK_GLOBAL_MEM_FENCE);
            }
        }
        if (iter < max_iter) {
            for (int i = 0; i < 8; i++) M[i] = state[i];
            state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
            state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
            state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
            state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
            sha256_compress(state, M);
        }
    }
}
