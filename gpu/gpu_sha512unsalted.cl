/* gpu_sha512unsalted.cl — Pre-padded unsalted SHA512 with mask expansion
 *
 * Input: 2048 pre-padded 128-byte blocks in passbuf (word_stride=128).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, bitlen as uint32 at byte offset 120.
 *   Kernel byte-swaps 16 × uint64 words to big-endian before SHA512 compress.
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0_lo, h0_hi, h1_lo, h1_hi)
 *
 * Primitives (hex_byte_be64, sha512_to_hex_lc, bswap64, K512, rotr64,
 * sha512_block) provided by gpu_common.cl
 */

__kernel void sha512_unsalted_batch(
    __global const uchar *words,         /* pre-padded 128-byte blocks */
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

    /* Load pre-padded block (16 uint64 = 128 bytes, little-endian from host) */
    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions — LE bytes within uint64 words */
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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
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
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
    sha512_block(state, M);

    uint max_iter = params.max_iter;
    /* SHA512: 16 hash words (8 x uint64 -> 16 x uint32).
     * hit_stride = 2 + 16 = 18, or 3 + 16 = 19 with iter */
    uint hit_stride = (max_iter > 1) ? 19 : 18;

    for (uint iter = 1; iter <= max_iter; iter++) {
        /* Byte-swap all 8 state words to LE, split into 16 uint32 */
        uint h[16];
        for (int i = 0; i < 8; i++) {
            ulong s = bswap64(state[i]);
            h[i*2]   = (uint)s;
            h[i*2+1] = (uint)(s >> 32);
        }

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
                for (int i = 0; i < 16; i++) hits[base+offset+i] = h[i];
                mem_fence(CLK_GLOBAL_MEM_FENCE);
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
            sha512_block(state, M);
            /* Second compress: padding block */
            M[0] = 0x8000000000000000UL;
            for (int i = 1; i < 15; i++) M[i] = 0;
            M[15] = 128 * 8;  /* 128 hex bytes = 1024 bits */
            sha512_block(state, M);
        }
    }
}

/* SHA384 — same compress as SHA512, different IV, truncated output (48 bytes) */
__kernel void sha384_unsalted_batch(
    __global const uchar *words,
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

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
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
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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

    ulong state[8] = {
        0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
        0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
        0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
        0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL
    };
    sha512_block(state, M);

    /* SHA384: 12 hash words (6 x uint64 -> 12 x uint32) */
    uint h[12];
    for (int i = 0; i < 6; i++) {
        ulong s = bswap64(state[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 14;  /* 2 + 12 */
            hits[base]   = word_idx;
            hits[base+1] = mask_idx;
            for (int i = 0; i < 12; i++) hits[base+2+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* SHA384RAW: sha384(sha384_binary(pass)) — binary iteration.
 * First SHA384(pass), then iterate sha384(48-byte binary).
 * Hit stride: 14 (word_idx, mask_idx, h0..h11) or
 *             15 (word_idx, mask_idx, iter, h0..h11) when max_iter > 1
 */
__kernel void sha384raw_unsalted_batch(
    __global const uchar *words, __global const ushort *unused_lens,
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

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend, n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++) append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = i >> 3; int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
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
                    int wi = pos >> 3; int bi = (pos & 7) << 3;
                    M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                int wi = pos >> 3; int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);
    M[14] = 0; M[15] = (ulong)bitlen;

    /* SHA384 IV */
    ulong state[8] = {
        0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
        0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
        0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
        0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL
    };
    sha512_block(state, M);

    /* Binary iteration: probe first, then sha384(48-byte binary) for next iter.
     * Set up fixed padding for 48-byte binary input once. */
    M[6] = 0x8000000000000000UL;
    for (int i = 7; i < 15; i++) M[i] = 0;
    M[15] = 384;  /* 48 bytes * 8 bits */

    uint max_iter = params.max_iter;
    /* SHA384RAW: 12 hash words. hit_stride = 2 + 12 = 14, or 3 + 12 = 15 with iter */
    uint hit_stride = (max_iter > 1) ? 15 : 14;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint h[12];
        for (int i = 0; i < 6; i++) {
            ulong s = bswap64(state[i]);
            h[i*2]   = (uint)s;
            h[i*2+1] = (uint)(s >> 32);
        }

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
                for (int i = 0; i < 12; i++) hits[base+offset+i] = h[i];
                mem_fence(CLK_GLOBAL_MEM_FENCE);
            }
        }
        if (iter < max_iter) {
            for (int i = 0; i < 6; i++) M[i] = state[i];
            state[0] = 0xcbbb9d5dc1059ed8UL; state[1] = 0x629a292a367cd507UL;
            state[2] = 0x9159015a3070dd17UL; state[3] = 0x152fecd8f70e5939UL;
            state[4] = 0x67332667ffc00b31UL; state[5] = 0x8eb44a8768581511UL;
            state[6] = 0xdb0c2e0d64f98fa7UL; state[7] = 0x47b5481dbefa4fa4UL;
            sha512_block(state, M);
        }
    }
}

/* SHA512RAW: sha512(sha512_binary(pass)) — binary iteration.
 * First SHA512(pass), then iterate sha512(64-byte binary).
 * Hit stride: 18 (word_idx, mask_idx, h0..h15) or
 *             19 (word_idx, mask_idx, iter, h0..h15) when max_iter > 1
 */
__kernel void sha512raw_unsalted_batch(
    __global const uchar *words, __global const ushort *unused_lens,
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

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend, n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++) append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = i >> 3; int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
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
                    int wi = pos >> 3; int bi = (pos & 7) << 3;
                    M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                int wi = pos >> 3; int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);
    M[14] = 0; M[15] = (ulong)bitlen;

    ulong state[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
        0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
        0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
    };
    sha512_block(state, M);

    /* Binary iteration: probe at each iteration, then sha512(binary_output).
     * Set up fixed padding for 64-byte binary input once. */
    M[8] = 0x8000000000000000UL;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = 512;  /* 64 bytes * 8 bits */

    uint max_iter = params.max_iter;
    /* SHA512RAW: 16 hash words. hit_stride = 2 + 16 = 18, or 3 + 16 = 19 with iter */
    uint hit_stride = (max_iter > 1) ? 19 : 18;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint h[16];
        for (int i = 0; i < 8; i++) {
            ulong s = bswap64(state[i]);
            h[i*2]   = (uint)s;
            h[i*2+1] = (uint)(s >> 32);
        }

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
                for (int i = 0; i < 16; i++) hits[base+offset+i] = h[i];
                mem_fence(CLK_GLOBAL_MEM_FENCE);
            }
        }
        if (iter < max_iter) {
            for (int i = 0; i < 8; i++) M[i] = state[i];
            state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;
            state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;
            state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;
            state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;
            sha512_block(state, M);
        }
    }
}
