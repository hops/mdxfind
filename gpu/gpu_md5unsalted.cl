/* gpu_md5unsalted.cl — Pre-padded unsalted MD5 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks in passbuf (word_stride=64).
 *   Each block has: password at offset n_prepend, 0x80 padding, M[14]=bitlen.
 *   Prepend/append gaps are zeroed — GPU fills them per mask combination.
 *
 * Dispatch: num_words × num_masks threads.
 *   word_idx = tid / num_masks, mask_idx = mask_start + (tid % num_masks)
 *
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw) or
 *             7 (word_idx, mask_idx, iter, hx, hy, hz, hw) when max_iter > 1
 */

/* Primitives (hex_byte_lc, md5_to_hex_lc, md5_block) provided by gpu_common.cl */

__kernel void md5_unsalted_batch(
    __global const uint *words,          /* pre-padded M[] blocks, 16 uint32 each */
    __global const ushort *unused_lens,  /* not used — total length is in M[14]/8 */
    __global const uchar *mask_desc,     /* mask descriptor: [prepend IDs][append IDs] */
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

    /* Load pre-padded M[] block (16 uint32 = 64 bytes) */
    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions into M[] */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            /* Prepend+append: split mask_idx into prepend and append indices.
             * append_combos fits uint32 for practical -N/-n combinations. */
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++)
                append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);

            /* Fill prepend chars at byte offset 0 */
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }

            /* Fill append chars */
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

    /* MD5 compress (single block, fully unrolled) */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Probe compact table at iteration 1 and emit hit */
    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, mask_idx, 1u, hx, hy, hz, hw)
        }
        if (iter < max_iter) {
            /* Hex-encode hash into M[0..7], set up constant padding */
            md5_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;  /* 32 hex bytes = 256 bits */
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md5_block(&hx, &hy, &hz, &hw, M);
        }
    }
}

/* MD5RAW: md5(md5_binary(pass)) — binary iteration, single block.
 * Hit stride: 6 */
__kernel void md5raw_unsalted_batch(
    __global const uint *words, __global const ushort *unused_lens,
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

    uint n_pre = params.n_prepend, n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++) append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = (uint)(mask_idx / append_combos);
        uint append_idx = (uint)(mask_idx % append_combos);
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3))) | ((uint)ch << ((i & 3) << 3));
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
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3))) | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    /* First MD5 of password */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Binary iteration: md5(md5_binary(prev)) for each iteration.
     * Set up fixed padding for 16-byte binary input once. */
    M[4] = 0x80;
    for (int i = 5; i < 14; i++) M[i] = 0;
    M[14] = 128; M[15] = 0;  /* 16 bytes = 128 bits */

    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        /* Binary re-hash: md5(previous_hash_binary) */
        M[0] = hx; M[1] = hy; M[2] = hz; M[3] = hw;
        hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, mask_idx, 1u, hx, hy, hz, hw)
        }
    }
}
