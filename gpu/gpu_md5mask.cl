/* ---- MD5 with mask expansion (GPU_CAT_MASK) ---- */
/* Mask descriptor buffer: [n_prepend charset IDs] [n_append charset IDs]
 * Charset IDs: 0=?d(10) 1=?l(26) 2=?u(26) 3=?s(33) 4=?a(95) 5=?b(256)
 * Each work item: tid / num_masks = word_idx, tid % num_masks = mask_idx
 * mask_idx is decomposed: prepend_idx × append_combos + append_idx */

/* cs_a and cs_b built dynamically to save constant memory */

__kernel void md5_mask_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *mask_desc, __global const uint *unused1, __global const ushort *unused2,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    int baselen = hexlens[word_idx];
    __global const uchar *base = hexhashes + word_idx * 256;

    /* Build candidate: [prepend chars] [base word] [append chars] */
    uchar cand[128];
    int clen = 0;

    /* Decode prepend: mask_idx split into prepend_idx and append_idx */
    uint n_total = params.n_prepend + params.n_append;
    uint append_combos = 1;
    for (uint i = 0; i < params.n_append; i++)
        append_combos *= mask_desc[params.n_prepend + i];

    uint prepend_idx = (uint)(mask_idx / append_combos);
    uint append_idx = (uint)(mask_idx % append_combos);

    /* Generate prepend chars (right to left in the index, left to right in output) */
    if (params.n_prepend > 0) {
        uint pidx = prepend_idx;
        for (int i = params.n_prepend - 1; i >= 0; i--) {
            uint sz = mask_desc[i];
            cand[i] = mask_desc[n_total + i * 256 + (pidx % sz)];
            pidx /= sz;
        }
        clen = params.n_prepend;
    }

    /* Copy base word */
    for (int i = 0; i < baselen; i++)
        cand[clen + i] = base[i];
    clen += baselen;

    /* Generate append chars */
    if (params.n_prepend > 0) {
        /* Prepend+append: use pre-computed append_idx */
        if (params.n_append > 0) {
            uint aidx = append_idx;
            for (int i = params.n_append - 1; i >= 0; i--) {
                int pos_idx = params.n_prepend + i;
                uint sz = mask_desc[pos_idx];
                cand[clen + i] = mask_desc[n_total + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
            }
            clen += params.n_append;
        }
    } else if (params.n_append > 0) {
        /* Append-only (brute-force): host pre-decomposes mask_start into
         * per-position base offsets. Kernel does fast uint32 local
         * decomposition and adds to base with carry. */
        uint local_idx = tid % params.num_masks;
        uint aidx = local_idx;
        uint carry = 0;
        for (int i = params.n_append - 1; i >= 0; i--) {
            int pos_idx = params.n_prepend + i;
            uint sz = mask_desc[pos_idx];
            uint local_digit = aidx % sz;
            aidx /= sz;
            uint base_digit = (i < 8)
                ? (uint)((params.mask_base0 >> (i * 8)) & 0xFF)
                : (uint)((params.mask_base1 >> ((i - 8) * 8)) & 0xFF);
            uint sum = base_digit + local_digit + carry;
            carry = sum / sz;
            uint final_digit = sum % sz;
            cand[clen + i] = mask_desc[n_total + pos_idx * 256 + final_digit];
        }
        clen += params.n_append;
    }

    /* MD5(candidate) */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    if (clen <= 55) {
        /* Single block */
        for (int i = 0; i < clen; i++)
            M[i >> 2] |= ((uint)cand[i]) << ((i & 3) << 3);
        M[clen >> 2] |= 0x80u << ((clen & 3) << 3);
        M[14] = clen * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
    } else {
        /* Two blocks */
        for (int i = 0; i < 64 && i < clen; i++)
            M[i >> 2] |= ((uint)cand[i]) << ((i & 3) << 3);
        md5_block(&hx, &hy, &hz, &hw, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        for (int i = 64; i < clen; i++) {
            int j = i - 64;
            M[j >> 2] |= ((uint)cand[i]) << ((j & 3) << 3);
        }
        int rem = clen - 64;
        M[rem >> 2] |= 0x80u << ((rem & 3) << 3);
        M[14] = clen * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, mask_idx, 1u, hx, hy, hz, hw)
    }
}
