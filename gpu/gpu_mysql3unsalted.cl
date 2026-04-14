/* gpu_mysql3unsalted.cl — MySQL OLD_PASSWORD() hash (e456, hashcat 200)
 *
 * Simple arithmetic hash: nr/nr2 accumulator loop over password bytes.
 * Uses pre-padded 64-byte block for consistent mask fill — password length
 * from M[14]/8, password bytes at LE offsets 0..total-1.
 * Skips spaces and tabs (matching CPU behavior).
 *
 * Output: 8 bytes (nr[31:0] || nr2[31:0]), big-endian.
 * Compact probe: first 4 bytes split into 2 × uint32 (nr_hi, nr_lo, nr2_hi, nr2_lo)
 * Actually: output is only 64 bits so we use nr and nr2 as the 4 probe words.
 *
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */


__kernel void mysql3_unsalted_batch(
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

    /* Load pre-padded block */
    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions — same LE byte manipulation */
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

    /* MySQL3 hash: arithmetic loop over password bytes */
    int passlen = M[14] >> 3;  /* total length from bit-length field */
    uint nr = 1345345333u, add = 7u, nr2 = 0x12345671u;

    for (int i = 0; i < passlen; i++) {
        /* Extract byte i from LE M[] */
        uchar c = (uchar)((M[i >> 2] >> ((i & 3) << 3)) & 0xFFu);
        if (c == ' ' || c == '\t') continue;
        uint tmp = (uint)c;
        nr ^= (((nr & 63u) + add) * tmp) + (nr << 8);
        nr2 += (nr2 << 8) ^ nr;
        add += tmp;
    }
    nr &= 0x7fffffffu;
    nr2 &= 0x7fffffffu;

    /* Output: 8 bytes BE = nr(4 bytes) || nr2(4 bytes)
     * For compact table probe, we need 4 × uint32.
     * Hash is only 64 bits; store nr and nr2 in LE byte order
     * to match how mdxfind stores 16-hex hashes in the compact table. */
    uint hx = ((nr >> 24) & 0xff) | ((nr >> 8) & 0xff00) |
              ((nr << 8) & 0xff0000) | ((nr << 24) & 0xff000000u);
    uint hy = ((nr2 >> 24) & 0xff) | ((nr2 >> 8) & 0xff00) |
              ((nr2 << 8) & 0xff0000) | ((nr2 << 24) & 0xff000000u);
    uint hz = 0, hw = 0;

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, mask_idx, 1u, hx, hy, hz, hw)
    }
}
