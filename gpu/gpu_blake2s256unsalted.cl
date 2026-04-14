/* gpu_blake2s256unsalted.cl — Pre-padded unsalted BLAKE2S-256 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte blocks in passbuf (word_stride=64).
 *   Packed by gpu_try_pack_unsalted(): password bytes at offset n_prepend,
 *   total_len stored as uint16 at bytes [56..57] (after the password area).
 *   BLAKE2S uses its own padding (no 0x80/bitlen), so the packer just stores
 *   the password and length — the kernel handles BLAKE2S-specific finalization.
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0, h1, h2, h3)
 *
 * Primitives (B2S_IV, B2S_SIGMA, b2s_compress) provided by gpu_common.cl
 */

__kernel void blake2s256_unsalted_batch(
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

    /* Load pre-packed block (16 uint32 = 64 bytes).
     * Password bytes start at n_prepend. Total length at M[14] bits [7:0] and [15:8]. */
    __global const uint *src = words + word_idx * 16;
    uchar buf[64];
    for (int i = 0; i < 64; i++)
        buf[i] = ((__global const uchar *)src)[i];

    /* Fill mask positions */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    /* Read total_len from packed M[14] (stored as bitlen by packer, extract byte len) */
    uint M14 = src[14];
    int total_len = M14 >> 3;

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
                buf[i] = ch;
            }

            if (n_app > 0) {
                int app_start = total_len - (int)n_app;
                uint aidx = append_idx;
                for (int i = (int)n_app - 1; i >= 0; i--) {
                    int pos_idx = n_pre + i;
                    uint sz = mask_desc[pos_idx];
                    uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                    aidx /= sz;
                    buf[app_start + i] = ch;
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
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
                buf[app_start + i] = ch;
            }
        }
    }

    /* BLAKE2S-256: single-block hash (password <= 55 bytes fits in one 64-byte block).
     * Init: IV ^ parameter block (outlen=32, keylen=0, fanout=1, depth=1). */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
    h[0] ^= 0x01010020u;

    /* Zero bytes after password */
    for (int i = total_len; i < 64; i++) buf[i] = 0;

    /* Single compress with last=1, counter=total_len */
    b2s_compress(h, buf, (ulong)total_len, 1);

    /* BLAKE2S-256: 8 hash words, output is LE — no bswap */
    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 10;  /* 2 + 8 */
            hits[base]   = word_idx;
            hits[base+1] = mask_idx;
            for (int i = 0; i < 8; i++) hits[base+2+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
