/* gpu_md6256unsalted.cl — Pre-packed unsalted MD6-256 with mask expansion
 *
 * Input: pre-packed N[89] (712-byte) blocks in passbuf.
 *   Host packs: Q[15] + K[8]=0 + U[1] + V[1] + B[64] with message in LE.
 *   Kernel byte-swaps B[64] portion to big-endian before compression.
 *   Mask positions fill B portion at doubled-and-swapped byte offsets.
 *
 * MD6-256: r=104 rounds, c=16, n=89, d=256.
 * Compression loop: 104*16 = 1664 steps.
 * Output: last 4 words of C[16] (= last d/w = 4 × uint64).
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */


/* bswap64 provided by gpu_common.cl */

/* MD6 compression loop constants */
#define MD6_n  89
#define MD6_c  16
#define MD6_r  104

/* Tap positions for feedback shift register */
#define MD6_t0  17
#define MD6_t1  18
#define MD6_t2  21
#define MD6_t3  31
#define MD6_t4  67
#define MD6_t5  89

/* Shift amounts for 16 steps per round (w=64) */
__constant int RS[16] = {10, 5,13,10,11,12, 2, 7,14,15, 7,13,11, 7, 6,12};
__constant int LS[16] = {11,24, 9,16,15, 9,27,15, 6, 2,29, 8,15, 5,31, 9};

__constant ulong MD6_S0 = 0x0123456789abcdefUL;
__constant ulong MD6_Smask = 0x7311c2812425cfa0UL;

/* MD6 main compression loop — operates on array A of length r*c+n */
void md6_compress_loop(ulong *A) {
    ulong S = MD6_S0;
    int i = MD6_n;
    for (int j = 0; j < MD6_r * MD6_c; j += MD6_c) {
        for (int step = 0; step < 16; step++) {
            ulong x = S;
            x ^= A[i + step - MD6_t5];
            x ^= A[i + step - MD6_t0];
            x ^= (A[i + step - MD6_t1] & A[i + step - MD6_t2]);
            x ^= (A[i + step - MD6_t3] & A[i + step - MD6_t4]);
            x ^= (x >> RS[step]);
            A[i + step] = x ^ (x << LS[step]);
        }
        S = (S << 1) ^ (S >> 63) ^ (S & MD6_Smask);
        i += 16;
    }
}

__kernel void md6_256_unsalted_batch(
    __global const uchar *words,         /* pre-packed N[89] blocks */
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

    /* Load pre-packed N[89] block.
     * Host packs: N[0..24] = Q+K+U+V (fixed), N[25..88] = B (message in LE).
     * word_stride stored in jobg, passed as upload size. */
    __global const ulong *src = (__global const ulong *)(words + word_idx * 712);

    /* We need a working array A of size r*c+n = 104*16+89 = 1753 words.
     * Load N into the first 89 positions. */
    ulong A[1753];
    for (int i = 0; i < 89; i++) A[i] = src[i];

    /* Fill mask positions into B portion (A[25..88]).
     * B is in LE uint64 from host packer; mask fills at LE byte positions.
     * We byte-swap B after mask fill. */
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
                int wi = 25 + (i >> 3);
                int bi = (i & 7) << 3;
                A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }

            if (n_app > 0) {
                ulong V = A[24];
                uint p_bits = (uint)((V >> 20) & 0xFFFFUL);
                int total_len = (64*64 - (int)p_bits) / 8;
                int app_start = total_len - (int)n_app;
                uint aidx = append_idx;
                for (int i = (int)n_app - 1; i >= 0; i--) {
                    int pos_idx = n_pre + i;
                    uint sz = mask_desc[pos_idx];
                    uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                    aidx /= sz;
                    int pos = app_start + i;
                    int wi = 25 + (pos >> 3);
                    int bi = (pos & 7) << 3;
                    A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            ulong V = A[24];
            uint p_bits = (uint)((V >> 20) & 0xFFFFUL);
            int total_len = (64*64 - (int)p_bits) / 8;
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
                int wi = 25 + (pos >> 3);
                int bi = (pos & 7) << 3;
                A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    /* Byte-swap B portion (A[25..88]) from LE to BE for MD6 compression */
    for (int i = 25; i < 89; i++) A[i] = bswap64(A[i]);

    /* Run MD6 compression */
    md6_compress_loop(A);

    /* Extract output: last c=16 words of A, then take last d/w=4 words.
     * C[0..15] = A[r*c+n-c .. r*c+n-1] = A[1737..1752]
     * Hash = C[12..15] (last 4 uint64 words = last 256 bits = 8 uint32) */
    ulong C[4];
    C[0] = A[1749]; C[1] = A[1750]; C[2] = A[1751]; C[3] = A[1752];

    /* Byte-swap all 4 uint64 output words to LE, split into 8 uint32 */
    uint h[8];
    for (int i = 0; i < 4; i++) {
        ulong s = bswap64(C[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

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
