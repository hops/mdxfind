/* gpu_keccakunsalted.cl — Keccak-224/256/384/512 unsalted with mask expansion
 *
 * Keccak (SHA-3 competition version, without NIST 0x06 domain separator).
 * Sponge construction: XOR padded message into zero state, apply Keccak-f[1600].
 * Natively little-endian — no byte-swap needed.
 *
 * Padding: 0x01 after message, 0x80 at last rate byte.
 * Rate: 144 (224), 136 (256), 104 (384), 72 (512) bytes.
 *
 * Input: pre-padded rate-sized blocks with Keccak padding applied by host.
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */


/* Keccak-f[1600] round constants */
__constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

/* Keccak-f[1600] rotation offsets (linearized [x][y] where index = x*5+y) */
__constant uint ROTC[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

/* Keccak-f[1600] permutation — 24 rounds on 5×5 state of uint64 */
void keccak_f1600(ulong *st) {
    for (int round = 0; round < 24; round++) {
        /* θ (theta) */
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4) % 5] ^ rotate(C[(x+1) % 5], (ulong)1);
            for (int y = 0; y < 25; y += 5)
                st[x+y] ^= D[x];
        }

        /* ρ (rho) + π (pi) combined */
        ulong B[25];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[x + 5 * ((2*y + 3*x) % 5)] = rotate(st[x*5+y], (ulong)ROTC[x*5+y]);

        /* χ (chi) */
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 25; y += 5)
                st[x+y] = B[x+y] ^ (~B[((x+1)%5)+y] & B[((x+2)%5)+y]);

        /* ι (iota) */
        st[0] ^= RC[round];
    }
}

/* Fill masks into LE uint64 block (rate_words uint64 words) */
void keccak_fill_masks(ulong *block, uint n_pre, uint n_app, uint mask_idx,
                       __global const uchar *mask_desc, int rate,
                       uint num_masks, ulong mask_base0, ulong mask_base1) {
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
            block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
        }

        if (n_app > 0) {
            uchar *bp = (uchar *)block;
            int total_len = bp[rate] | (bp[rate+1] << 8);
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
                block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    } else {
        /* Append-only (brute-force): host pre-decomposes mask_start into
         * per-position base offsets. Kernel does fast uint32 local
         * decomposition and adds to base with carry. */
        uchar *bp = (uchar *)block;
        int total_len = bp[rate] | (bp[rate+1] << 8);
        int app_start = total_len - (int)n_app;
        uint local_idx = mask_idx % num_masks;
        uint aidx = local_idx;
        uint carry = 0;
        for (int i = (int)n_app - 1; i >= 0; i--) {
            uint sz = mask_desc[i];
            uint local_digit = aidx % sz;
            aidx /= sz;
            uint base_digit = (i < 8)
                ? (uint)((mask_base0 >> (i * 8)) & 0xFF)
                : (uint)((mask_base1 >> ((i - 8) * 8)) & 0xFF);
            uint sum = base_digit + local_digit + carry;
            carry = sum / sz;
            uint final_digit = sum % sz;
            uchar ch = mask_desc[n_total_m + i * 256 + final_digit];
            int pos = app_start + i;
            int wi = pos >> 3;
            int bi = (pos & 7) << 3;
            block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
        }
    }
}

/* Generic Keccak unsalted kernel.
 * rate_words = rate / 8.
 * hash_words = output hash size in uint32 (e.g. 7 for 224, 8 for 256, etc.)
 * hash_ulongs = number of uint64 state words containing output (ceil(hash_words/2))
 * The host packs: message in LE, 0x01 at total_len, 0x80 at rate-1.
 * total_len stored as uint16 at byte offset rate for mask append. */
#define KECCAK_KERNEL(name, rate, rate_words, hash_words, hash_ulongs)         \
__kernel void name(                                                            \
    __global const uchar *words,                                               \
    __global const ushort *unused_lens,                                         \
    __global const uchar *mask_desc,                                           \
    __global const uint *unused1, __global const ushort *unused2,              \
    __global const uint *compact_fp, __global const uint *compact_idx,         \
    __global const OCLParams *params_buf,                                      \
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, \
    __global const ushort *hash_data_len,                                      \
    __global uint *hits, __global volatile uint *hit_count,                    \
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths) \
{                                                                              \
    OCLParams params = *params_buf;                                            \
    uint tid = get_global_id(0);                                               \
    uint word_idx = tid / params.num_masks;                                    \
    ulong mask_idx = params.mask_start + (tid % params.num_masks);              \
    if (word_idx >= params.num_words) return;                                  \
                                                                               \
    /* Load padded block (rate bytes + 2 bytes metadata) */                    \
    int stride = rate + 8; /* stride includes metadata + alignment */          \
    __global const ulong *src = (__global const ulong *)(words + word_idx * stride); \
    ulong block[rate_words];                                                   \
    for (int i = 0; i < rate_words; i++) block[i] = src[i];                   \
                                                                               \
    /* Fill mask positions — LE byte manipulation, same approach */            \
    if (params.n_prepend > 0 || params.n_append > 0)                           \
        keccak_fill_masks(block, params.n_prepend, params.n_append,            \
                          mask_idx, mask_desc, rate,                            \
                          params.num_masks, params.mask_base0, params.mask_base1); \
                                                                               \
    /* XOR block into zero state and run Keccak-f[1600] */                     \
    ulong st[25];                                                              \
    for (int i = 0; i < 25; i++) st[i] = 0;                                   \
    for (int i = 0; i < rate_words; i++) st[i] = block[i];                    \
    keccak_f1600(st);                                                          \
                                                                               \
    /* Extract all hash words from state (LE, no swap needed) */               \
    uint h[hash_words];                                                        \
    for (int i = 0; i < hash_ulongs; i++) {                                    \
        h[i*2]   = (uint)st[i];                                               \
        if (i*2+1 < hash_words) h[i*2+1] = (uint)(st[i] >> 32);              \
    }                                                                          \
                                                                               \
    /* Probe compact table with first 128 bits */                              \
    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,        \
                      params.compact_mask, params.max_probe,                   \
                      params.hash_data_count, hash_data_buf, hash_data_off,    \
                      overflow_keys, overflow_hashes, overflow_offsets,         \
                      params.overflow_count)) {                                \
        { uint _slot = atomic_add(hit_count, 1u);                              \
        if (_slot < params.max_hits) {                                          \
            uint _base = _slot * HIT_STRIDE;                                   \
            hits[_base] = word_idx; hits[_base+1] = mask_idx; hits[_base+2] = 1u; \
            for (int i = 0; i < hash_words; i++) hits[_base+3+i] = h[i];       \
            for (uint _z = 3 + hash_words; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
            mem_fence(CLK_GLOBAL_MEM_FENCE);                                   \
        } }                                                                    \
    }                                                                          \
}

/* Keccak-224: rate=144 bytes = 18 uint64 words, output=7 uint32 (4 uint64) (pad byte 0x01) */
KECCAK_KERNEL(keccak224_unsalted_batch, 144, 18, 7, 4)

/* Keccak-256: rate=136 bytes = 17 uint64 words, output=8 uint32 (4 uint64) (pad byte 0x01) */
KECCAK_KERNEL(keccak256_unsalted_batch, 136, 17, 8, 4)

/* Keccak-384: rate=104 bytes = 13 uint64 words, output=12 uint32 (6 uint64) (pad byte 0x01) */
KECCAK_KERNEL(keccak384_unsalted_batch, 104, 13, 12, 6)

/* Keccak-512: rate=72 bytes = 9 uint64 words, output=16 uint32 (8 uint64) (pad byte 0x01) */
KECCAK_KERNEL(keccak512_unsalted_batch, 72, 9, 16, 8)

/* SHA3-224/256/384/512: same permutation, pad byte 0x06 (applied by host packer).
 * Same rates as Keccak. Kernel is identical — only padding differs. */
KECCAK_KERNEL(sha3_224_unsalted_batch, 144, 18, 7, 4)
KECCAK_KERNEL(sha3_256_unsalted_batch, 136, 17, 8, 4)
KECCAK_KERNEL(sha3_384_unsalted_batch, 104, 13, 12, 6)
KECCAK_KERNEL(sha3_512_unsalted_batch, 72, 9, 16, 8)
