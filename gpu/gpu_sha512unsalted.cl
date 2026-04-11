/* gpu_sha512unsalted.cl — Pre-padded unsalted SHA512 with mask expansion
 *
 * Input: 2048 pre-padded 128-byte blocks in passbuf (word_stride=128).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, bitlen as uint32 at byte offset 120.
 *   Kernel byte-swaps 16 × uint64 words to big-endian before SHA512 compress.
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0_lo, h0_hi, h1_lo, h1_hi)
 */


/* Convert one byte to 2 packed hex chars (lowercase, BE ulong packing) */
ulong hex_byte_be64(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((ulong)(hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (ulong)(lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Convert 8 SHA512 BE ulong state words to 16 BE ulong M[] words of hex text.
 * Each state word (8 bytes BE) → 16 hex chars → 2 BE ulong M[] words.
 * Fills the entire M[0..15] block — padding must go in a second block. */
void sha512_to_hex_lc(ulong *state, ulong *M) {
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

ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL) | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL) | ((x >> 8) & 0xff000000UL) |
           ((x << 8) & 0xff00000000UL) | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL) | ((x << 56) & 0xff00000000000000UL);
}

__constant ulong K512[80] = {
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

ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

void sha512_compress(ulong *state, ulong *M) {
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    /* Load pre-padded block (16 uint64 = 128 bytes, little-endian from host) */
    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions — LE bytes within uint64 words */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= mask_desc[n_pre + i];

        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;

        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }

        if (n_app > 0) {
            /* Bit-length stored at uint32 offset 30 (byte 120) by host packer */
            __global const uint *src32 = (__global const uint *)(words + word_idx * 128);
            int total_len = src32[30];  /* already modified by mask, re-read from original? */
            /* Actually M[] has been loaded and may have mask prepend changes,
             * but the bitlen at byte 120 is untouched. Extract from M[15] low 32 bits. */
            total_len = (int)(M[15] & 0xFFFFFFFFUL);  /* host stored bitlen as LE uint32 at byte 120 */
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uint n_total_m = n_pre + n_app;
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
        ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
        uint hx = (uint)s0, hy = (uint)(s0 >> 32);
        uint hz = (uint)s1, hw = (uint)(s1 >> 32);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * hit_stride;
                hits[base]   = word_idx;
                hits[base+1] = mask_idx;
                if (hit_stride == 7) {
                    hits[base+2] = iter;
                    hits[base+3] = hx; hits[base+4] = hy;
                    hits[base+5] = hz; hits[base+6] = hw;
                } else {
                    hits[base+2] = hx; hits[base+3] = hy;
                    hits[base+4] = hz; hits[base+5] = hw;
                }
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
            sha512_compress(state, M);
            /* Second compress: padding block */
            M[0] = 0x8000000000000000UL;
            for (int i = 1; i < 15; i++) M[i] = 0;
            M[15] = 128 * 8;  /* 128 hex bytes = 1024 bits */
            sha512_compress(state, M);
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint n_total_m = n_pre + n_app;
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
                uint n_total_m = n_pre + n_app;
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

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base]   = word_idx;
            hits[base+1] = mask_idx;
            hits[base+2] = hx;
            hits[base+3] = hy;
            hits[base+4] = hz;
            hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* SHA384RAW: sha384(sha384_binary(pass)) — binary iteration.
 * First SHA384(pass) from pre-padded block, then SHA384(48-byte binary).
 * Hit stride: 6 */
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend, n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++) append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = i >> 3; int bi = (i & 7) << 3;
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
    sha512_compress(state, M);

    /* Second SHA384 on 48-byte binary: state[0..5] BE, pad at byte 48, len=384 */
    for (int i = 0; i < 6; i++) M[i] = state[i];
    M[6] = 0x8000000000000000UL;
    for (int i = 7; i < 15; i++) M[i] = 0;
    M[15] = 384;

    state[0] = 0xcbbb9d5dc1059ed8UL; state[1] = 0x629a292a367cd507UL;
    state[2] = 0x9159015a3070dd17UL; state[3] = 0x152fecd8f70e5939UL;
    state[4] = 0x67332667ffc00b31UL; state[5] = 0x8eb44a8768581511UL;
    state[6] = 0xdb0c2e0d64f98fa7UL; state[7] = 0x47b5481dbefa4fa4UL;
    sha512_compress(state, M);

    ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
    uint hx = (uint)s0, hy = (uint)(s0 >> 32);
    uint hz = (uint)s1, hw = (uint)(s1 >> 32);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = mask_idx;
            hits[base+2] = hx; hits[base+3] = hy; hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* SHA512RAW: sha512(sha512_binary(pass)) — binary iteration.
 * First SHA512(pass), then SHA512(64-byte binary) = one 128-byte block. */
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const ulong *src = (__global const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend, n_app = params.n_append;
    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++) append_combos *= mask_desc[n_pre + i];
        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = i >> 3; int bi = (i & 7) << 3;
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
    sha512_compress(state, M);

    /* Second SHA512 on 64-byte binary: state[0..7] BE + pad */
    for (int i = 0; i < 8; i++) M[i] = state[i];
    M[8] = 0x8000000000000000UL;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = 512;

    state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;
    state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;
    state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;
    state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;
    sha512_compress(state, M);

    ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
    uint hx = (uint)s0, hy = (uint)(s0 >> 32);
    uint hz = (uint)s1, hw = (uint)(s1 >> 32);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = mask_idx;
            hits[base+2] = hx; hits[base+3] = hy; hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
