/* gpu_sha256unsalted.cl — Pre-padded unsalted SHA256 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks in passbuf (word_stride=64).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, M[14]=bitlen (LE).
 *   Kernel byte-swaps to big-endian before SHA256 compress.
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0, h1, h2, h3) or
 *             7 (word_idx, mask_idx, iter, h0, h1, h2, h3) when max_iter > 1
 */


/* Convert one byte to 2 packed hex chars (lowercase, BE word packing) */
uint hex_byte_be(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Convert 8 SHA256 BE state words to 16 BE M[] words of hex text (lowercase).
 * 8 state words × 8 hex chars each = 64 hex chars = 16 M[] words.
 * Fills the entire M[0..15] block — padding must go in a second block. */
void sha256_to_hex_lc(uint *state, uint *M) {
    for (int i = 0; i < 8; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) |
           ((x << 8) & 0xff0000u) | ((x << 24) & 0xff000000u);
}

__constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

void sha256_compress(uint *state, uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++) {
        uint s0 = rotate(W[i-15], (uint)25) ^ rotate(W[i-15], (uint)14) ^ (W[i-15] >> 3);
        uint s1 = rotate(W[i-2], (uint)15) ^ rotate(W[i-2], (uint)13) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint S1 = rotate(e, (uint)26) ^ rotate(e, (uint)21) ^ rotate(e, (uint)7);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotate(a, (uint)30) ^ rotate(a, (uint)19) ^ rotate(a, (uint)10);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions — LE words, same as MD5 kernel */
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
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }

        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
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
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
        uint hx = bswap32(state[0]), hy = bswap32(state[1]);
        uint hz = bswap32(state[2]), hw = bswap32(state[3]);

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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
    uint M[16];
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
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }
        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
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

    uint hx = bswap32(state[0]), hy = bswap32(state[1]);
    uint hz = bswap32(state[2]), hw = bswap32(state[3]);

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

/* ---- SHA256RAW: sha256(sha256_binary(pass)) with binary iteration ----
 *
 * For hashcat modes 21400, 30420. With -i N, iterates N times on
 * binary 32-byte output: sha256(sha256(...sha256(pass)...))
 * Second+ iterations always hash exactly 32 bytes (single block, fixed padding).
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
    uint M[16];
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
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }
        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
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

    /* Binary iterations: sha256(32-byte binary output).
     * 32 bytes + 0x80 + zeros + bit-length 256 at M[15] — single block. */
    uint max_i = params.max_iter;
    if (max_i < 1) max_i = 1;
    for (uint iter = 1; iter <= max_i; iter++) {
        /* Pack state (BE) into M[] — already big-endian from compress */
        for (int i = 0; i < 8; i++) M[i] = state[i];
        M[8] = 0x80000000u;  /* 0x80 in high byte (BE) */
        for (int i = 9; i < 15; i++) M[i] = 0;
        M[15] = 256;  /* 32 bytes * 8 bits */

        state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
        state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
        state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
        state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
        sha256_compress(state, M);

        uint hx = bswap32(state[0]), hy = bswap32(state[1]);
        uint hz = bswap32(state[2]), hw = bswap32(state[3]);

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
}
