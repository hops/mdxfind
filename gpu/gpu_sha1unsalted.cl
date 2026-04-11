/* gpu_sha1unsalted.cl — Pre-padded unsalted SHA1 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks in passbuf (word_stride=64).
 *   Packed by gpu_try_pack_unsalted() in little-endian format:
 *   password at offset n_prepend, 0x80 padding, M[14]=bitlen (LE).
 *   Kernel byte-swaps to big-endian before SHA1 compress.
 *
 * Dispatch: num_words × num_masks threads.
 *   word_idx = tid / num_masks, mask_idx = mask_start + (tid % num_masks)
 *
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

/* Convert 5 SHA1 BE state words to 10 BE M[] words of hex text (lowercase).
 * SHA1 state is big-endian: byte 0 = bits 24-31 (MSB first).
 * Each state word -> 8 hex chars -> 2 BE M[] words. */
void sha1_to_hex_lc(uint *state, uint *M) {
    for (int i = 0; i < 5; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

/* Charset tables for mask expansion */

uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) |
           ((x << 8) & 0xff0000u) | ((x << 24) & 0xff000000u);
}

/* SHA1 block function — M[] must be big-endian uint32 words */
void sha1_compress(uint *state, uint *M) {
    uint W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++)
        W[i] = rotate(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], (uint)1);

    uint a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    uint t;
    for (int i = 0; i < 20; i++) {
        t = rotate(a, (uint)5) + ((b & c) | (~b & d)) + e + 0x5A827999u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 20; i < 40; i++) {
        t = rotate(a, (uint)5) + (b ^ c ^ d) + e + 0x6ED9EBA1u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 40; i < 60; i++) {
        t = rotate(a, (uint)5) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 60; i < 80; i++) {
        t = rotate(a, (uint)5) + (b ^ c ^ d) + e + 0xCA62C1D6u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

__kernel void sha1_unsalted_batch(
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    /* Load pre-padded M[] block (16 uint32 = 64 bytes, little-endian from host) */
    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions into M[] — operates on LE words (same as MD5 kernel) */
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

    /* Save bit-length from LE M[14] before byte-swap */
    uint bitlen = M[14];

    /* Convert M[] from little-endian to big-endian for SHA1 */
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);

    /* Fix up padding: SHA1 stores 64-bit big-endian bit count at M[14..15] */
    M[14] = 0;
    M[15] = bitlen;

    /* SHA1 compress */
    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(state, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
        /* Byte-swap state to LE for compact table probe */
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
            /* Hex-encode BE state into BE M[0..9] (40 hex chars), set padding */
            sha1_to_hex_lc(state, M);
            M[10] = 0x80000000u;
            for (int i = 11; i < 15; i++) M[i] = 0;
            M[15] = 40 * 8;  /* 40 hex bytes = 320 bits */
            state[0] = 0x67452301u; state[1] = 0xEFCDAB89u;
            state[2] = 0x98BADCFEu; state[3] = 0x10325476u; state[4] = 0xC3D2E1F0u;
            sha1_compress(state, M);
        }
    }
}

/* ---- SQL5: sha1(sha1(pass)) — hashcat mode 300 ----
 * Double SHA1 with binary iteration. Same pre-padded block for first SHA1,
 * then SHA1 of 20-byte binary result (fixed single-block padding).
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */
__kernel void sql5_unsalted_batch(
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

    /* First SHA1(password) */
    uint bitlen = M[14];
    for (int i = 0; i < 16; i++) M[i] = bswap32(M[i]);
    M[14] = 0; M[15] = bitlen;

    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(state, M);

    /* Second SHA1(20-byte binary): fixed padding block */
    for (int i = 0; i < 5; i++) M[i] = state[i];  /* already BE */
    M[5] = 0x80000000u;
    for (int i = 6; i < 15; i++) M[i] = 0;
    M[15] = 160;  /* 20 bytes * 8 bits */

    state[0] = 0x67452301u; state[1] = 0xEFCDAB89u;
    state[2] = 0x98BADCFEu; state[3] = 0x10325476u; state[4] = 0xC3D2E1F0u;
    sha1_compress(state, M);

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
