/* gpu_md4unsalted.cl — Pre-padded unsalted MD4 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks in passbuf (word_stride=64).
 *   Each block has: password at offset n_prepend, 0x80 padding, M[14]=bitlen.
 *   Prepend/append gaps are zeroed — GPU fills them per mask combination.
 *
 * Dispatch: num_words × num_masks threads.
 *   word_idx = tid / num_masks, mask_idx = mask_start + (tid % num_masks)
 *
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */

/* Charset tables for mask expansion */

/* Convert one byte to 2 packed hex chars (lowercase, LE) */
uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

/* Convert 4 MD4 LE uint32s to 8 LE M[] words of hex text (lowercase) */
void md4_to_hex_lc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

/* Fully unrolled MD4 compress — single block, M[] already in little-endian uint32
 * RFC 1320: 3 rounds × 16 steps = 48 operations (vs MD5's 64)
 * Round 1: F(b,c,d) = (b&c)|(~b&d),       no constant
 * Round 2: G(b,c,d) = (b&c)|(b&d)|(c&d),  constant 0x5A827999
 * Round 3: H(b,c,d) = b^c^d,               constant 0x6ED9EBA1
 */
void md4_compress(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
    /* Round 1: F(b,c,d) = (b&c)|(~b&d), rotations: 3,7,11,19 */
    a += ((b&c)|(~b&d)) + M[ 0]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 1]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[ 2]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[ 3]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[ 4]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 5]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[ 6]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[ 7]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[ 8]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[ 9]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[10]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[11]; b = ((b<<19)|(b>>13));
    a += ((b&c)|(~b&d)) + M[12]; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(~a&c)) + M[13]; d = ((d<< 7)|(d>>25));
    c += ((d&a)|(~d&b)) + M[14]; c = ((c<<11)|(c>>21));
    b += ((c&d)|(~c&a)) + M[15]; b = ((b<<19)|(b>>13));
    /* Round 2: G(b,c,d) = (b&c)|(b&d)|(c&d), constant 0x5A827999
     * Message order: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15 */
    a += ((b&c)|(b&d)|(c&d)) + M[ 0] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 4] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[ 8] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[12] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 1] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 5] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[ 9] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[13] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 2] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 6] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[10] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[14] + 0x5A827999u; b = ((b<<13)|(b>>19));
    a += ((b&c)|(b&d)|(c&d)) + M[ 3] + 0x5A827999u; a = ((a<< 3)|(a>>29));
    d += ((a&b)|(a&c)|(b&c)) + M[ 7] + 0x5A827999u; d = ((d<< 5)|(d>>27));
    c += ((d&a)|(d&b)|(a&b)) + M[11] + 0x5A827999u; c = ((c<< 9)|(c>>23));
    b += ((c&d)|(c&a)|(d&a)) + M[15] + 0x5A827999u; b = ((b<<13)|(b>>19));
    /* Round 3: H(b,c,d) = b^c^d, constant 0x6ED9EBA1
     * Message order: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15 */
    a += (b^c^d) + M[ 0] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[ 8] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 4] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[12] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 2] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[10] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 6] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[14] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 1] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[ 9] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 5] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[13] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    a += (b^c^d) + M[ 3] + 0x6ED9EBA1u; a = ((a<< 3)|(a>>29));
    d += (a^b^c) + M[11] + 0x6ED9EBA1u; d = ((d<< 9)|(d>>23));
    c += (d^a^b) + M[ 7] + 0x6ED9EBA1u; c = ((c<<11)|(c>>21));
    b += (c^d^a) + M[15] + 0x6ED9EBA1u; b = ((b<<15)|(b>>17));
    *hx += a; *hy += b; *hz += c; *hw += d;
}

__kernel void md4_unsalted_batch(
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

    /* Load pre-padded M[] block (16 uint32 = 64 bytes) */
    __global const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions into M[] */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        /* Compute append_combos for index decomposition */
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= mask_desc[n_pre + i];

        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;

        /* Fill prepend chars at byte offset 0 */
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                /* Set byte i in M[] (little-endian uint32) */
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }

        /* Fill append chars — total length from M[14]/8, append starts at total - n_app */
        if (n_app > 0) {
            int total_len = M[14] >> 3;  /* bit-length / 8 */
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

    /* MD4 compress (single block, fully unrolled) */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md4_compress(&hx, &hy, &hz, &hw, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
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
            md4_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md4_compress(&hx, &hy, &hz, &hw, M);
        }
    }
}

/* MD4UTF16 (NTLM-style): MD4 of UTF-16LE zero-extended password.
 * Host packs UTF-16LE (each byte → byte,0x00) with doubled offsets.
 * Mask positions are doubled: logical char i → byte 2*i in the block.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw) */
__kernel void md4utf16_unsalted_batch(
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

        /* Fill prepend: logical char i → UTF-16LE byte 2*i */
        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int pos = 2 * i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }

        /* Fill append: total UTF-16LE byte length from M[14]/8,
         * append starts at total_bytes - 2*n_app */
        if (n_app > 0) {
            int total_bytes = M[14] >> 3;
            int app_start = total_bytes - 2 * (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uint n_total_m = n_pre + n_app;
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + 2 * i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md4_compress(&hx, &hy, &hz, &hw, M);

    uint max_iter = params.max_iter;
    uint hit_stride = (max_iter > 1) ? 7 : 6;

    for (uint iter = 1; iter <= max_iter; iter++) {
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
            /* Iteration uses plain MD4 on hex text (not UTF16) */
            md4_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md4_compress(&hx, &hy, &hz, &hw, M);
        }
    }
}
