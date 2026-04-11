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

/* Charset tables for mask expansion */

/* Convert one byte to 2 packed hex chars (lowercase, LE) */
uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

/* Convert 4 MD5 LE uint32s to 8 LE M[] words of hex text (lowercase) */
void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

/* Fully unrolled MD5 compress — single block, M[] already in little-endian uint32 */
void md5_compress(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
    /* Round 1: F */
    a += ((b&c)|(~b&d)) + M[ 0] + 0xd76aa478u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 1] + 0xe8c7b756u; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[ 2] + 0x242070dbu; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[ 3] + 0xc1bdceeeu; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[ 4] + 0xf57c0fafu; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 5] + 0x4787c62au; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[ 6] + 0xa8304613u; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[ 7] + 0xfd469501u; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[ 8] + 0x698098d8u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 9] + 0x8b44f7afu; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[10] + 0xffff5bb1u; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[11] + 0x895cd7beu; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[12] + 0x6b901122u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[13] + 0xfd987193u; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[14] + 0xa679438eu; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[15] + 0x49b40821u; b = c + ((b<<22)|(b>>10));
    /* Round 2: G */
    a += ((d&b)|(~d&c)) + M[ 1] + 0xf61e2562u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[ 6] + 0xc040b340u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[11] + 0x265e5a51u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 0] + 0xe9b6c7aau; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[ 5] + 0xd62f105du; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[10] + 0x02441453u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[15] + 0xd8a1e681u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 4] + 0xe7d3fbc8u; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[ 9] + 0x21e1cde6u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[14] + 0xc33707d6u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[ 3] + 0xf4d50d87u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 8] + 0x455a14edu; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[13] + 0xa9e3e905u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[ 2] + 0xfcefa3f8u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[ 7] + 0x676f02d9u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[12] + 0x8d2a4c8au; b = c + ((b<<20)|(b>>12));
    /* Round 3: H */
    a += (b^c^d) + M[ 5] + 0xfffa3942u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 8] + 0x8771f681u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[11] + 0x6d9d6122u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[14] + 0xfde5380cu; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[ 1] + 0xa4beea44u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 4] + 0x4bdecfa9u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[ 7] + 0xf6bb4b60u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[10] + 0xbebfbc70u; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[13] + 0x289b7ec6u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 0] + 0xeaa127fau; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[ 3] + 0xd4ef3085u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[ 6] + 0x04881d05u; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[ 9] + 0xd9d4d039u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[12] + 0xe6db99e5u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[15] + 0x1fa27cf8u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[ 2] + 0xc4ac5665u; b = c + ((b<<23)|(b>> 9));
    /* Round 4: I */
    a += (c^(~d|b)) + M[ 0] + 0xf4292244u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[ 7] + 0x432aff97u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[14] + 0xab9423a7u; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 5] + 0xfc93a039u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[12] + 0x655b59c3u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[ 3] + 0x8f0ccc92u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[10] + 0xffeff47du; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 1] + 0x85845dd1u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[ 8] + 0x6fa87e4fu; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[15] + 0xfe2ce6e0u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[ 6] + 0xa3014314u; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[13] + 0x4e0811a1u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[ 4] + 0xf7537e82u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[11] + 0xbd3af235u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[ 2] + 0x2ad7d2bbu; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 9] + 0xeb86d391u; b = c + ((b<<21)|(b>>11));
    *hx += a; *hy += b; *hz += c; *hw += d;
}

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

    /* MD5 compress (single block, fully unrolled) */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_compress(&hx, &hy, &hz, &hw, M);

    /* Probe compact table at iteration 1 and emit hit */
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
            /* Hex-encode hash into M[0..7], set up constant padding */
            md5_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80;
            for (int i = 9; i < 14; i++) M[i] = 0;
            M[14] = 32 * 8; M[15] = 0;  /* 32 hex bytes = 256 bits */
            hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
            md5_compress(&hx, &hy, &hz, &hw, M);
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    __global const uint *src = words + word_idx * 16;
    uint M[16];
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

    /* First MD5 */
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_compress(&hx, &hy, &hz, &hw, M);

    /* Second MD5 on 16-byte binary result: M[0..3]=hash, M[4]=0x80, M[14]=128 */
    M[0] = hx; M[1] = hy; M[2] = hz; M[3] = hw;
    M[4] = 0x80;
    for (int i = 5; i < 14; i++) M[i] = 0;
    M[14] = 128; M[15] = 0;
    hx = 0x67452301u; hy = 0xEFCDAB89u; hz = 0x98BADCFEu; hw = 0x10325476u;
    md5_compress(&hx, &hy, &hz, &hw, M);

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
