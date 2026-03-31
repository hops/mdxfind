/* md5salt.cl — OpenCL kernels for mdxfind MD5SALT GPU acceleration */

typedef struct {
    ulong compact_mask;
    uint num_words;
    uint num_salts;
    uint salt_start;
    uint max_probe;
    uint hash_data_count;
    uint max_hits;
    uint overflow_count;
    uint max_iter;
} OCLParams;

__constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotate(a,s); }
#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotate(a,s); }
#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + rotate(a,s); }
#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + rotate(a,s); }

void md5_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {
    uint a = *h0, b = *h1, c = *h2, d = *h3;
    FF(a,b,c,d,M[0],(uint)7,0xd76aa478u);  FF(d,a,b,c,M[1],(uint)12,0xe8c7b756u);
    FF(c,d,a,b,M[2],(uint)17,0x242070dbu);  FF(b,c,d,a,M[3],(uint)22,0xc1bdceeeu);
    FF(a,b,c,d,M[4],(uint)7,0xf57c0fafu);   FF(d,a,b,c,M[5],(uint)12,0x4787c62au);
    FF(c,d,a,b,M[6],(uint)17,0xa8304613u);  FF(b,c,d,a,M[7],(uint)22,0xfd469501u);
    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);
    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);
    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);
    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);
    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);
    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);
    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);
    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);
    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);
    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);
    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);
    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);
    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);
    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);
    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);
    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);
    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);
    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);
    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);
    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);
    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);
    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);
    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);
    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);
    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);
    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);
    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);
    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

/* ---- SHA256 block function ---- */
__constant uint SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define S256_ROTR(x,n) rotate((x),(uint)(32-(n)))
#define S256_CH(x,y,z)  ((x & y) ^ (~x & z))
#define S256_MAJ(x,y,z) ((x & y) ^ (x & z) ^ (y & z))
#define S256_EP0(x)  (S256_ROTR(x,2)  ^ S256_ROTR(x,13) ^ S256_ROTR(x,22))
#define S256_EP1(x)  (S256_ROTR(x,6)  ^ S256_ROTR(x,11) ^ S256_ROTR(x,25))
#define S256_SIG0(x) (S256_ROTR(x,7)  ^ S256_ROTR(x,18) ^ (x >> 3))
#define S256_SIG1(x) (S256_ROTR(x,17) ^ S256_ROTR(x,19) ^ (x >> 10))

/* SHA256 processes big-endian uint32 words. M[] must be pre-swapped by caller. */
void sha256_block(uint *state, uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++)
        W[i] = S256_SIG1(W[i-2]) + W[i-7] + S256_SIG0(W[i-15]) + W[i-16];

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = h + S256_EP1(e) + S256_CH(e,f,g) + SHA256_K[i] + W[i];
        uint t2 = S256_EP0(a) + S256_MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* Byte-swap a uint32 (for SHA256 big-endian <-> little-endian conversion) */
uint bswap32(uint x) {
    return ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) |
           ((x << 8) & 0xff0000) | ((x << 24) & 0xff000000u);
}

/* Copy bytes into a big-endian M[] array for SHA256.
 * SHA256 M[] words are big-endian, so byte 0 goes into bits 31:24 of M[0]. */
void S_copy_bytes(uint *M, int byte_off, __global const uchar *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 4;
        int bi = 3 - ((byte_off + i) % 4);  /* big-endian: byte 0 = shift 24 */
        M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)src[i] << (bi * 8));
    }
}

void S_set_byte(uint *M, int byte_off, uchar val) {
    int wi = byte_off / 4;
    int bi = 3 - (byte_off % 4);
    M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)val << (bi * 8));
}

ulong compact_mix(ulong k) { return k ^ (k >> 32); }

int probe_compact(uint hx, uint hy, uint hz, uint hw,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count)
{
    ulong key = ((ulong)hy << 32) | hx;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = compact_mix(key) & compact_mask;
    for (int p = 0; p < (int)max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                __global const uint *ref = (__global const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3])
                    return 1;
            }
        }
        pos = (pos + 1) & compact_mask;
    }
    if (overflow_count > 0) {
        int lo = 0, hi = (int)overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);
                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                break;
            }
        }
    }
    return 0;
}

__kernel void md5salt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];
    for (int i = 8; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++)
        mbytes[32 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

__kernel void md5salt_sub8_24(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 4; i++) M[i] = mwords[i];
    for (int i = 4; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 16 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++) mbytes[16 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

__kernel void md5salt_iter(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];
    for (int i = 8; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++) mbytes[32 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    for (uint iter = 0; iter < params.max_iter; iter++) {
        if (iter > 0) {
            uint hwords[4]; hwords[0]=hx; hwords[1]=hy; hwords[2]=hz; hwords[3]=hw;
            uchar *hb = (uchar *)hwords;
            uint Mi[16];
            uchar *mb = (uchar *)Mi;
            for (int i = 0; i < 16; i++) {
                uchar hi = hb[i] >> 4, lo = hb[i] & 0xf;
                mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);
                mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);
            }
            Mi[8] = 0x80; Mi[9]=0; Mi[10]=0; Mi[11]=0;
            Mi[12]=0; Mi[13]=0; Mi[14]=256; Mi[15]=0;
            hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;
            md5_block(&hx, &hy, &hz, &hw, Mi);
        }
        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = word_idx; hits[base+1] = salt_idx;
                hits[base+2] = iter + 1;
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }
    }
}

/* ---- Hex conversion for MD5 iteration ---- */

/* Convert one byte to 2 packed hex chars (lowercase) */
uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

/* Convert one byte to 2 packed hex chars (uppercase) */
uint hex_byte_uc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('A' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('A' - 10));
    return hc | (lc << 8);
}

/* Convert 4 MD5 uint32s to 8 M[] words of hex text (lowercase) */
void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

/* Convert 4 MD5 uint32s to 8 M[] words of hex text (uppercase) */
void md5_to_hex_uc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_uc(b0) | (hex_byte_uc(b1) << 16);
        M[i*2+1] = hex_byte_uc(b2) | (hex_byte_uc(b3) << 16);
    }
}

/* ---- Bare MD5 iteration kernel (e1 lowercase, e2 uppercase) ----
 *
 * Input: hex(MD5(password)) pre-packed by CPU into M[] words.
 * GPU iterates: MD5(hex) -> check -> hex(result) -> MD5(hex) -> ...
 * One work item per word (no salt dimension).
 * Hit stride is always 7 (includes iteration number).
 */
__kernel void md5_iter_lc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    /* MD5 padding for 32-byte input -- constant across all iterations */
    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = tid; hits[base+1] = 0;
                hits[base+2] = iter + 1; /* +1: GPU iter 1 = mdxfind iter 2 */
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }

        if (iter < params.max_iter) {
            md5_to_hex_lc(hx, hy, hz, hw, M);
            /* M[8..15] unchanged: padding for 32-byte input */
        }
    }
}

__kernel void md5_iter_uc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = tid; hits[base+1] = 0;
                hits[base+2] = iter + 1; /* +1: GPU iter 1 = mdxfind iter 2 */
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }

        if (iter < params.max_iter) {
            md5_to_hex_uc(hx, hy, hz, hw, M);
        }
    }
}

/* ---- Build M[] from byte stream ----
 * Copy 'len' bytes from global source into M[], starting at byte offset 'off'.
 * M[] must be pre-zeroed. Works by accumulating bytes into uint32 words. */
void M_copy_bytes(uint *M, int off, __global const uchar *src, int len) {
    for (int i = 0; i < len; i++) {
        int pos = off + i;
        int word = pos >> 2;
        int shift = (pos & 3) << 3;
        M[word] |= ((uint)src[i]) << shift;
    }
}

/* Set a single byte in M[] */
void M_set_byte(uint *M, int pos, uint val) {
    M[pos >> 2] |= val << ((pos & 3) << 3);
}

/* ---- MD5(salt + password) kernel ----
 *
 * Input: raw password bytes in hexhashes buffer, length in hexlens.
 * GPU constructs salt + password, computes MD5, checks compact table.
 * Handles 1-block (total <= 55) and 2-block (55 < total <= 119) dynamically.
 * Hit stride is 7 (includes iteration number).
 */
__kernel void md5saltpass_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
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
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    /* Read password bytes and length */
    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;

    /* Read salt offset and length */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    /* Build message = salt + password, compute MD5 */
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 55) {
        M_copy_bytes(M, 0, salts + soff, slen);
        M_copy_bytes(M, slen, pass, plen);
        M_set_byte(M, total_len, 0x80);
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      } else {
        /* Two blocks: fill first 64 bytes from salt+pass */
        int salt_b1 = (slen < 64) ? slen : 64;
        M_copy_bytes(M, 0, salts + soff, salt_b1);
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0)
            M_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 64)
            M_set_byte(M, total_len, 0x80);
        md5_block(&hx, &hy, &hz, &hw, M);
        /* Second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) {
            M_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2);
            pos2 = salt_b2;
        }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) {
            M_copy_bytes(M, pos2, pass + pass_b1, pass_b2);
            pos2 += pass_b2;
        }
        if (total_len >= 64)
            M_set_byte(M, pos2, 0x80);
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      }
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 7;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = 1;
            hits[base+3] = hx; hits[base+4] = hy;
            hits[base+5] = hz; hits[base+6] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- MD5(password + salt) kernel ---- */
__kernel void md5passsalt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
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
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 55) {
        M_copy_bytes(M, 0, pass, plen);
        M_copy_bytes(M, plen, salts + soff, slen);
        M_set_byte(M, total_len, 0x80);
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      } else {
        /* Two blocks: fill first 64 bytes from pass+salt */
        int pass_b1 = (plen < 64) ? plen : 64;
        M_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0)
            M_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        /* If all data fits in block 1, put 0x80 here */
        if (total_len < 64)
            M_set_byte(M, total_len, 0x80);
        md5_block(&hx, &hy, &hz, &hw, M);
        /* Second block: remaining data + padding */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) {
            M_copy_bytes(M, 0, pass + pass_b1, pass_b2);
            pos2 = pass_b2;
        }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) {
            M_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2);
            pos2 += salt_b2;
        }
        if (total_len >= 64)
            M_set_byte(M, pos2, 0x80);
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      }
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 7;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = 1;
            hits[base+3] = hx; hits[base+4] = hy;
            hits[base+5] = hz; hits[base+6] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- MD5(MD5(salt).MD5(pass)) kernel (e367) ---- */
/* Salt buffer has hex(MD5(salt)) [32 bytes], hexhash buffer has hex(MD5(pass)) [32 bytes].
 * Total input is always exactly 64 bytes → deterministic 2-block MD5. */
__kernel void md5_md5saltmd5pass_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    /* Load hex(MD5(salt)) [32 bytes] into M[0..7] as LE uint32 words */
    uint M[16];
    { __global const uchar *sp = salts + salt_offsets[salt_idx];
      for (int i = 0; i < 8; i++)
        M[i] = (uint)sp[i*4] | ((uint)sp[i*4+1]<<8) | ((uint)sp[i*4+2]<<16) | ((uint)sp[i*4+3]<<24);
    }

    /* Load hex(MD5(pass)) [32 bytes] into M[8..15] */
    __global const uint *pass_words = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[8 + i] = pass_words[i];

    /* Block 1: MD5 of the 64 data bytes (no padding in this block) */
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Block 2: padding only — 0x80 at byte 0, length 512 bits at M[14] */
    for (int i = 0; i < 16; i++) M[i] = 0;
    M[0] = 0x00000080u;
    M[14] = 64 * 8;  /* 512 bits */
    md5_block(&hx, &hy, &hz, &hw, M);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- MD5CRYPT kernel (e511): $1$salt$hash, 1000 MD5 iterations ---- */
/* Salt buffer contains "$1$salt$" from Typesalt. Kernel extracts raw salt. */

/* Helper: build MD5 message block from a byte buffer, with padding */
void md5_oneshot(const uchar *data, int len, uint *out) {
    uint M[16];
    out[0] = 0x67452301u; out[1] = 0xEFCDAB89u;
    out[2] = 0x98BADCFEu; out[3] = 0x10325476u;

    int pos = 0;
    /* Process full 64-byte blocks */
    while (pos + 64 <= len) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)data[pos+i*4] | ((uint)data[pos+i*4+1]<<8) |
                   ((uint)data[pos+i*4+2]<<16) | ((uint)data[pos+i*4+3]<<24);
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
        pos += 64;
    }
    /* Final block(s) with padding */
    int rem = len - pos;
    uchar pad[128];
    for (int i = 0; i < 128; i++) pad[i] = 0;
    for (int i = 0; i < rem; i++) pad[i] = data[pos + i];
    pad[rem] = 0x80;
    int blocks = (rem < 56) ? 1 : 2;
    int lenoff = (blocks == 1) ? 56 : 120;
    pad[lenoff] = (len * 8) & 0xff;
    pad[lenoff+1] = ((len * 8) >> 8) & 0xff;
    pad[lenoff+2] = ((len * 8) >> 16) & 0xff;
    pad[lenoff+3] = ((len * 8) >> 24) & 0xff;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 16; i++)
            M[i] = (uint)pad[b*64+i*4] | ((uint)pad[b*64+i*4+1]<<8) |
                   ((uint)pad[b*64+i*4+2]<<16) | ((uint)pad[b*64+i*4+3]<<24);
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
    }
}

__kernel void md5crypt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
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
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];

    /* Salt buffer has "$1$salt$" — extract raw salt (skip "$1$", stop before trailing "$") */
    __global const uchar *salt_raw = salts + soff + 3; /* skip "$1$" */
    int saltlen = slen_full - 4; /* remove "$1$" prefix and trailing "$" */
    if (saltlen < 0) saltlen = 0;
    if (saltlen > 8) saltlen = 8;

    /* Local buffers */
    uchar buf[256]; /* working buffer for MD5 input */
    uint digest[4]; /* current MD5 state */

    /* Step 1: digest_b = MD5(password + salt + password) */
    int blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i];
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    md5_oneshot(buf, blen, digest);
    uchar digest_b[16];
    for (int i = 0; i < 4; i++) {
        digest_b[i*4]   = digest[i] & 0xff;
        digest_b[i*4+1] = (digest[i] >> 8) & 0xff;
        digest_b[i*4+2] = (digest[i] >> 16) & 0xff;
        digest_b[i*4+3] = (digest[i] >> 24) & 0xff;
    }

    /* Step 2: digest_a = MD5(password + "$1$" + salt + digest_b_chunks + bit_bytes) */
    blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = pass[i];
    buf[blen++] = '$'; buf[blen++] = '1'; buf[blen++] = '$';
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i];
    /* Append digest_b for password length bytes */
    for (int x = plen; x > 0; x -= 16) {
        int n = (x > 16) ? 16 : x;
        for (int i = 0; i < n; i++) buf[blen++] = digest_b[i];
    }
    /* Bit-dependent bytes */
    for (int x = plen; x != 0; x >>= 1)
        buf[blen++] = (x & 1) ? 0 : pass[0];
    md5_oneshot(buf, blen, digest);

    /* Step 3: 1000 iterations */
    uchar dig[16];
    for (int i = 0; i < 4; i++) {
        dig[i*4]   = digest[i] & 0xff;
        dig[i*4+1] = (digest[i] >> 8) & 0xff;
        dig[i*4+2] = (digest[i] >> 16) & 0xff;
        dig[i*4+3] = (digest[i] >> 24) & 0xff;
    }
    for (int x = 0; x < 1000; x++) {
        blen = 0;
        if (x & 1) { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        else       { for (int i = 0; i < 16; i++) buf[blen++] = dig[i]; }
        if (x % 3) { for (int i = 0; i < saltlen; i++) buf[blen++] = salt_raw[i]; }
        if (x % 7) { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        if (x & 1) { for (int i = 0; i < 16; i++) buf[blen++] = dig[i]; }
        else       { for (int i = 0; i < plen; i++) buf[blen++] = pass[i]; }
        md5_oneshot(buf, blen, digest);
        for (int i = 0; i < 4; i++) {
            dig[i*4]   = digest[i] & 0xff;
            dig[i*4+1] = (digest[i] >> 8) & 0xff;
            dig[i*4+2] = (digest[i] >> 16) & 0xff;
            dig[i*4+3] = (digest[i] >> 24) & 0xff;
        }
    }

    /* Probe compact table with 16-byte binary result */
    uint hx = digest[0], hy = digest[1], hz = digest[2], hw = digest[3];
    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- SHA256(password + salt) kernel (e413) ---- */
__kernel void sha256passsalt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
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
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint state[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 55) {
        S_copy_bytes(M, 0, pass, plen);
        S_copy_bytes(M, plen, salts + soff, slen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;  /* big-endian length in bits */
        sha256_block(state, M);
      } else {
        int pass_b1 = (plen < 64) ? plen : 64;
        S_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0)
            S_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        if (total_len < 64)
            S_set_byte(M, total_len, 0x80);
        sha256_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, 0, pass + pass_b1, pass_b2); pos2 = pass_b2; }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2); pos2 += salt_b2; }
        if (total_len >= 64)
            S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
      }
    }

    /* Byte-swap all 8 state words to match host's big-endian storage */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            /* SHA256 hit: stride 11 = widx + sidx + iter + 8 hash words */
            uint base = slot * 11;
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 8; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- SHA256(salt + password) kernel (e412) ---- */
__kernel void sha256saltpass_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
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
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    uint state[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 55) {
        S_copy_bytes(M, 0, salts + soff, slen);
        S_copy_bytes(M, slen, pass, plen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
      } else {
        int salt_b1 = (slen < 64) ? slen : 64;
        S_copy_bytes(M, 0, salts + soff, salt_b1);
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0)
            S_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 64)
            S_set_byte(M, total_len, 0x80);
        sha256_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2); pos2 = salt_b2; }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, pos2, pass + pass_b1, pass_b2); pos2 += pass_b2; }
        if (total_len >= 64)
            S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
      }
    }

    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 11;
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 8; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* Self-test kernel: each work item computes MD5("test"), writes 1 if correct, 0 if not.
 * Used at init to probe the maximum reliable dispatch size for this GPU. */
__kernel void gpu_selftest(__global uint *results) {
    uint tid = get_global_id(0);
    uint M[16];
    M[0] = 0x74736574u;  /* "test" LE */
    M[1] = 0x00000080u;  /* padding byte */
    for (int i = 2; i < 14; i++) M[i] = 0;
    M[14] = 32u;         /* 4 bytes * 8 bits */
    M[15] = 0;
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_block(&hx, &hy, &hz, &hw, M);
    /* MD5("test") = 098f6bcd... -> LE word0 = 0xcd6b8f09 */
    results[tid] = (hx == 0xcd6b8f09u && hy == 0x73d32146u) ? 1u : 0u;
}
