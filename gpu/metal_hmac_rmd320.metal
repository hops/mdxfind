/* metal_hmac_rmd320.metal -- HMAC-RIPEMD-320 GPU kernels (Metal)
 * RIPEMD-320: 64-byte block, 40-byte output (10 x uint32), little-endian.
 * Same round functions as RMD160 but both halves kept + cross-swaps.
 * HMAC block size = 64.
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 */

#define F1(x, y, z) ((x) ^ (y) ^ (z))
#define F2(x, y, z) ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z) (((x) | ~(y)) ^ (z))
#define F4(x, y, z) ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z) ((x) ^ ((y) | ~(z)))

#define RMD_STEP(FUNC, A, B, C, D, E, X, S, K) \
    (A) += FUNC((B), (C), (D)) + (X) + (K); \
    (A) = rotate((A), (uint)(S)) + (E); \
    (C) = rotate((C), (uint)10);

static void rmd320_block(thread uint *hash, thread const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3], E = hash[4];
    uint AA = hash[5], BB = hash[6], CC = hash[7], DD = hash[8], EE = hash[9];

    /* j=0...15 */
    RMD_STEP(F1, A, B, C, D, E, X[0], 11, 0x00000000u);
    RMD_STEP(F1, E, A, B, C, D, X[1], 14, 0x00000000u);
    RMD_STEP(F1, D, E, A, B, C, X[2], 15, 0x00000000u);
    RMD_STEP(F1, C, D, E, A, B, X[3], 12, 0x00000000u);
    RMD_STEP(F1, B, C, D, E, A, X[4], 5, 0x00000000u);
    RMD_STEP(F1, A, B, C, D, E, X[5], 8, 0x00000000u);
    RMD_STEP(F1, E, A, B, C, D, X[6], 7, 0x00000000u);
    RMD_STEP(F1, D, E, A, B, C, X[7], 9, 0x00000000u);
    RMD_STEP(F1, C, D, E, A, B, X[8], 11, 0x00000000u);
    RMD_STEP(F1, B, C, D, E, A, X[9], 13, 0x00000000u);
    RMD_STEP(F1, A, B, C, D, E, X[10], 14, 0x00000000u);
    RMD_STEP(F1, E, A, B, C, D, X[11], 15, 0x00000000u);
    RMD_STEP(F1, D, E, A, B, C, X[12], 6, 0x00000000u);
    RMD_STEP(F1, C, D, E, A, B, X[13], 7, 0x00000000u);
    RMD_STEP(F1, B, C, D, E, A, X[14], 9, 0x00000000u);
    RMD_STEP(F1, A, B, C, D, E, X[15], 8, 0x00000000u);
    RMD_STEP(F5, AA, BB, CC, DD, EE, X[5], 8, 0x50A28BE6u);
    RMD_STEP(F5, EE, AA, BB, CC, DD, X[14], 9, 0x50A28BE6u);
    RMD_STEP(F5, DD, EE, AA, BB, CC, X[7], 9, 0x50A28BE6u);
    RMD_STEP(F5, CC, DD, EE, AA, BB, X[0], 11, 0x50A28BE6u);
    RMD_STEP(F5, BB, CC, DD, EE, AA, X[9], 13, 0x50A28BE6u);
    RMD_STEP(F5, AA, BB, CC, DD, EE, X[2], 15, 0x50A28BE6u);
    RMD_STEP(F5, EE, AA, BB, CC, DD, X[11], 15, 0x50A28BE6u);
    RMD_STEP(F5, DD, EE, AA, BB, CC, X[4], 5, 0x50A28BE6u);
    RMD_STEP(F5, CC, DD, EE, AA, BB, X[13], 7, 0x50A28BE6u);
    RMD_STEP(F5, BB, CC, DD, EE, AA, X[6], 7, 0x50A28BE6u);
    RMD_STEP(F5, AA, BB, CC, DD, EE, X[15], 8, 0x50A28BE6u);
    RMD_STEP(F5, EE, AA, BB, CC, DD, X[8], 11, 0x50A28BE6u);
    RMD_STEP(F5, DD, EE, AA, BB, CC, X[1], 14, 0x50A28BE6u);
    RMD_STEP(F5, CC, DD, EE, AA, BB, X[10], 14, 0x50A28BE6u);
    RMD_STEP(F5, BB, CC, DD, EE, AA, X[3], 12, 0x50A28BE6u);
    RMD_STEP(F5, AA, BB, CC, DD, EE, X[12], 6, 0x50A28BE6u);
    { uint T = A; A = AA; AA = T; }
    /* j=16...31 */
    RMD_STEP(F2, E, A, B, C, D, X[7], 7, 0x5A827999u);
    RMD_STEP(F2, D, E, A, B, C, X[4], 6, 0x5A827999u);
    RMD_STEP(F2, C, D, E, A, B, X[13], 8, 0x5A827999u);
    RMD_STEP(F2, B, C, D, E, A, X[1], 13, 0x5A827999u);
    RMD_STEP(F2, A, B, C, D, E, X[10], 11, 0x5A827999u);
    RMD_STEP(F2, E, A, B, C, D, X[6], 9, 0x5A827999u);
    RMD_STEP(F2, D, E, A, B, C, X[15], 7, 0x5A827999u);
    RMD_STEP(F2, C, D, E, A, B, X[3], 15, 0x5A827999u);
    RMD_STEP(F2, B, C, D, E, A, X[12], 7, 0x5A827999u);
    RMD_STEP(F2, A, B, C, D, E, X[0], 12, 0x5A827999u);
    RMD_STEP(F2, E, A, B, C, D, X[9], 15, 0x5A827999u);
    RMD_STEP(F2, D, E, A, B, C, X[5], 9, 0x5A827999u);
    RMD_STEP(F2, C, D, E, A, B, X[2], 11, 0x5A827999u);
    RMD_STEP(F2, B, C, D, E, A, X[14], 7, 0x5A827999u);
    RMD_STEP(F2, A, B, C, D, E, X[11], 13, 0x5A827999u);
    RMD_STEP(F2, E, A, B, C, D, X[8], 12, 0x5A827999u);
    RMD_STEP(F4, EE, AA, BB, CC, DD, X[6], 9, 0x5C4DD124u);
    RMD_STEP(F4, DD, EE, AA, BB, CC, X[11], 13, 0x5C4DD124u);
    RMD_STEP(F4, CC, DD, EE, AA, BB, X[3], 15, 0x5C4DD124u);
    RMD_STEP(F4, BB, CC, DD, EE, AA, X[7], 7, 0x5C4DD124u);
    RMD_STEP(F4, AA, BB, CC, DD, EE, X[0], 12, 0x5C4DD124u);
    RMD_STEP(F4, EE, AA, BB, CC, DD, X[13], 8, 0x5C4DD124u);
    RMD_STEP(F4, DD, EE, AA, BB, CC, X[5], 9, 0x5C4DD124u);
    RMD_STEP(F4, CC, DD, EE, AA, BB, X[10], 11, 0x5C4DD124u);
    RMD_STEP(F4, BB, CC, DD, EE, AA, X[14], 7, 0x5C4DD124u);
    RMD_STEP(F4, AA, BB, CC, DD, EE, X[15], 7, 0x5C4DD124u);
    RMD_STEP(F4, EE, AA, BB, CC, DD, X[8], 12, 0x5C4DD124u);
    RMD_STEP(F4, DD, EE, AA, BB, CC, X[12], 7, 0x5C4DD124u);
    RMD_STEP(F4, CC, DD, EE, AA, BB, X[4], 6, 0x5C4DD124u);
    RMD_STEP(F4, BB, CC, DD, EE, AA, X[9], 15, 0x5C4DD124u);
    RMD_STEP(F4, AA, BB, CC, DD, EE, X[1], 13, 0x5C4DD124u);
    RMD_STEP(F4, EE, AA, BB, CC, DD, X[2], 11, 0x5C4DD124u);
    { uint T = B; B = BB; BB = T; }
    /* j=32...47 */
    RMD_STEP(F3, D, E, A, B, C, X[3], 11, 0x6ED9EBA1u);
    RMD_STEP(F3, C, D, E, A, B, X[10], 13, 0x6ED9EBA1u);
    RMD_STEP(F3, B, C, D, E, A, X[14], 6, 0x6ED9EBA1u);
    RMD_STEP(F3, A, B, C, D, E, X[4], 7, 0x6ED9EBA1u);
    RMD_STEP(F3, E, A, B, C, D, X[9], 14, 0x6ED9EBA1u);
    RMD_STEP(F3, D, E, A, B, C, X[15], 9, 0x6ED9EBA1u);
    RMD_STEP(F3, C, D, E, A, B, X[8], 13, 0x6ED9EBA1u);
    RMD_STEP(F3, B, C, D, E, A, X[1], 15, 0x6ED9EBA1u);
    RMD_STEP(F3, A, B, C, D, E, X[2], 14, 0x6ED9EBA1u);
    RMD_STEP(F3, E, A, B, C, D, X[7], 8, 0x6ED9EBA1u);
    RMD_STEP(F3, D, E, A, B, C, X[0], 13, 0x6ED9EBA1u);
    RMD_STEP(F3, C, D, E, A, B, X[6], 6, 0x6ED9EBA1u);
    RMD_STEP(F3, B, C, D, E, A, X[13], 5, 0x6ED9EBA1u);
    RMD_STEP(F3, A, B, C, D, E, X[11], 12, 0x6ED9EBA1u);
    RMD_STEP(F3, E, A, B, C, D, X[5], 7, 0x6ED9EBA1u);
    RMD_STEP(F3, D, E, A, B, C, X[12], 5, 0x6ED9EBA1u);
    RMD_STEP(F3, DD, EE, AA, BB, CC, X[15], 9, 0x6D703EF3u);
    RMD_STEP(F3, CC, DD, EE, AA, BB, X[5], 7, 0x6D703EF3u);
    RMD_STEP(F3, BB, CC, DD, EE, AA, X[1], 15, 0x6D703EF3u);
    RMD_STEP(F3, AA, BB, CC, DD, EE, X[3], 11, 0x6D703EF3u);
    RMD_STEP(F3, EE, AA, BB, CC, DD, X[7], 8, 0x6D703EF3u);
    RMD_STEP(F3, DD, EE, AA, BB, CC, X[14], 6, 0x6D703EF3u);
    RMD_STEP(F3, CC, DD, EE, AA, BB, X[6], 6, 0x6D703EF3u);
    RMD_STEP(F3, BB, CC, DD, EE, AA, X[9], 14, 0x6D703EF3u);
    RMD_STEP(F3, AA, BB, CC, DD, EE, X[11], 12, 0x6D703EF3u);
    RMD_STEP(F3, EE, AA, BB, CC, DD, X[8], 13, 0x6D703EF3u);
    RMD_STEP(F3, DD, EE, AA, BB, CC, X[12], 5, 0x6D703EF3u);
    RMD_STEP(F3, CC, DD, EE, AA, BB, X[2], 14, 0x6D703EF3u);
    RMD_STEP(F3, BB, CC, DD, EE, AA, X[10], 13, 0x6D703EF3u);
    RMD_STEP(F3, AA, BB, CC, DD, EE, X[0], 13, 0x6D703EF3u);
    RMD_STEP(F3, EE, AA, BB, CC, DD, X[4], 7, 0x6D703EF3u);
    RMD_STEP(F3, DD, EE, AA, BB, CC, X[13], 5, 0x6D703EF3u);
    { uint T = C; C = CC; CC = T; }
    /* j=48...63 */
    RMD_STEP(F4, C, D, E, A, B, X[1], 11, 0x8F1BBCDCu);
    RMD_STEP(F4, B, C, D, E, A, X[9], 12, 0x8F1BBCDCu);
    RMD_STEP(F4, A, B, C, D, E, X[11], 14, 0x8F1BBCDCu);
    RMD_STEP(F4, E, A, B, C, D, X[10], 15, 0x8F1BBCDCu);
    RMD_STEP(F4, D, E, A, B, C, X[0], 14, 0x8F1BBCDCu);
    RMD_STEP(F4, C, D, E, A, B, X[8], 15, 0x8F1BBCDCu);
    RMD_STEP(F4, B, C, D, E, A, X[12], 9, 0x8F1BBCDCu);
    RMD_STEP(F4, A, B, C, D, E, X[4], 8, 0x8F1BBCDCu);
    RMD_STEP(F4, E, A, B, C, D, X[13], 9, 0x8F1BBCDCu);
    RMD_STEP(F4, D, E, A, B, C, X[3], 14, 0x8F1BBCDCu);
    RMD_STEP(F4, C, D, E, A, B, X[7], 5, 0x8F1BBCDCu);
    RMD_STEP(F4, B, C, D, E, A, X[15], 6, 0x8F1BBCDCu);
    RMD_STEP(F4, A, B, C, D, E, X[14], 8, 0x8F1BBCDCu);
    RMD_STEP(F4, E, A, B, C, D, X[5], 6, 0x8F1BBCDCu);
    RMD_STEP(F4, D, E, A, B, C, X[6], 5, 0x8F1BBCDCu);
    RMD_STEP(F4, C, D, E, A, B, X[2], 12, 0x8F1BBCDCu);
    RMD_STEP(F2, CC, DD, EE, AA, BB, X[8], 15, 0x7A6D76E9u);
    RMD_STEP(F2, BB, CC, DD, EE, AA, X[6], 5, 0x7A6D76E9u);
    RMD_STEP(F2, AA, BB, CC, DD, EE, X[4], 8, 0x7A6D76E9u);
    RMD_STEP(F2, EE, AA, BB, CC, DD, X[1], 11, 0x7A6D76E9u);
    RMD_STEP(F2, DD, EE, AA, BB, CC, X[3], 14, 0x7A6D76E9u);
    RMD_STEP(F2, CC, DD, EE, AA, BB, X[11], 14, 0x7A6D76E9u);
    RMD_STEP(F2, BB, CC, DD, EE, AA, X[15], 6, 0x7A6D76E9u);
    RMD_STEP(F2, AA, BB, CC, DD, EE, X[0], 14, 0x7A6D76E9u);
    RMD_STEP(F2, EE, AA, BB, CC, DD, X[5], 6, 0x7A6D76E9u);
    RMD_STEP(F2, DD, EE, AA, BB, CC, X[12], 9, 0x7A6D76E9u);
    RMD_STEP(F2, CC, DD, EE, AA, BB, X[2], 12, 0x7A6D76E9u);
    RMD_STEP(F2, BB, CC, DD, EE, AA, X[13], 9, 0x7A6D76E9u);
    RMD_STEP(F2, AA, BB, CC, DD, EE, X[9], 12, 0x7A6D76E9u);
    RMD_STEP(F2, EE, AA, BB, CC, DD, X[7], 5, 0x7A6D76E9u);
    RMD_STEP(F2, DD, EE, AA, BB, CC, X[10], 15, 0x7A6D76E9u);
    RMD_STEP(F2, CC, DD, EE, AA, BB, X[14], 8, 0x7A6D76E9u);
    { uint T = D; D = DD; DD = T; }
    /* j=64...79 */
    RMD_STEP(F5, B, C, D, E, A, X[4], 9, 0xA953FD4Eu);
    RMD_STEP(F5, A, B, C, D, E, X[0], 15, 0xA953FD4Eu);
    RMD_STEP(F5, E, A, B, C, D, X[5], 5, 0xA953FD4Eu);
    RMD_STEP(F5, D, E, A, B, C, X[9], 11, 0xA953FD4Eu);
    RMD_STEP(F5, C, D, E, A, B, X[7], 6, 0xA953FD4Eu);
    RMD_STEP(F5, B, C, D, E, A, X[12], 8, 0xA953FD4Eu);
    RMD_STEP(F5, A, B, C, D, E, X[2], 13, 0xA953FD4Eu);
    RMD_STEP(F5, E, A, B, C, D, X[10], 12, 0xA953FD4Eu);
    RMD_STEP(F5, D, E, A, B, C, X[14], 5, 0xA953FD4Eu);
    RMD_STEP(F5, C, D, E, A, B, X[1], 12, 0xA953FD4Eu);
    RMD_STEP(F5, B, C, D, E, A, X[3], 13, 0xA953FD4Eu);
    RMD_STEP(F5, A, B, C, D, E, X[8], 14, 0xA953FD4Eu);
    RMD_STEP(F5, E, A, B, C, D, X[11], 11, 0xA953FD4Eu);
    RMD_STEP(F5, D, E, A, B, C, X[6], 8, 0xA953FD4Eu);
    RMD_STEP(F5, C, D, E, A, B, X[15], 5, 0xA953FD4Eu);
    RMD_STEP(F5, B, C, D, E, A, X[13], 6, 0xA953FD4Eu);
    RMD_STEP(F1, BB, CC, DD, EE, AA, X[12], 8, 0x00000000u);
    RMD_STEP(F1, AA, BB, CC, DD, EE, X[15], 5, 0x00000000u);
    RMD_STEP(F1, EE, AA, BB, CC, DD, X[10], 12, 0x00000000u);
    RMD_STEP(F1, DD, EE, AA, BB, CC, X[4], 9, 0x00000000u);
    RMD_STEP(F1, CC, DD, EE, AA, BB, X[1], 12, 0x00000000u);
    RMD_STEP(F1, BB, CC, DD, EE, AA, X[5], 5, 0x00000000u);
    RMD_STEP(F1, AA, BB, CC, DD, EE, X[8], 14, 0x00000000u);
    RMD_STEP(F1, EE, AA, BB, CC, DD, X[7], 6, 0x00000000u);
    RMD_STEP(F1, DD, EE, AA, BB, CC, X[6], 8, 0x00000000u);
    RMD_STEP(F1, CC, DD, EE, AA, BB, X[2], 13, 0x00000000u);
    RMD_STEP(F1, BB, CC, DD, EE, AA, X[13], 6, 0x00000000u);
    RMD_STEP(F1, AA, BB, CC, DD, EE, X[14], 5, 0x00000000u);
    RMD_STEP(F1, EE, AA, BB, CC, DD, X[0], 15, 0x00000000u);
    RMD_STEP(F1, DD, EE, AA, BB, CC, X[3], 13, 0x00000000u);
    RMD_STEP(F1, CC, DD, EE, AA, BB, X[9], 11, 0x00000000u);
    RMD_STEP(F1, BB, CC, DD, EE, AA, X[11], 11, 0x00000000u);
    { uint T = E; E = EE; EE = T; }

    hash[0] += A;  hash[1] += B;  hash[2] += C;  hash[3] += D;  hash[4] += EE;
    hash[5] += AA; hash[6] += BB; hash[7] += CC; hash[8] += DD; hash[9] += E;
}

/* Copy global bytes into LE M[] */
static void rmd320_copy_g(thread uint *M, int off, device const uint8_t *src, int len) {
    thread uchar *mb = (thread uchar *)M;
    for (int i = 0; i < len; i++) mb[off + i] = src[i];
}

/* SALTED_PARAMS provided by metal_common.metal */

/* PROBE7_NOOVF provided by metal_common.metal */
#define PROBE7(oh0,oh1,oh2,oh3,widx,sidx) { \
    uint4 _ph = uint4(oh0,oh1,oh2,oh3); \
    PROBE7_NOOVF(_ph, widx, sidx) }

#define RMD320_IV \
    { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u, \
      0x76543210u, 0xFEDCBA98u, 0x89ABCDEFu, 0x01234567u, 0x3C2D1E0Fu }

#define HMAC_RMD320_BODY(kdata, klen_v, mdata, mlen_v) \
    uint key_block[16]; \
    for (int i = 0; i < 16; i++) key_block[i] = 0; \
    int klen = klen_v; \
    if (klen > 64) { \
        uint kstate[10] = RMD320_IV; \
        uint KM[16]; for (int i = 0; i < 16; i++) KM[i] = 0; \
        rmd320_copy_g(KM, 0, kdata, (klen < 64) ? klen : 64); \
        if (klen <= 55) { ((thread uchar *)KM)[klen] = 0x80; KM[14] = klen * 8; } \
        rmd320_block(kstate, KM); \
        if (klen > 55) { \
            for (int i = 0; i < 16; i++) KM[i] = 0; \
            int rem = klen - 64; \
            if (rem > 0) rmd320_copy_g(KM, 0, kdata + 64, rem); \
            ((thread uchar *)KM)[rem] = 0x80; \
            KM[14] = klen * 8; \
            rmd320_block(kstate, KM); \
        } \
        for (int i = 0; i < 10; i++) key_block[i] = kstate[i]; \
        klen = 40; \
    } else { \
        thread uchar *kb = (thread uchar *)key_block; \
        for (int i = 0; i < klen; i++) kb[i] = kdata[i]; \
    } \
    /* Inner: RMD320((key ^ ipad) || message) */ \
    uint ipad[16], M[16]; \
    for (int i = 0; i < 16; i++) ipad[i] = key_block[i] ^ 0x36363636u; \
    for (int i = 0; i < 16; i++) M[i] = ipad[i]; \
    uint istate[10] = RMD320_IV; \
    rmd320_block(istate, M); \
    int mlen = mlen_v; \
    for (int i = 0; i < 16; i++) M[i] = 0; \
    if (mlen <= 55) { \
        rmd320_copy_g(M, 0, mdata, mlen); \
        ((thread uchar *)M)[mlen] = 0x80; \
        M[14] = (64 + mlen) * 8; \
        rmd320_block(istate, M); \
    } else { \
        rmd320_copy_g(M, 0, mdata, (mlen < 64) ? mlen : 64); \
        if (mlen < 64) ((thread uchar *)M)[mlen] = 0x80; \
        rmd320_block(istate, M); \
        for (int i = 0; i < 16; i++) M[i] = 0; \
        int rem = mlen - 64; \
        if (rem > 0) rmd320_copy_g(M, 0, mdata + 64, rem); \
        if (mlen >= 64) ((thread uchar *)M)[(rem > 0) ? rem : 0] = 0x80; \
        M[14] = (64 + mlen) * 8; \
        rmd320_block(istate, M); \
    } \
    /* Outer: RMD320((key ^ opad) || inner_hash) */ \
    uint opad[16]; \
    for (int i = 0; i < 16; i++) opad[i] = key_block[i] ^ 0x5c5c5c5cu; \
    for (int i = 0; i < 16; i++) M[i] = opad[i]; \
    uint ostate[10] = RMD320_IV; \
    rmd320_block(ostate, M); \
    /* inner hash (40 bytes LE) + padding */ \
    for (int i = 0; i < 10; i++) M[i] = istate[i]; \
    ((thread uchar *)M)[40] = 0x80; \
    for (int i = 11; i < 14; i++) M[i] = 0; \
    M[14] = (64 + 40) * 8; \
    M[15] = 0; \
    rmd320_block(ostate, M); \
    uint hx = ostate[0], hy = ostate[1], hz = ostate[2], hw = ostate[3];

kernel void hmac_rmd320_ksalt_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int klen_v = salt_lens[salt_idx];
    device const uint8_t *key = salts + soff;

    HMAC_RMD320_BODY(key, klen_v, pass, plen)
    PROBE7(hx, hy, hz, hw, word_idx, salt_idx)
}

kernel void hmac_rmd320_kpass_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int klen_v = hexlens[word_idx];
    device const uint8_t *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen_v = salt_lens[salt_idx];
    device const uint8_t *msg = salts + soff;

    HMAC_RMD320_BODY(key, klen_v, msg, mlen_v)
    PROBE7(hx, hy, hz, hw, word_idx, salt_idx)
}
