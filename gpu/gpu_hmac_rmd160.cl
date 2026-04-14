/* gpu_hmac_rmd160.cl -- HMAC-RIPEMD-160 GPU kernels
 *
 * hmac_rmd160_ksalt_batch: HMAC-RMD160 key=$salt (e211, hashcat 6060)
 *   Key = salt (from Typeuser), Message = password
 *
 * hmac_rmd160_kpass_batch: HMAC-RMD160 key=$pass (e798, hashcat 6050)
 *   Key = password, Message = salt (from Typesalt)
 *
 * HMAC(K, M) = RMD160((K ^ opad) || RMD160((K ^ ipad) || M))
 * Block size = 64 bytes, output = 20 bytes (5 x uint32)
 * ipad = 0x36, opad = 0x5c
 * RIPEMD-160 is little-endian -- no bswap needed for key/ipad/opad blocks.
 *
 * Hit stride: 7 (word_idx, salt_idx, iter, h0, h1, h2, h3)
 * Compact probe: h[0..3] directly (LE, no bswap)
 *
 * Primitives (RMD macros, L1-L5, R1-R5, rmd160_block) provided by gpu_common.cl
 */

/* ---- HMAC-RMD160 key=$salt (e211, hashcat 6060) ---- */

/* Key = salt, Message = password.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_rmd160_ksalt_batch(
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
    int klen = salt_lens[salt_idx];
    __global const uchar *key = salts + soff;

    /* Step 1: Prepare key block (LE uint32 words).
     * If key > 64 bytes, hash it first with RMD160. */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        /* key = RMD160(original_key) */
        uint kh[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        uchar *mb = (uchar *)M;
        for (int i = 0; i < 64; i++) mb[i] = key[i];
        rmd160_block(kh, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = klen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = key[64 + i];
        mb[rem] = 0x80;
        M[14] = klen * 8;
        rmd160_block(kh, M);
        /* RMD160 output is LE -- store directly */
        for (int i = 0; i < 5; i++) key_block[i] = kh[i];
        klen = 20;
    } else {
        uchar *kb = (uchar *)key_block;
        for (int i = 0; i < klen; i++) kb[i] = key[i];
    }

    /* Step 2: Inner hash = RMD160((key ^ ipad) || message) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    /* RMD160 is LE -- ipad goes directly into M[], no bswap */
    for (int i = 0; i < 16; i++) M[i] = ipad[i];

    uint ih[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    rmd160_block(ih, M);  /* Process ipad block (64 bytes) */

    /* Continue with message (password) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    int mlen = plen;
    uchar *mb = (uchar *)M;
    if (mlen <= 55) {
        for (int i = 0; i < mlen; i++) mb[i] = pass[i];
        mb[mlen] = 0x80;
        M[14] = (64 + mlen) * 8;  /* total = ipad_block + message */
        rmd160_block(ih, M);
    } else {
        int copy1 = (mlen < 64) ? mlen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = pass[i];
        if (mlen < 64) mb[mlen] = 0x80;
        rmd160_block(ih, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = mlen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = pass[64 + i];
        if (mlen >= 64) mb[(rem > 0) ? rem : 0] = 0x80;
        M[14] = (64 + mlen) * 8;
        rmd160_block(ih, M);
    }

    /* Inner hash result: ih[0..4] (LE) */

    /* Step 3: Outer hash = RMD160((key ^ opad) || inner_hash) */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = opad_block[i];

    uint oh[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    rmd160_block(oh, M);  /* Process opad block (64 bytes) */

    /* Continue with inner_hash (20 bytes LE) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    M[0] = ih[0]; M[1] = ih[1]; M[2] = ih[2]; M[3] = ih[3]; M[4] = ih[4];
    M[5] = 0x80;  /* 0x80 at byte 20 = word 5, LE byte 0 */
    M[14] = (64 + 20) * 8;  /* 672 bits */
    rmd160_block(oh, M);

    /* Step 4: Check result -- RMD160 is LE, no bswap needed */
    if (probe_compact(oh[0], oh[1], oh[2], oh[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, oh[0], oh[1], oh[2], oh[3])
    }
}

/* ---- HMAC-RMD160 key=$pass (e798, hashcat 6050) ---- */

/* Key = password, Message = salt.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_rmd160_kpass_batch(
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

    int klen = hexlens[word_idx];  /* password = key */
    __global const uchar *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen = salt_lens[salt_idx];  /* salt = message */
    __global const uchar *msg = salts + soff;

    /* Step 1: Prepare key block from password */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        uint kh[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        uchar *mb = (uchar *)M;
        for (int i = 0; i < 64; i++) mb[i] = key[i];
        rmd160_block(kh, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = klen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = key[64 + i];
        mb[rem] = 0x80;
        M[14] = klen * 8;
        rmd160_block(kh, M);
        for (int i = 0; i < 5; i++) key_block[i] = kh[i];
        klen = 20;
    } else {
        uchar *kb = (uchar *)key_block;
        for (int i = 0; i < klen; i++) kb[i] = key[i];
    }

    /* Step 2: Inner hash = RMD160((key ^ ipad) || salt) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    for (int i = 0; i < 16; i++) M[i] = ipad[i];

    uint ih[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    rmd160_block(ih, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    uchar *mb = (uchar *)M;
    if (mlen <= 55) {
        for (int i = 0; i < mlen; i++) mb[i] = msg[i];
        mb[mlen] = 0x80;
        M[14] = (64 + mlen) * 8;
        rmd160_block(ih, M);
    } else {
        int copy1 = (mlen < 64) ? mlen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = msg[i];
        if (mlen < 64) mb[mlen] = 0x80;
        rmd160_block(ih, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = mlen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = msg[64 + i];
        if (mlen >= 64) mb[(rem > 0) ? rem : 0] = 0x80;
        M[14] = (64 + mlen) * 8;
        rmd160_block(ih, M);
    }

    /* Step 3: Outer hash */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = opad_block[i];

    uint oh[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    rmd160_block(oh, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    M[0] = ih[0]; M[1] = ih[1]; M[2] = ih[2]; M[3] = ih[3]; M[4] = ih[4];
    M[5] = 0x80;
    M[14] = (64 + 20) * 8;  /* 672 bits */
    rmd160_block(oh, M);

    if (probe_compact(oh[0], oh[1], oh[2], oh[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, oh[0], oh[1], oh[2], oh[3])
    }
}
