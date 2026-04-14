/* metal_sha1.metal — SHA1 salted GPU kernels for Apple Metal
 *
 * Kernels:
 *   sha1passsalt_batch  — sha1($pass.$salt)         (e405)
 *   sha1saltpass_batch  — sha1($salt.$pass)         (e385)
 *   sha1dru_batch       — SHA1DRU (1M iterations)   (e404)
 *   hmac_sha1_ksalt_batch — HMAC-SHA1 key=$salt     (e215)
 *   hmac_sha1_kpass_batch — HMAC-SHA1 key=$pass     (e793)
 *
 * Hit stride: 8 (word_idx, salt_idx, iter_num, h0, h1, h2, h3, h4)
 */

/* bswap32, sha1_compress, S_copy_bytes, S_set_byte, S_copy_bytes_priv,
 * S_set_byte_priv, SALTED_PARAMS, PROBE8 all provided by metal_common.metal */

/* ---- SHA1(password + salt) kernel (e405) ---- */
kernel void sha1passsalt_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    if (total_len <= 55) {
        S_copy_bytes(M, 0, pass, plen);
        S_copy_bytes(M, plen, salts + soff, slen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha1_compress(state, M);
    } else {
        int pass_b1 = (plen < 64) ? plen : 64;
        S_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0) S_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha1_compress(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, 0, pass + pass_b1, pass_b2); pos2 = pass_b2; }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2); pos2 += salt_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha1_compress(state, M);
    }

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    PROBE8(h, word_idx, salt_idx)
}

/* ---- SHA1(salt + password) kernel (e385) ---- */
kernel void sha1saltpass_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    if (total_len <= 55) {
        S_copy_bytes(M, 0, salts + soff, slen);
        S_copy_bytes(M, slen, pass, plen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha1_compress(state, M);
    } else {
        int salt_b1 = (slen < 64) ? slen : 64;
        S_copy_bytes(M, 0, salts + soff, salt_b1);
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0) S_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha1_compress(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2); pos2 = salt_b2; }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, pos2, pass + pass_b1, pass_b2); pos2 += pass_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha1_compress(state, M);
    }

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    PROBE8(h, word_idx, salt_idx)
}

/* ---- SHA1DRU kernel (e404): SHA1 iterated 1,000,001 times ---- */
/* SHA1(pass) already done on CPU. hexhashes[tid*256..+39] = hex(SHA1(pass)),
 * password at offset 40. hexlens[tid] = 40 + plen.
 * GPU_CAT_ITER: dispatched as num_words work items (no salts). */
kernel void sha1dru_batch(SALTED_PARAMS) {
    if (tid >= params.num_words) return;

    device const uint8_t *wordbuf = hexhashes + tid * 256;
    int total_packed = hexlens[tid];
    int plen = total_packed - 40;
    device const uint8_t *pass = wordbuf + 40;

    /* Decode hex ASCII at wordbuf[0..39] back to 5 big-endian state words */
    uint state[5];
    for (int w = 0; w < 5; w++) {
        uint val = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t c = wordbuf[w * 8 + j];
            uint nibble = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            val = (val << 4) | nibble;
        }
        state[w] = val;
    }

    uint8_t hexlut[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

    /* 1,000,000 iterations of SHA1(hex(hash) + pass) */
    for (int iter = 0; iter < 1000000; iter++) {
        uint8_t hexbuf[96];
        for (int w = 0; w < 5; w++) {
            uint sv = state[w];
            hexbuf[w*8+0] = hexlut[(sv >> 28) & 0xf];
            hexbuf[w*8+1] = hexlut[(sv >> 24) & 0xf];
            hexbuf[w*8+2] = hexlut[(sv >> 20) & 0xf];
            hexbuf[w*8+3] = hexlut[(sv >> 16) & 0xf];
            hexbuf[w*8+4] = hexlut[(sv >> 12) & 0xf];
            hexbuf[w*8+5] = hexlut[(sv >> 8) & 0xf];
            hexbuf[w*8+6] = hexlut[(sv >> 4) & 0xf];
            hexbuf[w*8+7] = hexlut[sv & 0xf];
        }
        for (int i = 0; i < plen; i++) hexbuf[40 + i] = pass[i];
        int total = 40 + plen;

        state[0] = 0x67452301u; state[1] = 0xEFCDAB89u;
        state[2] = 0x98BADCFEu; state[3] = 0x10325476u; state[4] = 0xC3D2E1F0u;
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        if (total <= 55) {
            S_copy_bytes_priv(M, 0, hexbuf, total);
            S_set_byte_priv(M, total, 0x80);
            M[15] = total * 8;
            sha1_compress(state, M);
        } else {
            S_copy_bytes_priv(M, 0, hexbuf, 64);
            sha1_compress(state, M);
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = total - 64;
            if (rem > 0) S_copy_bytes_priv(M, 0, hexbuf + 64, rem);
            S_set_byte_priv(M, rem, 0x80);
            M[15] = total * 8;
            sha1_compress(state, M);
        }
    }

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    PROBE8(h, tid, 0)
}

/* ---- HMAC-SHA1 key=$salt (e215, hashcat 160) ---- */
/* Key = salt, Message = password */
kernel void hmac_sha1_ksalt_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int klen = salt_lens[salt_idx];
    device const uint8_t *key = salts + soff;

    /* Prepare key block (LE byte order). If key > 64 bytes, hash it first. */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint KM[16];
        for (int i = 0; i < 16; i++) KM[i] = 0;
        S_copy_bytes(KM, 0, key, 64);
        sha1_compress(kstate, KM);
        for (int i = 0; i < 16; i++) KM[i] = 0;
        int rem = klen - 64;
        if (rem > 0) S_copy_bytes(KM, 0, key + 64, rem);
        S_set_byte(KM, rem, 0x80);
        KM[15] = klen * 8;
        sha1_compress(kstate, KM);
        for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
        klen = 20;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Inner hash = SHA1((key ^ ipad) || message) */
    uint ipad_blk[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad_blk[i] = key_block[i] ^ 0x36363636u;
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad_blk[i]);

    uint istate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(istate, M);

    /* Message = password */
    for (int i = 0; i < 16; i++) M[i] = 0;
    int mlen = plen;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, pass, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_compress(istate, M);
    } else {
        S_copy_bytes(M, 0, pass, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha1_compress(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, pass + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_compress(istate, M);
    }

    /* Outer hash = SHA1((key ^ opad) || inner_hash) */
    uint opad_blk[16];
    for (int i = 0; i < 16; i++)
        opad_blk[i] = key_block[i] ^ 0x5c5c5c5cu;
    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_blk[i]);

    uint ostate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(ostate, M);

    /* Inner hash result (20 bytes BE in istate) + padding */
    for (int i = 0; i < 5; i++) M[i] = istate[i];
    M[5] = 0x80000000u;
    for (int i = 6; i < 15; i++) M[i] = 0;
    M[15] = (64 + 20) * 8;
    sha1_compress(ostate, M);

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(ostate[i]);

    PROBE8(h, word_idx, salt_idx)
}

/* ---- HMAC-SHA1 key=$pass (e793, hashcat 150) ---- */
/* Key = password, Message = salt */
kernel void hmac_sha1_kpass_batch(SALTED_PARAMS) {
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int klen = hexlens[word_idx];
    device const uint8_t *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen = salt_lens[salt_idx];
    device const uint8_t *msg = salts + soff;

    /* Prepare key block from password */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint KM[16];
        for (int i = 0; i < 16; i++) KM[i] = 0;
        S_copy_bytes(KM, 0, key, 64);
        sha1_compress(kstate, KM);
        for (int i = 0; i < 16; i++) KM[i] = 0;
        int rem = klen - 64;
        if (rem > 0) S_copy_bytes(KM, 0, key + 64, rem);
        S_set_byte(KM, rem, 0x80);
        KM[15] = klen * 8;
        sha1_compress(kstate, KM);
        for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
        klen = 20;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Inner hash = SHA1((key ^ ipad) || salt) */
    uint ipad_blk[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad_blk[i] = key_block[i] ^ 0x36363636u;
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad_blk[i]);

    uint istate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, msg, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_compress(istate, M);
    } else {
        S_copy_bytes(M, 0, msg, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha1_compress(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, msg + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_compress(istate, M);
    }

    /* Outer hash */
    uint opad_blk[16];
    for (int i = 0; i < 16; i++)
        opad_blk[i] = key_block[i] ^ 0x5c5c5c5cu;
    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_blk[i]);

    uint ostate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_compress(ostate, M);

    for (int i = 0; i < 5; i++) M[i] = istate[i];
    M[5] = 0x80000000u;
    for (int i = 6; i < 15; i++) M[i] = 0;
    M[15] = (64 + 20) * 8;
    sha1_compress(ostate, M);

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(ostate[i]);

    PROBE8(h, word_idx, salt_idx)
}
