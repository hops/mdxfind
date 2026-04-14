/* bswap32, sha1_block, S_copy_bytes, S_set_byte all provided by gpu_common.cl */

/* ---- SHA1(password + salt) kernel (e405) ---- */
__kernel void sha1passsalt_batch(
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

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;
      if (total_len <= 55) {
        S_copy_bytes(M, 0, pass, plen);
        S_copy_bytes(M, plen, salts + soff, slen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha1_block(state, M);
      } else {
        int pass_b1 = (plen < 64) ? plen : 64;
        S_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0) S_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha1_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, 0, pass + pass_b1, pass_b2); pos2 = pass_b2; }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2); pos2 += salt_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha1_block(state, M);
      }
    }

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_5(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

/* ---- SHA1(salt + password) kernel (e385) ---- */
__kernel void sha1saltpass_batch(
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

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;
      if (total_len <= 55) {
        S_copy_bytes(M, 0, salts + soff, slen);
        S_copy_bytes(M, slen, pass, plen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha1_block(state, M);
      } else {
        int salt_b1 = (slen < 64) ? slen : 64;
        S_copy_bytes(M, 0, salts + soff, salt_b1);
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0) S_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha1_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2); pos2 = salt_b2; }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, pos2, pass + pass_b1, pass_b2); pos2 += pass_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha1_block(state, M);
      }
    }

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_5(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

/* Copy private-memory bytes into big-endian M[] for SHA1/SHA256 */
void S_copy_bytes_priv(uint *M, int byte_off, const uchar *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 4;
        int bi = 3 - ((byte_off + i) % 4);
        M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)src[i] << (bi * 8));
    }
}

void S_set_byte_priv(uint *M, int byte_off, uchar val) {
    int wi = byte_off / 4;
    int bi = 3 - (byte_off % 4);
    M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)val << (bi * 8));
}

/* ---- SHA1DRU kernel (e404): SHA1 iterated 1,000,001 times ---- */
/* SHA1(pass), then 1M iterations of SHA1(hex(hash) + pass).
 * No salts — GPU_CAT_ITER, dispatched as num_words work items. */
__kernel void sha1dru_batch(
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
    if (tid >= params.num_words) return;

    /* hexhash buffer: [0..39] = hex(SHA1(pass)), [40..40+plen-1] = raw password */
    __global const uchar *wordbuf = hexhashes + tid * 256;
    int plen = hexlens[tid]; /* passlen stored in clen[] on host */

    /* Read password length from the passbuf area — stored after hex */
    /* Actually plen here is hexlen=40, we need the raw pass length.
     * The raw password starts at offset 40, its length = job->clen.
     * But we only have hexlens[] which was set to 40. We need clen.
     * Store plen in the passlen array instead — gpujob reads it. */
    /* For now: scan for the end of the password in the buffer */
    /* Actually: the pass length is encoded as total - 40 where total is
     * stored... let me use a simpler approach. The CPU packs:
     * hexlen[idx] = 40 (the hex portion). Password is at wordbuf+40
     * and its length is in g->passlen[idx] which maps to... we don't
     * have a separate buffer for passlen on the GPU.
     *
     * Simplest: pack total length (40 + plen) in hexlens, kernel subtracts 40. */

    /* hexlens[tid] = 40; password at wordbuf+40, length = clen from host */
    /* We'll change hexlen to 40+plen so kernel can derive plen */
    int total_packed = hexlens[tid]; /* set to 40 + plen by host */
    plen = total_packed - 40;
    __global const uchar *pass = wordbuf + 40;

    /* SHA1(pass) already done on CPU — state is in hex at wordbuf[0..39] */
    /* Decode hex back to 5 big-endian state words */
    uint state[5];
    for (int w = 0; w < 5; w++) {
        __global const uint *hw = (__global const uint *)(wordbuf + w * 8);
        /* hex chars are ASCII in LE uint32 words — reconstruct the big-endian SHA1 word */
        /* Actually the hex is stored as raw ASCII bytes. Parse them. */
        uint val = 0;
        for (int j = 0; j < 8; j++) {
            uchar c = wordbuf[w*8+j];
            uint nibble = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            val = (val << 4) | nibble;
        }
        state[w] = val;
    }

    uchar hexlut[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

    /* 1,000,000 iterations of SHA1(hex(hash) + pass) */
    for (int iter = 0; iter < 1000000; iter++) {
        uchar hexbuf[96];
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

        /* SHA1(hexbuf, total) */
        state[0] = 0x67452301u; state[1] = 0xEFCDAB89u;
        state[2] = 0x98BADCFEu; state[3] = 0x10325476u; state[4] = 0xC3D2E1F0u;
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        if (total <= 55) {
            S_copy_bytes_priv(M, 0, hexbuf, total);
            S_set_byte_priv(M, total, 0x80);
            M[15] = total * 8;
            sha1_block(state, M);
        } else {
            /* Two blocks */
            S_copy_bytes_priv(M, 0, hexbuf, 64);
            sha1_block(state, M);
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = total - 64;
            if (rem > 0) S_copy_bytes_priv(M, 0, hexbuf + 64, rem);
            S_set_byte_priv(M, rem, 0x80);
            M[15] = total * 8;
            sha1_block(state, M);
        }
    }

    /* Byte-swap state for compact table probe */
    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(state[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_5(hits, hit_count, params.max_hits, tid, 0, 1, h)
    }
}

/* gpu_hmac_sha1.cl — HMAC-SHA1 GPU kernels
 *
 * hmac_sha1_ksalt_batch: HMAC-SHA1 key=$salt (e215, hashcat 160)
 *   Key = salt (from Typeuser), Message = password
 *
 * hmac_sha1_kpass_batch: HMAC-SHA1 key=$pass (e793, hashcat 150)
 *   Key = password, Message = salt (from Typesalt)
 *
 * HMAC(K, M) = H((K ^ opad) || H((K ^ ipad) || M))
 * Block size = 64 bytes, ipad = 0x36, opad = 0x5c
 * SHA1 is big-endian — bswap needed for ipad/opad blocks.
 *
 * Hit stride: 8 (word_idx, salt_idx, iter, h[0..4])
 */

/* HMAC-SHA1 key=$salt: key=salt, message=password.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_sha1_ksalt_batch(
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

    /* Step 1: Prepare key block (LE byte order, will bswap for SHA1).
     * If key > 64 bytes, hash it first. */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        /* key = SHA1(original_key) */
        uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, 64);
        sha1_block(kstate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = klen - 64;
        if (rem > 0) S_copy_bytes(M, 0, key + 64, rem);
        S_set_byte(M, rem, 0x80);
        M[15] = klen * 8;
        sha1_block(kstate, M);
        /* Store hashed key (20 bytes BE) as LE bytes in key_block */
        for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
        klen = 20;
    } else {
        /* Copy key bytes into key_block as little-endian uint32 */
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Step 2: Inner hash = SHA1((key ^ ipad) || message) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    /* Convert ipad from LE to BE for SHA1 */
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);

    uint istate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_block(istate, M);  /* Process ipad block (64 bytes) */

    /* Continue with message (password) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    int mlen = plen;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, pass, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;  /* total = ipad_block + message */
        sha1_block(istate, M);
    } else {
        S_copy_bytes(M, 0, pass, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha1_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, pass + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_block(istate, M);
    }

    /* Inner hash result (20 bytes, big-endian in istate) */

    /* Step 3: Outer hash = SHA1((key ^ opad) || inner_hash) */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);

    uint ostate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_block(ostate, M);  /* Process opad block (64 bytes) */

    /* Continue with inner_hash (20 bytes BE) + padding.
     * istate[] is already in BE — copy directly into M[]. */
    for (int i = 0; i < 5; i++) M[i] = istate[i];
    M[5] = 0x80000000u;
    for (int i = 6; i < 15; i++) M[i] = 0;
    M[15] = (64 + 20) * 8;  /* opad_block(64) + inner_hash(20) = 84 bytes = 672 bits */
    sha1_block(ostate, M);

    /* Step 4: Check result — bswap for compact table (LE) */
    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_5(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

/* HMAC-SHA1 key=$pass: key=password, message=salt.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_sha1_kpass_batch(
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
        uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, 64);
        sha1_block(kstate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = klen - 64;
        if (rem > 0) S_copy_bytes(M, 0, key + 64, rem);
        S_set_byte(M, rem, 0x80);
        M[15] = klen * 8;
        sha1_block(kstate, M);
        for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
        klen = 20;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Step 2: Inner hash = SHA1((key ^ ipad) || salt) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);

    uint istate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, msg, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_block(istate, M);
    } else {
        S_copy_bytes(M, 0, msg, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha1_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, msg + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha1_block(istate, M);
    }

    /* Step 3: Outer hash */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);

    uint ostate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    sha1_block(ostate, M);

    for (int i = 0; i < 5; i++) M[i] = istate[i];
    M[5] = 0x80000000u;
    for (int i = 6; i < 15; i++) M[i] = 0;
    M[15] = (64 + 20) * 8;  /* 672 bits */
    sha1_block(ostate, M);

    uint h[5];
    for (int i = 0; i < 5; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_5(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}
