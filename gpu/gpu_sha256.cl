/* SHA256_K[], S256_* macros, sha256_block(), bswap32(), S_copy_bytes(),
 * S_set_byte(), M_copy_bytes(), M_set_byte() all provided by gpu_common.cl */

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
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
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
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}
/* gpu_hmac_sha256.cl — HMAC-SHA256 GPU kernels
 *
 * hmac_sha256_ksalt_batch: HMAC-SHA256 key=$salt (e217, hashcat 1460)
 *   Key = salt (from Typeuser), Message = password
 *
 * hmac_sha256_kpass_batch: HMAC-SHA256 key=$pass (e795, hashcat 1450)
 *   Key = password, Message = salt (from Typesalt)
 *
 * HMAC(K, M) = H((K ⊕ opad) || H((K ⊕ ipad) || M))
 * Block size = 64 bytes, ipad = 0x36, opad = 0x5c
 *
 * Hit stride: 11 (word_idx, salt_idx, iter, h[0..7])
 */

/* HMAC-SHA256 key=$salt: key=salt, message=password.
 * Dispatch: num_words × num_salts threads. */
__kernel void hmac_sha256_ksalt_batch(
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

    /* Step 1: Prepare key block.
     * If key > 64 bytes, hash it. For typical salts this won't happen. */
    uint key_block[16];  /* 64 bytes as uint32 */
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        /* key = SHA256(original_key) */
        uint kstate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                           0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, (klen < 64) ? klen : 64);
        if (klen <= 55) { S_set_byte(M, klen, 0x80); M[15] = klen * 8; }
        sha256_block(kstate, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 64;
            if (rem > 0) S_copy_bytes(M, 0, key + 64, rem);
            S_set_byte(M, rem, 0x80);
            M[15] = klen * 8;
            sha256_block(kstate, M);
        }
        /* Store hashed key (32 bytes BE) as LE bytes in key_block */
        for (int i = 0; i < 8; i++) key_block[i] = bswap32(kstate[i]);
        klen = 32;
    } else {
        /* Copy key bytes into key_block as little-endian uint32 */
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Step 2: Inner hash = SHA256((key ⊕ ipad) || message) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    /* Convert ipad to big-endian for SHA256 */
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);

    uint istate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    sha256_block(istate, M);  /* Process ipad block (64 bytes) */

    /* Continue with message (password) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    int mlen = plen;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, pass, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;  /* total = ipad_block + message */
        sha256_block(istate, M);
    } else {
        S_copy_bytes(M, 0, pass, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha256_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, pass + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha256_block(istate, M);
    }

    /* Inner hash result (32 bytes, big-endian in istate) */

    /* Step 3: Outer hash = SHA256((key ⊕ opad) || inner_hash) */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);

    uint ostate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    sha256_block(ostate, M);  /* Process opad block (64 bytes) */

    /* Continue with inner_hash (32 bytes BE) + padding.
     * istate[] is already in BE — copy directly into M[]. */
    for (int i = 0; i < 8; i++) M[i] = istate[i];
    M[8] = 0x80000000u;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = (64 + 32) * 8;  /* opad_block(64) + inner_hash(32) = 96 bytes */
    sha256_block(ostate, M);

    /* Step 4: Check result */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

/* HMAC-SHA256 key=$pass: key=password, message=salt.
 * Dispatch: num_words × num_salts threads. */
__kernel void hmac_sha256_kpass_batch(
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
        uint kstate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                           0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, (klen < 64) ? klen : 64);
        if (klen <= 55) { S_set_byte(M, klen, 0x80); M[15] = klen * 8; }
        sha256_block(kstate, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 64;
            if (rem > 0) S_copy_bytes(M, 0, key + 64, rem);
            S_set_byte(M, rem, 0x80);
            M[15] = klen * 8;
            sha256_block(kstate, M);
        }
        for (int i = 0; i < 8; i++) key_block[i] = bswap32(kstate[i]);
        klen = 32;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 2;
            int bi = (i & 3) << 3;
            key_block[wi] |= ((uint)key[i]) << bi;
        }
    }

    /* Step 2: Inner hash = SHA256((key ⊕ ipad) || salt) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);

    uint istate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    sha256_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, msg, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;
        sha256_block(istate, M);
    } else {
        S_copy_bytes(M, 0, msg, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha256_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, msg + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha256_block(istate, M);
    }

    /* Step 3: Outer hash */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);

    uint ostate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    sha256_block(ostate, M);

    for (int i = 0; i < 8; i++) M[i] = istate[i];
    M[8] = 0x80000000u;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = (64 + 32) * 8;
    sha256_block(ostate, M);

    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

/* HMAC-SHA224 key=$salt (e216, hashcat na) and key=$pass (e794, hashcat na)
 * Same compress as SHA256, SHA224 IV, inner hash = 28 bytes.
 * Hit stride: 11 (reuse SHA256 stride, extra word unused) */

#define SHA224_IV0 0xc1059ed8u
#define SHA224_IV1 0x367cd507u
#define SHA224_IV2 0x3070dd17u
#define SHA224_IV3 0xf70e5939u
#define SHA224_IV4 0xffc00b31u
#define SHA224_IV5 0x68581511u
#define SHA224_IV6 0x64f98fa7u
#define SHA224_IV7 0xbefa4fa4u

__kernel void hmac_sha224_ksalt_batch(
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

    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;
    if (klen > 64) {
        uint kstate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                           SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
        uint M[16]; for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, (klen < 64) ? klen : 64);
        if (klen <= 55) { S_set_byte(M, klen, 0x80); M[15] = klen * 8; }
        sha256_block(kstate, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            S_copy_bytes(M, 0, key + 64, klen - 64);
            S_set_byte(M, klen - 64, 0x80);
            M[15] = klen * 8;
            sha256_block(kstate, M);
        }
        for (int i = 0; i < 7; i++) key_block[i] = bswap32(kstate[i]);
        klen = 28;
    } else {
        for (int i = 0; i < klen; i++)
            key_block[i >> 2] |= ((uint)key[i]) << ((i & 3) << 3);
    }

    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++) ipad[i] = key_block[i] ^ 0x36363636u;
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);
    uint istate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                       SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
    sha256_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (plen <= 55) {
        S_copy_bytes(M, 0, pass, plen);
        S_set_byte(M, plen, 0x80);
        M[15] = (64 + plen) * 8;
        sha256_block(istate, M);
    } else {
        S_copy_bytes(M, 0, pass, (plen < 64) ? plen : 64);
        if (plen < 64) S_set_byte(M, plen, 0x80);
        sha256_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = plen - 64;
        if (rem > 0) S_copy_bytes(M, 0, pass + 64, rem);
        if (plen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + plen) * 8;
        sha256_block(istate, M);
    }

    /* Outer: SHA224(opad || 28-byte inner_hash) */
    uint opad_block[16];
    for (int i = 0; i < 16; i++) opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;
    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);
    uint ostate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                       SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
    sha256_block(ostate, M);

    for (int i = 0; i < 7; i++) M[i] = istate[i];  /* 28 bytes BE */
    M[7] = 0x80000000u;  /* 0x80 at byte 28 */
    for (int i = 8; i < 15; i++) M[i] = 0;
    M[15] = (64 + 28) * 8;  /* 736 */
    sha256_block(ostate, M);

    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}

__kernel void hmac_sha224_kpass_batch(
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

    int klen = hexlens[word_idx];
    __global const uchar *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen = salt_lens[salt_idx];
    __global const uchar *msg = salts + soff;

    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;
    if (klen > 64) {
        uint kstate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                           SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
        uint M[16]; for (int i = 0; i < 16; i++) M[i] = 0;
        S_copy_bytes(M, 0, key, (klen < 64) ? klen : 64);
        if (klen <= 55) { S_set_byte(M, klen, 0x80); M[15] = klen * 8; }
        sha256_block(kstate, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            S_copy_bytes(M, 0, key + 64, klen - 64);
            S_set_byte(M, klen - 64, 0x80);
            M[15] = klen * 8;
            sha256_block(kstate, M);
        }
        for (int i = 0; i < 7; i++) key_block[i] = bswap32(kstate[i]);
        klen = 28;
    } else {
        for (int i = 0; i < klen; i++)
            key_block[i >> 2] |= ((uint)key[i]) << ((i & 3) << 3);
    }

    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++) ipad[i] = key_block[i] ^ 0x36363636u;
    for (int i = 0; i < 16; i++) M[i] = bswap32(ipad[i]);
    uint istate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                       SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
    sha256_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 55) {
        S_copy_bytes(M, 0, msg, mlen);
        S_set_byte(M, mlen, 0x80);
        M[15] = (64 + mlen) * 8;
        sha256_block(istate, M);
    } else {
        S_copy_bytes(M, 0, msg, (mlen < 64) ? mlen : 64);
        if (mlen < 64) S_set_byte(M, mlen, 0x80);
        sha256_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 64;
        if (rem > 0) S_copy_bytes(M, 0, msg + 64, rem);
        if (mlen >= 64) S_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (64 + mlen) * 8;
        sha256_block(istate, M);
    }

    uint opad_block[16];
    for (int i = 0; i < 16; i++) opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;
    for (int i = 0; i < 16; i++) M[i] = bswap32(opad_block[i]);
    uint ostate[8] = { SHA224_IV0,SHA224_IV1,SHA224_IV2,SHA224_IV3,
                       SHA224_IV4,SHA224_IV5,SHA224_IV6,SHA224_IV7 };
    sha256_block(ostate, M);

    for (int i = 0; i < 7; i++) M[i] = istate[i];
    M[7] = 0x80000000u;
    for (int i = 8; i < 15; i++) M[i] = 0;
    M[15] = (64 + 28) * 8;
    sha256_block(ostate, M);

    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(ostate[i]);

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, h)
    }
}
