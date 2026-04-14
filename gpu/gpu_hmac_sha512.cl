/* gpu_hmac_sha512.cl — HMAC-SHA512 GPU kernels
 *
 * hmac_sha512_ksalt_batch: key=$salt (e218, hashcat 1760)
 * hmac_sha512_kpass_batch: key=$pass (e797, hashcat 1750)
 *
 * SHA512 block size = 128 bytes, output = 64 bytes.
 * HMAC = SHA512((K⊕opad) || SHA512((K⊕ipad) || M))
 *
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 * Compact probe: first 16 bytes of 64-byte hash
 *
 * Primitives (bswap64, K512, rotr64, sha512_block, S512_copy_bytes,
 * S512_set_byte) provided by gpu_common.cl
 */

#define SHA512_IV0 0x6a09e667f3bcc908UL
#define SHA512_IV1 0xbb67ae8584caa73bUL
#define SHA512_IV2 0x3c6ef372fe94f82bUL
#define SHA512_IV3 0xa54ff53a5f1d36f1UL
#define SHA512_IV4 0x510e527fade682d1UL
#define SHA512_IV5 0x9b05688c2b3e6c1fUL
#define SHA512_IV6 0x1f83d9abfb41bd6bUL
#define SHA512_IV7 0x5be0cd19137e2179UL

/* HMAC-SHA512 key=$salt: key=salt, message=password */
__kernel void hmac_sha512_ksalt_batch(
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

    /* key_block: 128 bytes as LE for XOR, then bswap to BE for SHA512 */
    ulong key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 128) {
        /* key = SHA512(original_key) — rare for salts */
        ulong kst[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
        ulong M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S512_copy_bytes(M, 0, key, (klen < 128) ? klen : 128);
        if (klen <= 111) { S512_set_byte(M, klen, 0x80); M[15] = (ulong)klen * 8; }
        sha512_block(kst, M);
        if (klen > 111) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 128;
            if (rem > 0) S512_copy_bytes(M, 0, key + 128, rem);
            S512_set_byte(M, rem, 0x80);
            M[15] = (ulong)klen * 8;
            sha512_block(kst, M);
        }
        for (int i = 0; i < 8; i++) key_block[i] = bswap64(kst[i]);
        klen = 64;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 3;
            int bi = (i & 7) << 3;
            key_block[wi] |= ((ulong)key[i]) << bi;
        }
    }

    /* Inner hash: SHA512((key ⊕ ipad) || message) */
    ulong ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x3636363636363636UL;

    /* bswap ipad to BE for SHA512 */
    for (int i = 0; i < 16; i++) M[i] = bswap64(ipad[i]);

    ulong istate[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    sha512_block(istate, M);  /* 128-byte ipad block */

    /* Continue with message (password) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    if (plen <= 111) {
        S512_copy_bytes(M, 0, pass, plen);
        S512_set_byte(M, plen, 0x80);
        M[15] = (ulong)(128 + plen) * 8;
        sha512_block(istate, M);
    } else {
        S512_copy_bytes(M, 0, pass, (plen < 128) ? plen : 128);
        if (plen < 128) S512_set_byte(M, plen, 0x80);
        sha512_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = plen - 128;
        if (rem > 0) S512_copy_bytes(M, 0, pass + 128, rem);
        if (plen >= 128) S512_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (ulong)(128 + plen) * 8;
        sha512_block(istate, M);
    }

    /* Outer hash: SHA512((key ⊕ opad) || inner_hash) */
    ulong opad[16];
    for (int i = 0; i < 16; i++)
        opad[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(opad[i]);

    ulong ostate[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    sha512_block(ostate, M);  /* 128-byte opad block */

    /* inner hash (64 bytes BE) + padding.
     * Fits in one 128-byte block: 64 bytes data + 0x80 + zeros + 128-bit length. */
    for (int i = 0; i < 8; i++) M[i] = istate[i];  /* already BE */
    M[8] = 0x8000000000000000UL;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = (ulong)(128 + 64) * 8;  /* 192 bytes total */
    sha512_block(ostate, M);

    /* HMAC-SHA512: 16 hash words (8 x uint64 -> 16 x uint32) */
    uint h[16];
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(ostate[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 19;  /* 3 + 16 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 16; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* HMAC-SHA512 key=$pass: key=password, message=salt */
__kernel void hmac_sha512_kpass_batch(
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

    ulong key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 128) {
        ulong kst[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
        ulong M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        S512_copy_bytes(M, 0, key, (klen < 128) ? klen : 128);
        if (klen <= 111) { S512_set_byte(M, klen, 0x80); M[15] = (ulong)klen * 8; }
        sha512_block(kst, M);
        if (klen > 111) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 128;
            if (rem > 0) S512_copy_bytes(M, 0, key + 128, rem);
            S512_set_byte(M, rem, 0x80);
            M[15] = (ulong)klen * 8;
            sha512_block(kst, M);
        }
        for (int i = 0; i < 8; i++) key_block[i] = bswap64(kst[i]);
        klen = 64;
    } else {
        for (int i = 0; i < klen; i++) {
            int wi = i >> 3;
            int bi = (i & 7) << 3;
            key_block[wi] |= ((ulong)key[i]) << bi;
        }
    }

    ulong ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x3636363636363636UL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(ipad[i]);

    ulong istate[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    sha512_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 111) {
        S512_copy_bytes(M, 0, msg, mlen);
        S512_set_byte(M, mlen, 0x80);
        M[15] = (ulong)(128 + mlen) * 8;
        sha512_block(istate, M);
    } else {
        S512_copy_bytes(M, 0, msg, (mlen < 128) ? mlen : 128);
        if (mlen < 128) S512_set_byte(M, mlen, 0x80);
        sha512_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 128;
        if (rem > 0) S512_copy_bytes(M, 0, msg + 128, rem);
        if (mlen >= 128) S512_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (ulong)(128 + mlen) * 8;
        sha512_block(istate, M);
    }

    ulong opad[16];
    for (int i = 0; i < 16; i++)
        opad[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(opad[i]);

    ulong ostate[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                         SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    sha512_block(ostate, M);

    for (int i = 0; i < 8; i++) M[i] = istate[i];
    M[8] = 0x8000000000000000UL;
    for (int i = 9; i < 15; i++) M[i] = 0;
    M[15] = (ulong)(128 + 64) * 8;
    sha512_block(ostate, M);

    /* HMAC-SHA512: 16 hash words */
    uint h[16];
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(ostate[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 19;  /* 3 + 16 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 16; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* HMAC-SHA384: same compress as SHA512, SHA384 IV, inner hash = 48 bytes */

#define SHA384_IV0 0xcbbb9d5dc1059ed8UL
#define SHA384_IV1 0x629a292a367cd507UL
#define SHA384_IV2 0x9159015a3070dd17UL
#define SHA384_IV3 0x152fecd8f70e5939UL
#define SHA384_IV4 0x67332667ffc00b31UL
#define SHA384_IV5 0x8eb44a8768581511UL
#define SHA384_IV6 0xdb0c2e0d64f98fa7UL
#define SHA384_IV7 0x47b5481dbefa4fa4UL

__kernel void hmac_sha384_ksalt_batch(
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

    ulong key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;
    if (klen > 128) {
        ulong kst[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
        ulong M[16]; for (int i = 0; i < 16; i++) M[i] = 0;
        S512_copy_bytes(M, 0, key, (klen < 128) ? klen : 128);
        if (klen <= 111) { S512_set_byte(M, klen, 0x80); M[15] = (ulong)klen * 8; }
        sha512_block(kst, M);
        if (klen > 111) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            S512_copy_bytes(M, 0, key + 128, klen - 128);
            S512_set_byte(M, klen - 128, 0x80);
            M[15] = (ulong)klen * 8;
            sha512_block(kst, M);
        }
        for (int i = 0; i < 6; i++) key_block[i] = bswap64(kst[i]);
        klen = 48;
    } else {
        for (int i = 0; i < klen; i++)
            key_block[i >> 3] |= ((ulong)key[i]) << ((i & 7) << 3);
    }

    ulong ipad[16], M[16];
    for (int i = 0; i < 16; i++) ipad[i] = key_block[i] ^ 0x3636363636363636UL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(ipad[i]);
    ulong istate[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
    sha512_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (plen <= 111) {
        S512_copy_bytes(M, 0, pass, plen);
        S512_set_byte(M, plen, 0x80);
        M[15] = (ulong)(128 + plen) * 8;
        sha512_block(istate, M);
    } else {
        S512_copy_bytes(M, 0, pass, (plen < 128) ? plen : 128);
        if (plen < 128) S512_set_byte(M, plen, 0x80);
        sha512_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = plen - 128;
        if (rem > 0) S512_copy_bytes(M, 0, pass + 128, rem);
        if (plen >= 128) S512_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (ulong)(128 + plen) * 8;
        sha512_block(istate, M);
    }

    /* Outer: SHA384(opad || 48-byte inner_hash) */
    ulong opad[16];
    for (int i = 0; i < 16; i++) opad[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(opad[i]);
    ulong ostate[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
    sha512_block(ostate, M);

    /* 48 bytes (6 ulong) inner hash + padding */
    for (int i = 0; i < 6; i++) M[i] = istate[i];
    M[6] = 0x8000000000000000UL;  /* 0x80 at byte 48 */
    for (int i = 7; i < 15; i++) M[i] = 0;
    M[15] = (ulong)(128 + 48) * 8;  /* 1408 */
    sha512_block(ostate, M);

    /* HMAC-SHA384: 12 hash words (6 x uint64 -> 12 x uint32) */
    uint h[12];
    for (int i = 0; i < 6; i++) {
        ulong s = bswap64(ostate[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 15;  /* 3 + 12 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 12; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

__kernel void hmac_sha384_kpass_batch(
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

    ulong key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;
    if (klen > 128) {
        ulong kst[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
        ulong M[16]; for (int i = 0; i < 16; i++) M[i] = 0;
        S512_copy_bytes(M, 0, key, (klen < 128) ? klen : 128);
        if (klen <= 111) { S512_set_byte(M, klen, 0x80); M[15] = (ulong)klen * 8; }
        sha512_block(kst, M);
        if (klen > 111) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            S512_copy_bytes(M, 0, key + 128, klen - 128);
            S512_set_byte(M, klen - 128, 0x80);
            M[15] = (ulong)klen * 8;
            sha512_block(kst, M);
        }
        for (int i = 0; i < 6; i++) key_block[i] = bswap64(kst[i]);
        klen = 48;
    } else {
        for (int i = 0; i < klen; i++)
            key_block[i >> 3] |= ((ulong)key[i]) << ((i & 7) << 3);
    }

    ulong ipad[16], M[16];
    for (int i = 0; i < 16; i++) ipad[i] = key_block[i] ^ 0x3636363636363636UL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(ipad[i]);
    ulong istate[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
    sha512_block(istate, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    if (mlen <= 111) {
        S512_copy_bytes(M, 0, msg, mlen);
        S512_set_byte(M, mlen, 0x80);
        M[15] = (ulong)(128 + mlen) * 8;
        sha512_block(istate, M);
    } else {
        S512_copy_bytes(M, 0, msg, (mlen < 128) ? mlen : 128);
        if (mlen < 128) S512_set_byte(M, mlen, 0x80);
        sha512_block(istate, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int rem = mlen - 128;
        if (rem > 0) S512_copy_bytes(M, 0, msg + 128, rem);
        if (mlen >= 128) S512_set_byte(M, (rem > 0) ? rem : 0, 0x80);
        M[15] = (ulong)(128 + mlen) * 8;
        sha512_block(istate, M);
    }

    ulong opad[16];
    for (int i = 0; i < 16; i++) opad[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
    for (int i = 0; i < 16; i++) M[i] = bswap64(opad[i]);
    ulong ostate[8] = { SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,
                         SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7 };
    sha512_block(ostate, M);

    for (int i = 0; i < 6; i++) M[i] = istate[i];
    M[6] = 0x8000000000000000UL;
    for (int i = 7; i < 15; i++) M[i] = 0;
    M[15] = (ulong)(128 + 48) * 8;
    sha512_block(ostate, M);

    /* HMAC-SHA384: 12 hash words */
    uint h[12];
    for (int i = 0; i < 6; i++) {
        ulong s = bswap64(ostate[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 15;  /* 3 + 12 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 12; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- SHA512(pass + salt) kernel (e386, hashcat 1710) ---- */
__kernel void sha512passsalt_batch(
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

    ulong state[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                        SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    { ulong M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 111) {
        S512_copy_bytes(M, 0, pass, plen);
        S512_copy_bytes(M, plen, salts + soff, slen);
        S512_set_byte(M, total_len, 0x80);
        M[15] = (ulong)total_len * 8;
        sha512_block(state, M);
      } else {
        int pass_b1 = (plen < 128) ? plen : 128;
        S512_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 128 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0)
            S512_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        if (total_len < 128)
            S512_set_byte(M, total_len, 0x80);
        sha512_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S512_copy_bytes(M, 0, pass + pass_b1, pass_b2); pos2 = pass_b2; }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S512_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2); pos2 += salt_b2; }
        if (total_len >= 128)
            S512_set_byte(M, pos2, 0x80);
        M[15] = (ulong)total_len * 8;
        sha512_block(state, M);
      }
    }

    /* SHA512PASSSALT: 16 hash words */
    uint h[16];
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(state[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 19;  /* 3 + 16 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 16; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

/* ---- SHA512(salt + pass) kernel (e388, hashcat 1720) ---- */
__kernel void sha512saltpass_batch(
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

    ulong state[8] = { SHA512_IV0, SHA512_IV1, SHA512_IV2, SHA512_IV3,
                        SHA512_IV4, SHA512_IV5, SHA512_IV6, SHA512_IV7 };
    { ulong M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;

      if (total_len <= 111) {
        S512_copy_bytes(M, 0, salts + soff, slen);
        S512_copy_bytes(M, slen, pass, plen);
        S512_set_byte(M, total_len, 0x80);
        M[15] = (ulong)total_len * 8;
        sha512_block(state, M);
      } else {
        int salt_b1 = (slen < 128) ? slen : 128;
        S512_copy_bytes(M, 0, salts + soff, salt_b1);
        int pass_b1 = 128 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0)
            S512_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 128)
            S512_set_byte(M, total_len, 0x80);
        sha512_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S512_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2); pos2 = salt_b2; }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S512_copy_bytes(M, pos2, pass + pass_b1, pass_b2); pos2 += pass_b2; }
        if (total_len >= 128)
            S512_set_byte(M, pos2, 0x80);
        M[15] = (ulong)total_len * 8;
        sha512_block(state, M);
      }
    }

    /* SHA512SALTPASS: 16 hash words */
    uint h[16];
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(state[i]);
        h[i*2]   = (uint)s;
        h[i*2+1] = (uint)(s >> 32);
    }

    if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 19;  /* 3 + 16 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 16; i++) hits[base+3+i] = h[i];
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
