/* gpu_kernels.cl — MD5SALT / MD5SALTPASS / SHA1SALT / SHA256SALT etc.
 *
 * All primitives (OCLParams, K[], FF/GG/HH/II, md5_block, sha1_block,
 * sha256_block, bswap32, S_copy_bytes, S_set_byte, M_copy_bytes,
 * M_set_byte, compact_mix, probe_compact, hex_byte_lc/uc,
 * md5_to_hex_lc/uc) provided by gpu_common.cl
 */

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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
    }
}

/* hex_byte_lc, md5_to_hex_lc provided by gpu_common.cl */

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
            EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, iter + 1, hx, hy, hz, hw)
        }
    }
}

/* hex_byte_lc, hex_byte_uc, md5_to_hex_lc, md5_to_hex_uc provided by gpu_common.cl */

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
            EMIT_HIT_4(hits, hit_count, params.max_hits, tid, 0, iter + 1, hx, hy, hz, hw)
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
            EMIT_HIT_4(hits, hit_count, params.max_hits, tid, 0, iter + 1, hx, hy, hz, hw)
        }

        if (iter < params.max_iter) {
            md5_to_hex_uc(hx, hy, hz, hw, M);
        }
    }
}

/* M_copy_bytes, M_set_byte provided by gpu_common.cl */

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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, hx, hy, hz, hw)
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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, hx, hy, hz, hw)
    }
}

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

/* ---- PHPBB3/phpass kernel (e455): MD5 iterated 2^N times ---- */
/* Salt buffer has "$H$Csalt8chr" (12 bytes). Iteration count from salt[3].
 * Algorithm: MD5(salt + pass), then N iterations of MD5(hash_binary + pass).
 * Password max 39 bytes (single MD5 block for 16 + 39 = 55). */
__constant uchar phpitoa64_k[] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

__kernel void phpbb3_batch(
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
    __global const uchar *salt_data = salts + soff;

    /* Decode iteration count from salt[3] */
    uchar ic = salt_data[3];
    int log2count = 0;
    for (int k = 0; k < 64; k++) { if (phpitoa64_k[k] == ic) { log2count = k; break; } }
    uint count = 1u << log2count;

    /* Step 1: MD5(salt[4..11] + password) — salt is 8 bytes */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;
    M_copy_bytes(M, 0, salt_data + 4, 8);
    M_copy_bytes(M, 8, pass, plen);
    int total = 8 + plen;
    M_set_byte(M, total, 0x80);
    M[14] = total * 8;
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Step 2: N iterations of MD5(hash_binary[16] + password[plen]) */
    /* Pre-pack M[] with password, padding, length — constant across iterations */
    total = 16 + plen;
    for (int i = 0; i < 16; i++) M[i] = 0;
    M_copy_bytes(M, 16, pass, plen);
    M_set_byte(M, total, 0x80);
    M[14] = total * 8;
    /* Iteration loop: only M[0..3] change (the hash digest) */
    for (uint ic2 = 0; ic2 < count; ic2++) {
        M[0] = hx; M[1] = hy; M[2] = hz; M[3] = hw;
        hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
    }
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

/* ---- MD5 with mask expansion (GPU_CAT_MASK) ---- */
/* Mask descriptor buffer: [n_prepend charset IDs] [n_append charset IDs]
 * Charset IDs: 0=?d(10) 1=?l(26) 2=?u(26) 3=?s(33) 4=?a(95) 5=?b(256)
 * Each work item: tid / num_masks = word_idx, tid % num_masks = mask_idx
 * mask_idx is decomposed: prepend_idx × append_combos + append_idx */

/* cs_a and cs_b built dynamically to save constant memory */

__kernel void md5_mask_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *mask_desc, __global const uint *unused1, __global const ushort *unused2,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    int baselen = hexlens[word_idx];
    __global const uchar *base = hexhashes + word_idx * 256;

    /* Build candidate: [prepend chars] [base word] [append chars] */
    uchar cand[128];
    int clen = 0;

    /* Decode prepend: mask_idx split into prepend_idx and append_idx */
    uint n_total = params.n_prepend + params.n_append;
    uint append_combos = 1;
    for (uint i = 0; i < params.n_append; i++)
        append_combos *= mask_desc[params.n_prepend + i];

    uint prepend_idx = (uint)(mask_idx / append_combos);
    uint append_idx = (uint)(mask_idx % append_combos);

    /* Generate prepend chars (right to left in the index, left to right in output) */
    if (params.n_prepend > 0) {
        uint pidx = prepend_idx;
        for (int i = params.n_prepend - 1; i >= 0; i--) {
            uint sz = mask_desc[i];
            cand[i] = mask_desc[n_total + i * 256 + (pidx % sz)];
            pidx /= sz;
        }
        clen = params.n_prepend;
    }

    /* Copy base word */
    for (int i = 0; i < baselen; i++)
        cand[clen + i] = base[i];
    clen += baselen;

    /* Generate append chars */
    if (params.n_prepend > 0) {
        if (params.n_append > 0) {
            uint aidx = append_idx;
            for (int i = params.n_append - 1; i >= 0; i--) {
                int pos_idx = params.n_prepend + i;
                uint sz = mask_desc[pos_idx];
                cand[clen + i] = mask_desc[n_total + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
            }
            clen += params.n_append;
        }
    } else if (params.n_append > 0) {
        /* Append-only (brute-force): host pre-decomposes mask_start into
         * per-position base offsets. Kernel does fast uint32 local
         * decomposition and adds to base with carry. */
        uint local_idx = tid % params.num_masks;
        uint aidx = local_idx;
        uint carry = 0;
        for (int i = params.n_append - 1; i >= 0; i--) {
            int pos_idx = params.n_prepend + i;
            uint sz = mask_desc[pos_idx];
            uint local_digit = aidx % sz;
            aidx /= sz;
            uint base_digit = (i < 8)
                ? (uint)((params.mask_base0 >> (i * 8)) & 0xFF)
                : (uint)((params.mask_base1 >> ((i - 8) * 8)) & 0xFF);
            uint sum = base_digit + local_digit + carry;
            carry = sum / sz;
            uint final_digit = sum % sz;
            cand[clen + i] = mask_desc[n_total + pos_idx * 256 + final_digit];
        }
        clen += params.n_append;
    }

    /* MD5(candidate) */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    if (clen <= 55) {
        /* Single block */
        for (int i = 0; i < clen; i++)
            M[i >> 2] |= ((uint)cand[i]) << ((i & 3) << 3);
        M[clen >> 2] |= 0x80u << ((clen & 3) << 3);
        M[14] = clen * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
    } else {
        /* Two blocks */
        for (int i = 0; i < 64 && i < clen; i++)
            M[i >> 2] |= ((uint)cand[i]) << ((i & 3) << 3);
        md5_block(&hx, &hy, &hz, &hw, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        for (int i = 64; i < clen; i++) {
            int j = i - 64;
            M[j >> 2] |= ((uint)cand[i]) << ((j & 3) << 3);
        }
        int rem = clen - 64;
        M[rem >> 2] |= 0x80u << ((rem & 3) << 3);
        M[14] = clen * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, mask_idx, 1u, hx, hy, hz, hw)
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
    { __global const uint *sw = (__global const uint *)(salts + salt_offsets[salt_idx]);
      for (int i = 0; i < 8; i++) M[i] = sw[i];
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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
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
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1u, hx, hy, hz, hw)
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
