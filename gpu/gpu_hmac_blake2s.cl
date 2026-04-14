/* gpu_hmac_blake2s.cl — HMAC-BLAKE2S GPU kernel (e828, hashcat 33300)
 * BLAKE2s: 64-byte block, 32-byte output, little-endian, no tables.
 * HMAC key=$pass: BLAKE2S((K⊕opad) || BLAKE2S((K⊕ipad) || salt))
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 *
 * Primitives (B2S_IV, B2S_SIGMA, b2s_compress) provided by gpu_common.cl
 */

/* Full blake2s hash (no key, outlen=32) on a buffer up to 128 bytes */
void blake2s_hash(uint *out, const uchar *data, int datalen) {
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
    h[0] ^= 0x01010020u;  /* outlen=32, keylen=0, fanout=1, depth=1 */

    uchar buf[64];
    ulong counter = 0;
    int pos = 0;
    const uchar *p = data;
    int remaining = datalen;

    while (remaining > 0) {
        if (pos == 64) {
            counter += 64;
            b2s_compress(h, buf, counter, 0);
            pos = 0;
        }
        int take = 64 - pos;
        if (take > remaining) take = remaining;
        for (int i = 0; i < take; i++) buf[pos + i] = p[i];
        pos += take; p += take; remaining -= take;
    }
    counter += pos;
    for (int i = pos; i < 64; i++) buf[i] = 0;
    b2s_compress(h, buf, counter, 1);
    for (int i = 0; i < 8; i++) out[i] = h[i];
}

__kernel void hmac_blake2s_kpass_batch(
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

    /* Prepare key pad (64 bytes) */
    uchar kpad[64];
    if (klen > 64) {
        uint kh[8];
        uchar kb[256];
        for (int i = 0; i < klen; i++) kb[i] = key[i];
        blake2s_hash(kh, kb, klen);
        for (int i = 0; i < 32; i++) kpad[i] = ((uchar *)kh)[i];
        for (int i = 32; i < 64; i++) kpad[i] = 0;
    } else {
        for (int i = 0; i < klen; i++) kpad[i] = key[i];
        for (int i = klen; i < 64; i++) kpad[i] = 0;
    }

    /* Inner: BLAKE2S((K ^ ipad) || salt) */
    uchar ibuf[192];  /* max 64 + 128 bytes */
    for (int i = 0; i < 64; i++) ibuf[i] = kpad[i] ^ 0x36;
    for (int i = 0; i < mlen; i++) ibuf[64 + i] = msg[i];
    uint inner[8];
    blake2s_hash(inner, ibuf, 64 + mlen);

    /* Outer: BLAKE2S((K ^ opad) || inner_hash) */
    uchar obuf[96];
    for (int i = 0; i < 64; i++) obuf[i] = kpad[i] ^ 0x5c;
    for (int i = 0; i < 32; i++) obuf[64 + i] = ((uchar *)inner)[i];
    uint outer[8];
    blake2s_hash(outer, obuf, 96);

    uint hx = outer[0], hy = outer[1], hz = outer[2], hw = outer[3];

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 7;
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            hits[base+3] = hx; hits[base+4] = hy;
            hits[base+5] = hz; hits[base+6] = hw;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
