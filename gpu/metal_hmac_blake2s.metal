/* metal_hmac_blake2s.metal — HMAC-BLAKE2S GPU kernel (e828, hashcat 33300)
 * BLAKE2s: 64-byte block, 32-byte output, little-endian, no tables.
 * HMAC key=$pass: BLAKE2S((K^opad) || BLAKE2S((K^ipad) || salt))
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 */

/* B2S_IV, B2S_SIGMA, b2s_compress all provided by metal_common.metal */

static void blake2s_hash(thread uint *out, thread const uchar *data, int datalen) {
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
    h[0] ^= 0x01010020u;

    uchar buf[64];
    ulong counter = 0;
    int pos = 0;
    thread const uchar *p = data;
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

kernel void hmac_blake2s_kpass_batch(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hexlens     [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *salts       [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens   [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int klen = hexlens[word_idx];
    device const uint8_t *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen = salt_lens[salt_idx];
    device const uint8_t *msg = salts + soff;

    uchar kpad[64];
    if (klen > 64) {
        uint kh[8];
        uchar kb[256];
        for (int i = 0; i < klen; i++) kb[i] = key[i];
        blake2s_hash(kh, kb, klen);
        for (int i = 0; i < 32; i++) kpad[i] = ((thread uchar *)kh)[i];
        for (int i = 32; i < 64; i++) kpad[i] = 0;
    } else {
        for (int i = 0; i < klen; i++) kpad[i] = key[i];
        for (int i = klen; i < 64; i++) kpad[i] = 0;
    }

    /* Inner: BLAKE2S((K ^ ipad) || salt) */
    uchar ibuf[192];
    for (int i = 0; i < 64; i++) ibuf[i] = kpad[i] ^ 0x36;
    for (int i = 0; i < mlen; i++) ibuf[64 + i] = msg[i];
    uint inner[8];
    blake2s_hash(inner, ibuf, 64 + mlen);

    /* Outer: BLAKE2S((K ^ opad) || inner_hash) */
    uchar obuf[96];
    for (int i = 0; i < 64; i++) obuf[i] = kpad[i] ^ 0x5c;
    for (int i = 0; i < 32; i++) obuf[64 + i] = ((thread uchar *)inner)[i];
    uint outer[8];
    blake2s_hash(outer, obuf, 96);

    uint hx = outer[0], hy = outer[1], hz = outer[2], hw = outer[3];

    uint4 h = uint4(hx, hy, hz, hw);
    ulong key2 = (ulong(h.y) << 32) | h.x;
    uint fp = uint(key2 >> 32);
    if (fp == 0) fp = 1;
    ulong pos = (key2 ^ (key2 >> 32)) & params.compact_mask;
    for (uint p = 0; p < params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 7;
                        hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
                        hits[base+3] = h.x; hits[base+4] = h.y;
                        hits[base+5] = h.z; hits[base+6] = h.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    if (params.overflow_count > 0) {
        int lo = 0, hi = int(params.overflow_count) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key2 < mkey) hi = mid - 1;
            else if (key2 > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 7;
                        hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
                        hits[base+3] = h.x; hits[base+4] = h.y;
                        hits[base+5] = h.z; hits[base+6] = h.w;
                    }
                    return;
                }
                break;
            }
        }
    }
}
