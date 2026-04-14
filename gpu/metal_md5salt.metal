kernel void md5salt_probe(
    device const uint8_t    *words        [[buffer(0)]],
    device const uint       *word_offsets [[buffer(1)]],
    device const ushort     *word_lens    [[buffer(2)]],
    device const uint8_t    *salts        [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens    [[buffer(5)]],
    device const uint       *compact_fp   [[buffer(6)]],
    device const uint       *compact_idx  [[buffer(7)]],
    constant MetalParams    &params       [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits          [[buffer(12)]],
    device atomic_uint      *hit_count     [[buffer(13)]],
    uint                     tid           [[thread_position_in_grid]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    /* Step 1: Read word */
    uint woff = word_offsets[word_idx];
    int wlen = word_lens[word_idx];
    uint8_t word[256];
    for (int i = 0; i < wlen && i < 255; i++)
        word[i] = words[woff + i];

    /* Step 2: MD5(word) */
    uint4 h1;
    if (wlen <= 55)
        md5_short(word, wlen, h1);
    else
        md5_two(word, wlen, h1);

    /* Step 3: Hex-encode to 32 chars */
    uint8_t hexbuf[128];
    thread uint8_t *bin = (thread uint8_t *)&h1;
    hex_encode(bin, hexbuf);

    /* Step 4: Append salt */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    for (int i = 0; i < slen && i < 90; i++)
        hexbuf[32 + i] = salts[soff + i];
    int total_len = 32 + slen;

    /* Step 5: MD5(hex + salt) */
    uint4 h2;
    if (total_len <= 55)
        md5_short(hexbuf, total_len, h2);
    else
        md5_two(hexbuf, total_len, h2);

    /* Step 6: Compact table probe */
    thread uint8_t *hashbytes = (thread uint8_t *)&h2;
    uint64_t key = *(thread const uint64_t *)hashbytes;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) return;  /* empty slot = miss */
        if (cfp == fp) {
            /* fingerprint match — compare full hash */
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                int hlen = hash_data_len[idx];
                if (hlen > 16) hlen = 16;
                uint64_t off = hash_data_off[idx];
                bool match = true;
                for (int i = 0; i < hlen; i++) {
                    if (hashbytes[i] != hash_data_buf[off + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < 65536) {
                        hits[slot * 2]     = word_idx;
                        hits[slot * 2 + 1] = salt_idx;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Max probe exhausted — possible overflow. Report as hit for CPU validation. */
    {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 5;
            hits[base]     = tid;
            hits[base + 1] = h2.x;
            hits[base + 2] = h2.y;
            hits[base + 3] = h2.z;
            hits[base + 4] = h2.w;
        }
    }
}

/* Second kernel: pre-hashed input. "words" buffer contains the 32-char hex hash
 * already computed by CPU. GPU just appends each salt and does one MD5 + probe.
 * num_words is always 1, num_salts GPU threads. */
kernel void md5salt_salts_only(
    device const uint8_t    *hexhash      [[buffer(0)]],
    device const uint       *word_offsets [[buffer(1)]],
    device const ushort     *word_lens    [[buffer(2)]],
    device const uint8_t    *salts        [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens    [[buffer(5)]],
    device const uint       *compact_fp   [[buffer(6)]],
    device const uint       *compact_idx  [[buffer(7)]],
    constant MetalParams    &params       [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits          [[buffer(12)]],
    device atomic_uint      *hit_count     [[buffer(13)]],
    uint                     tid           [[thread_position_in_grid]])
{
    if (tid >= params.num_salts) return;

    /* Copy pre-computed 32-char hex hash */
    uint8_t buf[128];
    int hlen = word_lens[0];
    for (int i = 0; i < hlen && i < 55; i++)
        buf[i] = hexhash[i];

    /* Append salt */
    uint soff = salt_offsets[tid];
    int slen = salt_lens[tid];
    for (int i = 0; i < slen && i < 90; i++)
        buf[hlen + i] = salts[soff + i];
    int total_len = hlen + slen;

    /* MD5(hex + salt) */
    uint4 h2;
    if (total_len <= 55)
        md5_short(buf, total_len, h2);
    else
        md5_two(buf, total_len, h2);

    /* Compact table probe */
    thread uint8_t *hashbytes = (thread uint8_t *)&h2;
    uint64_t key = *(thread const uint64_t *)hashbytes;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) return;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                int dlen = hash_data_len[idx];
                if (dlen > 16) dlen = 16;
                uint64_t off = hash_data_off[idx];
                bool match = true;
                for (int i = 0; i < dlen; i++) {
                    if (hashbytes[i] != hash_data_buf[off + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 5;
                        hits[base]     = tid;
                        hits[base + 1] = h2.x;
                        hits[base + 2] = h2.y;
                        hits[base + 3] = h2.z;
                        hits[base + 4] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Max probe exhausted — possible overflow. Report as hit for CPU validation. */
    {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 5;
            hits[base]     = tid;
            hits[base + 1] = h2.x;
            hits[base + 2] = h2.y;
            hits[base + 3] = h2.z;
            hits[base + 4] = h2.w;
        }
    }
}

/* Third kernel: batch of pre-hashed words × all salts.
 * Grid: num_words * num_salts threads.
 * hexhashes buffer: 256 bytes per word, packed contiguously. */
kernel void md5salt_batch_prehashed(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
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

    /* Shared prestate: thread 0 computes MD5 rounds 0-7 for its word.
     * Other threads use it ONLY if they share the same word_idx.
     * At threadgroup boundaries (num_salts % tgsize != 0), some threads
     * may have a different word_idx and must compute their own. */
    threadgroup uint shared_M[8];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 8; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Load M[0..7] — use shared if same word, else load own */
    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 8; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            M[i] = mwords[i];
    }

    /* Append salt into M[8..] at byte position 32 */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;

    uint4 h2;
    if (slen <= 23) {
        /* Single block: fill M[8..15] with salt + padding + bitlen. */
        for (int i = 8; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[32 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        if (use_shared) {
            h2 = shared_prestate;
            md5_block_from8(h2, M);
        } else {
            /* Full md5_block for boundary threads */
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block_full(h2, M);
        }
    } else {
        /* Two-block: fall back to full computation */
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 8; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 90; i++)
            buf[32 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Compact table probe — use wide compares */
    uint64_t key = ((thread const uint64_t *)&h2)[0];
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                uint64_t off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h2.x == ref[0] && h2.y == ref[1] &&
                    h2.z == ref[2] && h2.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Binary search the overflow table */
    if (params.overflow_count == 0) return;
    {
        int lo = 0, hi = (int)params.overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            uint64_t mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                /* Key match — compare full hash as 4 uint32 */
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                bool match = (h2.x == oref[0] && h2.y == oref[1] &&
                              h2.z == oref[2] && h2.w == oref[3]);
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
                /* Key matched but hash didn't — search duplicates both directions */
                for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                return;
            }
        }
    }
}

/* Iterating batch kernel: MD5(hex+salt), then MD5(hex(result)) for each iteration.
 * Identical setup to md5salt_batch_prehashed, but probes at each iteration. */
kernel void md5salt_batch_iter(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
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

    threadgroup uint shared_M[8];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 8; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 8; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            M[i] = mwords[i];
    }

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;

    uint4 h2;
    if (slen <= 23) {
        for (int i = 8; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[32 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        if (use_shared) {
            h2 = shared_prestate;
            md5_block_from8(h2, M);
        } else {
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block_full(h2, M);
        }
    } else {
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 8; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 90; i++)
            buf[32 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Iterate: probe at each iteration, then MD5(hex(h2)) for next */
    for (uint iter = 0; iter < params.max_iter; iter++) {
        /* If not first iteration, compute MD5(hex(h2)) */
        if (iter > 0) {
            uint Mi[16];
            hash_to_hex_M(h2, Mi);
            Mi[8] = 0x80;
            Mi[9] = 0; Mi[10] = 0; Mi[11] = 0;
            Mi[12] = 0; Mi[13] = 0;
            Mi[14] = 256; /* 32 * 8 bits */
            Mi[15] = 0;
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block_full(h2, Mi);
        }

        /* Compact table probe — record hit but continue iterating */
        uint64_t key = ((thread const uint64_t *)&h2)[0];
        uint fp = (uint)(key >> 32);
        if (fp == 0) fp = 1;
        uint64_t pos = compact_mix(key) & params.compact_mask;
        bool found = false;

        for (int p = 0; p < (int)params.max_probe; p++) {
            uint cfp = compact_fp[pos];
            if (cfp == 0) break;
            if (cfp == fp) {
                uint idx = compact_idx[pos];
                if (idx < params.hash_data_count) {
                    uint64_t off = hash_data_off[idx];
                    device const uint *ref = (device const uint *)(hash_data_buf + off);
                    if (h2.x == ref[0] && h2.y == ref[1] &&
                        h2.z == ref[2] && h2.w == ref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 7;
                            hits[base]     = word_idx;
                            hits[base + 1] = salt_idx;
                            hits[base + 2] = iter + 1;
                            hits[base + 3] = h2.x;
                            hits[base + 4] = h2.y;
                            hits[base + 5] = h2.z;
                            hits[base + 6] = h2.w;
                        }
                        found = true;
                        break;
                    }
                }
            }
            pos = (pos + 1) & params.compact_mask;
        }

        /* Binary search overflow */
        if (!found && params.overflow_count > 0) {
            int lo = 0, hi = (int)params.overflow_count - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                uint64_t mkey = overflow_keys[mid];
                if (key < mkey) hi = mid - 1;
                else if (key > mkey) lo = mid + 1;
                else {
                    uint ooff = overflow_offsets[mid];
                    device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 7;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = iter + 1;
                            hits[base+3] = h2.x; hits[base+4] = h2.y;
                            hits[base+5] = h2.z; hits[base+6] = h2.w;
                        }
                        break;
                    }
                    for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                        oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h2.x == oref[0] && h2.y == oref[1] &&
                            h2.z == oref[2] && h2.w == oref[3]) {
                            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                            if (slot < params.max_hits) {
                                uint base = slot * 7;
                                hits[base] = word_idx; hits[base+1] = salt_idx;
                                hits[base+2] = iter + 1;
                                hits[base+3] = h2.x; hits[base+4] = h2.y;
                                hits[base+5] = h2.z; hits[base+6] = h2.w;
                            }
                            found = true; break;
                        }
                    }
                    if (!found) {
                        for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                            oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                            if (h2.x == oref[0] && h2.y == oref[1] &&
                                h2.z == oref[2] && h2.w == oref[3]) {
                                uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                                if (slot < params.max_hits) {
                                    uint base = slot * 7;
                                    hits[base] = word_idx; hits[base+1] = salt_idx;
                                    hits[base+2] = iter + 1;
                                    hits[base+3] = h2.x; hits[base+4] = h2.y;
                                    hits[base+5] = h2.z; hits[base+6] = h2.w;
                                }
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    } /* end iteration loop */
}

/* Sub8-24 batch kernel: hexlen=16 (chars 8-23 of MD5 hex), 4-word prestate.
 * Identical buffer layout to batch_prehashed, but constant 16-byte hex input. */
kernel void md5salt_batch_sub8_24(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
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

    /* Shared prestate: 4 M words, 4 prestate rounds for hexlen=16 */
    threadgroup uint shared_M[4];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 4; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 4; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Load M[0..3] — use shared if same word, else load own */
    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 4; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 4; i++)
            M[i] = mwords[i];
    }

    /* Append salt at byte position 16 */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 16 + slen;

    uint4 h2;
    if (total_len <= 55) {
        /* Single block: fill M[4..15] with salt + padding + bitlen */
        for (int i = 4; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[16 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        /* Use full md5_block for now — prestate optimization can be added later */
        h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
        md5_block_full(h2, M);
    } else {
        /* Two-block: fall back to full computation */
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 4; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 106; i++)
            buf[16 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Compact table probe — use wide compares */
    uint64_t key = ((thread const uint64_t *)&h2)[0];
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                uint64_t off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h2.x == ref[0] && h2.y == ref[1] &&
                    h2.z == ref[2] && h2.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Binary search the overflow table */
    if (params.overflow_count == 0) return;
    {
        int lo = 0, hi = (int)params.overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            uint64_t mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                bool match = (h2.x == oref[0] && h2.y == oref[1] &&
                              h2.z == oref[2] && h2.w == oref[3]);
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
                for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                return;
            }
        }
    }
}

