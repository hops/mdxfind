/* SHA256 K256[], sha256_block(), S_copy_bytes, S_set_byte, bswap32
 * all provided by metal_common.metal */

/* ---- SHA256(password + salt) kernel (e413) ---- */
kernel void sha256passsalt_batch(
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

    int plen = hex_lens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint state[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    if (total_len <= 55) {
        S_copy_bytes(M, 0, pass, plen);
        S_copy_bytes(M, plen, salts + soff, slen);
        S_set_byte(M, total_len, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
    } else {
        int pass_b1 = (plen < 64) ? plen : 64;
        S_copy_bytes(M, 0, pass, pass_b1);
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0) S_copy_bytes(M, pass_b1, salts + soff, salt_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha256_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, 0, pass + pass_b1, pass_b2); pos2 = pass_b2; }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, pos2, salts + soff + salt_b1, salt_b2); pos2 += salt_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
    }

    /* SHA256: 8 hash words */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

    /* Probe compact table with first 4 words */
    ulong key = (ulong(h[1]) << 32) | h[0];
    uint fp = uint(key >> 32); if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & params.compact_mask;
    bool found = false;
    for (uint p = 0; p < params.max_probe && !found; p++) {
        uint cfp = compact_fp[pos]; if (cfp == 0) break;
        if (cfp == fp) { uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h[0] == ref[0] && h[1] == ref[1] && h[2] == ref[2] && h[3] == ref[3])
                    found = true;
            } }
        pos = (pos + 1) & params.compact_mask;
    }
    if (!found && params.overflow_count > 0) {
        int lo = 0, hi2 = int(params.overflow_count) - 1;
        while (lo <= hi2 && !found) {
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid];
            if (key < mkey) hi2 = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                        found = true; }
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) {
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                        found = true; }
                break; } } }
    if (found) {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 11;  /* 3 + 8 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 8; i++) hits[base+3+i] = h[i]; } }
}

/* ---- SHA256(salt + password) kernel (e412) ---- */
kernel void sha256saltpass_batch(
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

    int plen = hex_lens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    uint state[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    uint M[16];
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
        if (pass_b1 > 0) S_copy_bytes(M, salt_b1, pass, pass_b1);
        if (total_len < 64) S_set_byte(M, total_len, 0x80);
        sha256_block(state, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) { S_copy_bytes(M, 0, salts + soff + salt_b1, salt_b2); pos2 = salt_b2; }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) { S_copy_bytes(M, pos2, pass + pass_b1, pass_b2); pos2 += pass_b2; }
        if (total_len >= 64) S_set_byte(M, pos2, 0x80);
        M[15] = total_len * 8;
        sha256_block(state, M);
    }

    /* SHA256: 8 hash words */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = bswap32(state[i]);

    ulong key = (ulong(h[1]) << 32) | h[0];
    uint fp = uint(key >> 32); if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & params.compact_mask;
    bool found = false;
    for (uint p = 0; p < params.max_probe && !found; p++) {
        uint cfp = compact_fp[pos]; if (cfp == 0) break;
        if (cfp == fp) { uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h[0] == ref[0] && h[1] == ref[1] && h[2] == ref[2] && h[3] == ref[3])
                    found = true;
            } }
        pos = (pos + 1) & params.compact_mask;
    }
    if (!found && params.overflow_count > 0) {
        int lo = 0, hi2 = int(params.overflow_count) - 1;
        while (lo <= hi2 && !found) {
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid];
            if (key < mkey) hi2 = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                        found = true; }
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) {
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                        found = true; }
                break; } } }
    if (found) {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 11;  /* 3 + 8 */
            hits[base] = word_idx; hits[base+1] = salt_idx; hits[base+2] = 1;
            for (int i = 0; i < 8; i++) hits[base+3+i] = h[i]; } }
}

