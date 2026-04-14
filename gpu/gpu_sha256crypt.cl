/* gpu_sha256crypt.cl -- SHA256CRYPT (e512, hashcat 7400) OpenCL kernel
 *
 * glibc crypt-sha256: $5$[rounds=N$]salt$hash
 * Default 5000 rounds, salt up to 16 chars, SHA256 (32-byte digest).
 *
 * Hit stride: 11 (word_idx, salt_idx, iter, h[0..7])
 * Compact probe: first 16 bytes of 32-byte hash as 4 x uint32
 */

/* SHA256 primitives (bswap32, sha256_block) provided by gpu_common.cl.
 * SC_IV aliases for sha256crypt init convenience. */

#define SC_IV0 0x6a09e667u
#define SC_IV1 0xbb67ae85u
#define SC_IV2 0x3c6ef372u
#define SC_IV3 0xa54ff53au
#define SC_IV4 0x510e527fu
#define SC_IV5 0x9b05688cu
#define SC_IV6 0x1f83d9abu
#define SC_IV7 0x5be0cd19u

/* ---- Buffered SHA256: init / update / final ---- */

void sc_sha256_init(uint *state) {
    state[0] = SC_IV0; state[1] = SC_IV1;
    state[2] = SC_IV2; state[3] = SC_IV3;
    state[4] = SC_IV4; state[5] = SC_IV5;
    state[6] = SC_IV6; state[7] = SC_IV7;
}

/* Store a byte into big-endian uint buffer at byte position pos */
void sc_buf_set_byte(uint *buf, int pos, uchar val) {
    int wi = pos >> 2;
    int bi = (3 - (pos & 3)) << 3;
    buf[wi] = (buf[wi] & ~(0xffu << bi)) | ((uint)val << bi);
}

/* Get a byte from big-endian uint buffer at byte position pos */
uchar sc_buf_get_byte(uint *buf, int pos) {
    int wi = pos >> 2;
    int bi = (3 - (pos & 3)) << 3;
    return (uchar)(buf[wi] >> bi);
}

/* Update: accumulate bytes into 64-byte buffer, compress when full.
 * buf is 16 uints (64 bytes), bufpos is current byte position,
 * counter tracks total bytes processed. Data from private memory. */
void sc_sha256_update(uint *state, uint *buf, int *bufpos,
                      uint *counter, const uchar *data, int len)
{
    *counter += (uint)len;
    int bp = *bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]);
        bp++;
        if (bp == 64) {
            sha256_block(state, buf);
            for (int j = 0; j < 16; j++) buf[j] = 0;
            bp = 0;
        }
    }
    *bufpos = bp;
}

/* Update from global memory */
void sc_sha256_update_g(uint *state, uint *buf, int *bufpos,
                        uint *counter, __global const uchar *data, int len)
{
    *counter += (uint)len;
    int bp = *bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]);
        bp++;
        if (bp == 64) {
            sha256_block(state, buf);
            for (int j = 0; j < 16; j++) buf[j] = 0;
            bp = 0;
        }
    }
    *bufpos = bp;
}

/* Final: pad and produce 32-byte digest as raw bytes (big-endian state words) */
void sc_sha256_final(uint *state, uint *buf, int bufpos,
                     uint counter, uchar *out)
{
    /* Append 0x80 */
    sc_buf_set_byte(buf, bufpos, 0x80);
    bufpos++;

    /* If not enough room for 8-byte length (need pos <= 56), flush */
    if (bufpos > 56) {
        for (int i = bufpos; i < 64; i++)
            sc_buf_set_byte(buf, i, 0);
        sha256_block(state, buf);
        for (int j = 0; j < 16; j++) buf[j] = 0;
        bufpos = 0;
    }

    /* Zero remaining bytes up to length field */
    for (int i = bufpos; i < 56; i++)
        sc_buf_set_byte(buf, i, 0);

    /* Append 64-bit length in bits (big-endian). High 32 bits = 0 for our sizes. */
    buf[14] = 0;
    buf[15] = counter * 8;

    sha256_block(state, buf);

    /* Extract 32 bytes from state (big-endian) */
    for (int i = 0; i < 8; i++) {
        uint w = state[i];
        out[i*4+0] = (uchar)(w >> 24);
        out[i*4+1] = (uchar)(w >> 16);
        out[i*4+2] = (uchar)(w >> 8);
        out[i*4+3] = (uchar)(w);
    }
}

/* Convenience: one-shot SHA256(data, len) -> out[32] from private memory */
void sc_sha256_oneshot(const uchar *data, int len, uchar *out) {
    uint state[8];
    uint buf[16];
    int bufpos = 0;
    uint counter = 0;
    sc_sha256_init(state);
    for (int j = 0; j < 16; j++) buf[j] = 0;
    sc_sha256_update(state, buf, &bufpos, &counter, data, len);
    sc_sha256_final(state, buf, bufpos, counter, out);
}

/* ---- SHA256CRYPT kernel ---- */

__kernel void sha256crypt_batch(
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

    /* Copy password to private memory */
    uchar pw[256];
    for (int i = 0; i < plen; i++) pw[i] = pass[i];

    /* Read salt string */
    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];
    __global const uchar *salt_str = salts + soff;

    /* Parse "$5$[rounds=N$]salt[$]" */
    int spos = 3; /* skip "$5$" */
    int rounds = 5000;

    /* Check for "rounds=" */
    if (slen_full > 10 && salt_str[3] == 'r' && salt_str[4] == 'o' &&
        salt_str[5] == 'u' && salt_str[6] == 'n' && salt_str[7] == 'd' &&
        salt_str[8] == 's' && salt_str[9] == '=') {
        /* Parse decimal number */
        rounds = 0;
        spos = 10;
        while (spos < slen_full && salt_str[spos] >= '0' && salt_str[spos] <= '9') {
            rounds = rounds * 10 + (salt_str[spos] - '0');
            spos++;
        }
        if (rounds < 1000) rounds = 1000;
        if (rounds > 999999999) rounds = 999999999;
        /* Skip the '$' after the number */
        if (spos < slen_full && salt_str[spos] == '$') spos++;
    }

    /* Extract raw salt (up to 16 chars, stop at '$' or end) */
    uchar raw_salt[16];
    int saltlen = 0;
    for (int i = spos; i < slen_full && saltlen < 16; i++) {
        if (salt_str[i] == '$') break;
        raw_salt[saltlen++] = salt_str[i];
    }
    if (saltlen == 0) return;

    /* Working buffers */
    uint state[8], ctx_buf[16];
    int ctx_bufpos;
    uint ctx_counter;
    uchar tmp[256]; /* scratch for building messages */
    uchar digest_a[32]; /* Hash A = alt_result in glibc */
    uchar digest_b[32]; /* temporary */

    /* ---- Step 1: Hash A = SHA256(pass + salt + pass) ---- */
    {
        int tlen = 0;
        for (int i = 0; i < plen; i++) tmp[tlen++] = pw[i];
        for (int i = 0; i < saltlen; i++) tmp[tlen++] = raw_salt[i];
        for (int i = 0; i < plen; i++) tmp[tlen++] = pw[i];
        sc_sha256_oneshot(tmp, tlen, digest_a);
    }

    /* ---- Step 2: Hash B ---- */
    /* init; update(pass + salt); for passlen bytes: update chunks of A; bit-loop; final -> curin */
    sc_sha256_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;

    /* update(pass + salt) */
    sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, pw, plen);
    sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, raw_salt, saltlen);

    /* For passlen bytes, update chunks of digest_a (32 bytes each) */
    for (int x = plen; x > 32; x -= 32)
        sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, digest_a, 32);
    /* Remaining bytes */
    {
        int x = plen;
        while (x > 32) x -= 32;
        sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, digest_a, x);
    }

    /* Bit loop: for each bit of passlen (LSB to MSB) */
    for (int x = plen; x != 0; x >>= 1) {
        if (x & 1)
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, digest_a, 32);
        else
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, pw, plen);
    }

    uchar curin[32];
    sc_sha256_final(state, ctx_buf, ctx_bufpos, ctx_counter, curin);

    /* ---- Step 3: Hash P = SHA256(pass repeated passlen times) ---- */
    sc_sha256_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;
    for (int x = 0; x < plen; x++)
        sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, pw, plen);
    sc_sha256_final(state, ctx_buf, ctx_bufpos, ctx_counter, digest_b);

    /* P-bytes: digest_b repeated to fill passlen bytes */
    uchar p_bytes[256];
    for (int i = 0; i < plen; i++)
        p_bytes[i] = digest_b[i % 32];

    /* ---- Step 4: Hash S = SHA256(salt repeated 16+curin[0] times) ---- */
    sc_sha256_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;
    int s_repeats = 16 + (uint)curin[0];
    for (int x = 0; x < s_repeats; x++)
        sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, raw_salt, saltlen);
    sc_sha256_final(state, ctx_buf, ctx_bufpos, ctx_counter, digest_b);

    /* S-bytes: first saltlen bytes of digest_b */
    uchar s_bytes[16];
    for (int i = 0; i < saltlen; i++)
        s_bytes[i] = digest_b[i];

    /* ---- Step 5: Main loop: rounds iterations ---- */
    for (int r = 0; r < rounds; r++) {
        sc_sha256_init(state);
        for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
        ctx_bufpos = 0; ctx_counter = 0;

        if (r & 1)
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, p_bytes, plen);
        else
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, curin, 32);

        if (r % 3)
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, s_bytes, saltlen);

        if (r % 7)
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, p_bytes, plen);

        if (r & 1)
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, curin, 32);
        else
            sc_sha256_update(state, ctx_buf, &ctx_bufpos, &ctx_counter, p_bytes, plen);

        sc_sha256_final(state, ctx_buf, ctx_bufpos, ctx_counter, curin);
    }

    /* ---- Check result against compact hash table ---- */
    /* curin[0..31] is the raw 32-byte hash.
     * Compact probe uses first 16 bytes as 4 x uint32 (little-endian read). */
    uint hx = (uint)curin[0] | ((uint)curin[1] << 8) |
              ((uint)curin[2] << 16) | ((uint)curin[3] << 24);
    uint hy = (uint)curin[4] | ((uint)curin[5] << 8) |
              ((uint)curin[6] << 16) | ((uint)curin[7] << 24);
    uint hz = (uint)curin[8] | ((uint)curin[9] << 8) |
              ((uint)curin[10] << 16) | ((uint)curin[11] << 24);
    uint hw = (uint)curin[12] | ((uint)curin[13] << 8) |
              ((uint)curin[14] << 16) | ((uint)curin[15] << 24);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        { uint _h[8];
            for (int _i = 0; _i < 8; _i++) _h[_i] = (uint)curin[_i*4] | ((uint)curin[_i*4+1] << 8) | ((uint)curin[_i*4+2] << 16) | ((uint)curin[_i*4+3] << 24);
            EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, _h) }
    }
}
