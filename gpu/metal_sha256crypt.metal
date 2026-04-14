/* metal_sha256crypt.metal -- SHA256CRYPT (e512, hashcat 7400)
 * glibc crypt-sha256: $5$[rounds=N$]salt$hash
 * Default 5000 rounds, salt up to 16 chars, SHA256 (32-byte digest).
 * Hit stride: 11 (word_idx, salt_idx, iter, h[0..7])
 */

constant uint K256_SC[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static void sc_sha256_block(thread uint *state, thread uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++) {
        uint x = W[i-15]; uint s0 = ((x>>7)|(x<<25))^((x>>18)|(x<<14))^(x>>3);
        x = W[i-2]; uint s1 = ((x>>17)|(x<<15))^((x>>19)|(x<<13))^(x>>10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i = 0; i < 64; i++) {
        uint S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
        uint ch=(e&f)^(~e&g); uint t1=h+S1+ch+K256_SC[i]+W[i];
        uint S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint maj=(a&b)^(a&c)^(b&c); uint t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void sc_buf_set_byte(thread uint *buf, int pos, uchar val) {
    int wi = pos >> 2; int bi = (3 - (pos & 3)) << 3;
    buf[wi] = (buf[wi] & ~(0xffu << bi)) | ((uint)val << bi);
}

static void sc_sha256_init(thread uint *state) {
    state[0]=0x6a09e667u;state[1]=0xbb67ae85u;state[2]=0x3c6ef372u;state[3]=0xa54ff53au;
    state[4]=0x510e527fu;state[5]=0x9b05688cu;state[6]=0x1f83d9abu;state[7]=0x5be0cd19u;
}

static void sc_sha256_update(thread uint *state, thread uint *buf, thread int *bufpos,
                              thread uint *counter, thread const uchar *data, int len) {
    *counter += (uint)len;
    int bp = *bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]); bp++;
        if (bp == 64) { sc_sha256_block(state, buf); for (int j=0;j<16;j++) buf[j]=0; bp=0; }
    }
    *bufpos = bp;
}

static void sc_sha256_update_g(thread uint *state, thread uint *buf, thread int *bufpos,
                                thread uint *counter, device const uint8_t *data, int len) {
    *counter += (uint)len;
    int bp = *bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]); bp++;
        if (bp == 64) { sc_sha256_block(state, buf); for (int j=0;j<16;j++) buf[j]=0; bp=0; }
    }
    *bufpos = bp;
}

static void sc_sha256_final(thread uint *state, thread uint *buf, int bufpos,
                             uint counter, thread uchar *out) {
    sc_buf_set_byte(buf, bufpos, 0x80); bufpos++;
    if (bufpos > 56) {
        for (int i=bufpos;i<64;i++) sc_buf_set_byte(buf,i,0);
        sc_sha256_block(state,buf); for (int j=0;j<16;j++) buf[j]=0; bufpos=0;
    }
    for (int i=bufpos;i<56;i++) sc_buf_set_byte(buf,i,0);
    buf[14]=0; buf[15]=counter*8;
    sc_sha256_block(state,buf);
    for (int i=0;i<8;i++) {
        uint w=state[i];
        out[i*4]=(uchar)(w>>24); out[i*4+1]=(uchar)(w>>16);
        out[i*4+2]=(uchar)(w>>8); out[i*4+3]=(uchar)(w);
    }
}

static void sc_sha256_oneshot(thread const uchar *data, int len, thread uchar *out) {
    uint state[8],buf[16]; int bufpos=0; uint counter=0;
    sc_sha256_init(state); for (int j=0;j<16;j++) buf[j]=0;
    sc_sha256_update(state,buf,&bufpos,&counter,data,len);
    sc_sha256_final(state,buf,bufpos,counter,out);
}

kernel void sha256crypt_batch(
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

    int plen = hexlens[word_idx];
    device const uint8_t *pass = hexhashes + word_idx * 256;
    uchar pw[256];
    for (int i = 0; i < plen; i++) pw[i] = pass[i];

    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];
    device const uint8_t *salt_str = salts + soff;

    int spos = 3; int rounds = 5000;
    if (slen_full > 10 && salt_str[3]=='r' && salt_str[4]=='o' && salt_str[5]=='u' &&
        salt_str[6]=='n' && salt_str[7]=='d' && salt_str[8]=='s' && salt_str[9]=='=') {
        rounds = 0; spos = 10;
        while (spos < slen_full && salt_str[spos] >= '0' && salt_str[spos] <= '9') {
            rounds = rounds * 10 + (salt_str[spos] - '0'); spos++;
        }
        if (rounds < 1000) rounds = 1000;
        if (rounds > 999999999) rounds = 999999999;
        if (spos < slen_full && salt_str[spos] == '$') spos++;
    }
    uchar raw_salt[16]; int saltlen = 0;
    for (int i = spos; i < slen_full && saltlen < 16; i++) {
        if (salt_str[i] == '$') break;
        raw_salt[saltlen++] = salt_str[i];
    }
    if (saltlen == 0) return;

    uint state[8], ctx_buf[16]; int ctx_bufpos; uint ctx_counter;
    uchar tmp[256], digest_a[32], digest_b[32];

    /* Step 1 */
    { int tlen=0; for (int i=0;i<plen;i++) tmp[tlen++]=pw[i];
      for (int i=0;i<saltlen;i++) tmp[tlen++]=raw_salt[i];
      for (int i=0;i<plen;i++) tmp[tlen++]=pw[i];
      sc_sha256_oneshot(tmp,tlen,digest_a); }

    /* Step 2 */
    sc_sha256_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,raw_salt,saltlen);
    for (int x=plen;x>32;x-=32) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,32);
    { int x=plen; while(x>32) x-=32; sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,x); }
    for (int x=plen;x!=0;x>>=1) {
        if (x&1) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,32);
        else sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    }
    uchar curin[32];
    sc_sha256_final(state,ctx_buf,ctx_bufpos,ctx_counter,curin);

    /* Step 3: Hash P */
    sc_sha256_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    for (int x=0;x<plen;x++) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    sc_sha256_final(state,ctx_buf,ctx_bufpos,ctx_counter,digest_b);
    uchar p_bytes[256]; for (int i=0;i<plen;i++) p_bytes[i]=digest_b[i%32];

    /* Step 4: Hash S */
    sc_sha256_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    int s_repeats = 16 + (uint)curin[0];
    for (int x=0;x<s_repeats;x++) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,raw_salt,saltlen);
    sc_sha256_final(state,ctx_buf,ctx_bufpos,ctx_counter,digest_b);
    uchar s_bytes[16]; for (int i=0;i<saltlen;i++) s_bytes[i]=digest_b[i];

    /* Step 5: Main loop */
    for (int r=0;r<rounds;r++) {
        sc_sha256_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
        if (r&1) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        else sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,curin,32);
        if (r%3) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,s_bytes,saltlen);
        if (r%7) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        if (r&1) sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,curin,32);
        else sc_sha256_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        sc_sha256_final(state,ctx_buf,ctx_bufpos,ctx_counter,curin);
    }

    uint hx=(uint)curin[0]|((uint)curin[1]<<8)|((uint)curin[2]<<16)|((uint)curin[3]<<24);
    uint hy=(uint)curin[4]|((uint)curin[5]<<8)|((uint)curin[6]<<16)|((uint)curin[7]<<24);
    uint hz=(uint)curin[8]|((uint)curin[9]<<8)|((uint)curin[10]<<16)|((uint)curin[11]<<24);
    uint hw=(uint)curin[12]|((uint)curin[13]<<8)|((uint)curin[14]<<16)|((uint)curin[15]<<24);

    uint4 h = uint4(hx, hy, hz, hw);
    ulong key = (ulong(h.y) << 32) | h.x;
    uint fp = uint(key >> 32); if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & params.compact_mask;
    for (uint p = 0; p < params.max_probe; p++) {
        uint cfp = compact_fp[pos]; if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h.x==ref[0]&&h.y==ref[1]&&h.z==ref[2]&&h.w==ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 11;
                        hits[base]=word_idx; hits[base+1]=salt_idx; hits[base+2]=1;
                        hits[base+3]=hx; hits[base+4]=hy; hits[base+5]=hz; hits[base+6]=hw;
                        hits[base+7]=(uint)curin[16]|((uint)curin[17]<<8)|((uint)curin[18]<<16)|((uint)curin[19]<<24);
                        hits[base+8]=(uint)curin[20]|((uint)curin[21]<<8)|((uint)curin[22]<<16)|((uint)curin[23]<<24);
                        hits[base+9]=(uint)curin[24]|((uint)curin[25]<<8)|((uint)curin[26]<<16)|((uint)curin[27]<<24);
                        hits[base+10]=(uint)curin[28]|((uint)curin[29]<<8)|((uint)curin[30]<<16)|((uint)curin[31]<<24);
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
}
