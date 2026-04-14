/* metal_sha512crypt.metal -- SHA512CRYPT (e513, hashcat 1800)
 * glibc crypt-sha512: $6$[rounds=N$]salt$hash
 * Default 5000 rounds, salt up to 16 chars, SHA512 (64-byte digest).
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 */

constant ulong K512_SC[80] = {
    0x428a2f98d728ae22UL,0x7137449123ef65cdUL,0xb5c0fbcfec4d3b2fUL,0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL,0x59f111f1b605d019UL,0x923f82a4af194f9bUL,0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL,0x12835b0145706fbeUL,0x243185be4ee4b28cUL,0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL,0x80deb1fe3b1696b1UL,0x9bdc06a725c71235UL,0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL,0xefbe4786384f25e3UL,0x0fc19dc68b8cd5b5UL,0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL,0x4a7484aa6ea6e483UL,0x5cb0a9dcbd41fbd4UL,0x76f988da831153b5UL,
    0x983e5152ee66dfabUL,0xa831c66d2db43210UL,0xb00327c898fb213fUL,0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL,0xd5a79147930aa725UL,0x06ca6351e003826fUL,0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL,0x2e1b21385c26c926UL,0x4d2c6dfc5ac42aedUL,0x53380d139d95b3dfUL,
    0x650a73548baf63deUL,0x766a0abb3c77b2a8UL,0x81c2c92e47edaee6UL,0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL,0xa81a664bbc423001UL,0xc24b8b70d0f89791UL,0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL,0xd69906245565a910UL,0xf40e35855771202aUL,0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL,0x1e376c085141ab53UL,0x2748774cdf8eeb99UL,0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL,0x4ed8aa4ae3418acbUL,0x5b9cca4f7763e373UL,0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL,0x78a5636f43172f60UL,0x84c87814a1f0ab72UL,0x8cc702081a6439ecUL,
    0x90befffa23631e28UL,0xa4506cebde82bde9UL,0xbef9a3f7b2c67915UL,0xc67178f2e372532bUL,
    0xca273eceea26619cUL,0xd186b8c721c0c207UL,0xeada7dd6cde0eb1eUL,0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL,0x0a637dc5a2c898a6UL,0x113f9804bef90daeUL,0x1b710b35131c471bUL,
    0x28db77f523047d84UL,0x32caab7b40c72493UL,0x3c9ebe0a15c9bebcUL,0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL,0x597f299cfc657e2aUL,0x5fcb6fab3ad6faecUL,0x6c44198c4a475817UL
};

static void sc_sha512_block(thread ulong *state, thread ulong *M) {
    /* Fully unrolled SHA-512 compression to avoid Metal/AMD compiler bug
     * that corrupts state in loops with 11+ iterations of 8-register rotation. */
    ulong W[16];
    for (int i=0;i<16;i++) W[i]=M[i];
    ulong a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    #define SC_R(i,wi) { \
        ulong S1=((e>>14)|(e<<50))^((e>>18)|(e<<46))^((e>>41)|(e<<23)); \
        ulong ch=(e&f)^(~e&g); ulong t1=h+S1+ch+K512_SC[i]+(wi); \
        ulong S0=((a>>28)|(a<<36))^((a>>34)|(a<<30))^((a>>39)|(a<<25)); \
        ulong maj=(a&b)^(a&c)^(b&c); ulong t2=S0+maj; \
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
    #define SC_WS(i) { \
        int j=(i)&15; \
        ulong x=W[((i)-15)&15]; ulong s0=((x>>1)|(x<<63))^((x>>8)|(x<<56))^(x>>7); \
        x=W[((i)-2)&15]; ulong s1=((x>>19)|(x<<45))^((x>>61)|(x<<3))^(x>>6); \
        W[j]=W[j]+s0+W[((i)-7)&15]+s1; }
    SC_R( 0,W[ 0]); SC_R( 1,W[ 1]); SC_R( 2,W[ 2]); SC_R( 3,W[ 3]);
    SC_R( 4,W[ 4]); SC_R( 5,W[ 5]); SC_R( 6,W[ 6]); SC_R( 7,W[ 7]);
    SC_R( 8,W[ 8]); SC_R( 9,W[ 9]); SC_R(10,W[10]); SC_R(11,W[11]);
    SC_R(12,W[12]); SC_R(13,W[13]); SC_R(14,W[14]); SC_R(15,W[15]);
    SC_WS(16); SC_R(16,W[ 0]); SC_WS(17); SC_R(17,W[ 1]); SC_WS(18); SC_R(18,W[ 2]); SC_WS(19); SC_R(19,W[ 3]);
    SC_WS(20); SC_R(20,W[ 4]); SC_WS(21); SC_R(21,W[ 5]); SC_WS(22); SC_R(22,W[ 6]); SC_WS(23); SC_R(23,W[ 7]);
    SC_WS(24); SC_R(24,W[ 8]); SC_WS(25); SC_R(25,W[ 9]); SC_WS(26); SC_R(26,W[10]); SC_WS(27); SC_R(27,W[11]);
    SC_WS(28); SC_R(28,W[12]); SC_WS(29); SC_R(29,W[13]); SC_WS(30); SC_R(30,W[14]); SC_WS(31); SC_R(31,W[15]);
    SC_WS(32); SC_R(32,W[ 0]); SC_WS(33); SC_R(33,W[ 1]); SC_WS(34); SC_R(34,W[ 2]); SC_WS(35); SC_R(35,W[ 3]);
    SC_WS(36); SC_R(36,W[ 4]); SC_WS(37); SC_R(37,W[ 5]); SC_WS(38); SC_R(38,W[ 6]); SC_WS(39); SC_R(39,W[ 7]);
    SC_WS(40); SC_R(40,W[ 8]); SC_WS(41); SC_R(41,W[ 9]); SC_WS(42); SC_R(42,W[10]); SC_WS(43); SC_R(43,W[11]);
    SC_WS(44); SC_R(44,W[12]); SC_WS(45); SC_R(45,W[13]); SC_WS(46); SC_R(46,W[14]); SC_WS(47); SC_R(47,W[15]);
    SC_WS(48); SC_R(48,W[ 0]); SC_WS(49); SC_R(49,W[ 1]); SC_WS(50); SC_R(50,W[ 2]); SC_WS(51); SC_R(51,W[ 3]);
    SC_WS(52); SC_R(52,W[ 4]); SC_WS(53); SC_R(53,W[ 5]); SC_WS(54); SC_R(54,W[ 6]); SC_WS(55); SC_R(55,W[ 7]);
    SC_WS(56); SC_R(56,W[ 8]); SC_WS(57); SC_R(57,W[ 9]); SC_WS(58); SC_R(58,W[10]); SC_WS(59); SC_R(59,W[11]);
    SC_WS(60); SC_R(60,W[12]); SC_WS(61); SC_R(61,W[13]); SC_WS(62); SC_R(62,W[14]); SC_WS(63); SC_R(63,W[15]);
    SC_WS(64); SC_R(64,W[ 0]); SC_WS(65); SC_R(65,W[ 1]); SC_WS(66); SC_R(66,W[ 2]); SC_WS(67); SC_R(67,W[ 3]);
    SC_WS(68); SC_R(68,W[ 4]); SC_WS(69); SC_R(69,W[ 5]); SC_WS(70); SC_R(70,W[ 6]); SC_WS(71); SC_R(71,W[ 7]);
    SC_WS(72); SC_R(72,W[ 8]); SC_WS(73); SC_R(73,W[ 9]); SC_WS(74); SC_R(74,W[10]); SC_WS(75); SC_R(75,W[11]);
    SC_WS(76); SC_R(76,W[12]); SC_WS(77); SC_R(77,W[13]); SC_WS(78); SC_R(78,W[14]); SC_WS(79); SC_R(79,W[15]);
    #undef SC_R
    #undef SC_WS
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void sc_buf_set_byte64(thread ulong *buf, int pos, uchar val) {
    int wi=pos>>3; int bi=(7-(pos&7))<<3;
    buf[wi]=(buf[wi]&~(0xffUL<<bi))|((ulong)val<<bi);
}

static void sc_sha512_init(thread ulong *state) {
    state[0]=0x6a09e667f3bcc908UL;state[1]=0xbb67ae8584caa73bUL;
    state[2]=0x3c6ef372fe94f82bUL;state[3]=0xa54ff53a5f1d36f1UL;
    state[4]=0x510e527fade682d1UL;state[5]=0x9b05688c2b3e6c1fUL;
    state[6]=0x1f83d9abfb41bd6bUL;state[7]=0x5be0cd19137e2179UL;
}

static void sc_sha512_update(thread ulong *state, thread ulong *buf, thread int *bufpos,
                              thread ulong *counter, thread const uchar *data, int len) {
    *counter+=(ulong)len; int bp=*bufpos;
    for (int i=0;i<len;i++) {
        sc_buf_set_byte64(buf,bp,data[i]); bp++;
        if (bp==128) { sc_sha512_block(state,buf); for (int j=0;j<16;j++) buf[j]=0; bp=0; }
    }
    *bufpos=bp;
}

static void sc_sha512_update_g(thread ulong *state, thread ulong *buf, thread int *bufpos,
                                thread ulong *counter, device const uint8_t *data, int len) {
    *counter+=(ulong)len; int bp=*bufpos;
    for (int i=0;i<len;i++) {
        sc_buf_set_byte64(buf,bp,data[i]); bp++;
        if (bp==128) { sc_sha512_block(state,buf); for (int j=0;j<16;j++) buf[j]=0; bp=0; }
    }
    *bufpos=bp;
}

static void sc_sha512_final(thread ulong *state, thread ulong *buf, int bufpos,
                             ulong counter, thread uchar *out) {
    sc_buf_set_byte64(buf,bufpos,0x80); bufpos++;
    if (bufpos>112) {
        for (int i=bufpos;i<128;i++) sc_buf_set_byte64(buf,i,0);
        sc_sha512_block(state,buf); for (int j=0;j<16;j++) buf[j]=0; bufpos=0;
    }
    for (int i=bufpos;i<112;i++) sc_buf_set_byte64(buf,i,0);
    buf[14]=0; buf[15]=counter*8;
    sc_sha512_block(state,buf);
    for (int i=0;i<8;i++) {
        ulong w=state[i];
        out[i*8]=(uchar)(w>>56); out[i*8+1]=(uchar)(w>>48);
        out[i*8+2]=(uchar)(w>>40); out[i*8+3]=(uchar)(w>>32);
        out[i*8+4]=(uchar)(w>>24); out[i*8+5]=(uchar)(w>>16);
        out[i*8+6]=(uchar)(w>>8); out[i*8+7]=(uchar)(w);
    }
}

static void sc_sha512_oneshot(thread const uchar *data, int len, thread uchar *out) {
    ulong state[8],buf[16]; int bufpos=0; ulong counter=0;
    sc_sha512_init(state); for (int j=0;j<16;j++) buf[j]=0;
    sc_sha512_update(state,buf,&bufpos,&counter,data,len);
    sc_sha512_final(state,buf,bufpos,counter,out);
}

kernel void sha512crypt_batch(
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
    uchar pw[256]; for (int i=0;i<plen;i++) pw[i]=pass[i];

    uint soff = salt_offsets[salt_idx];
    int slen_full = salt_lens[salt_idx];
    device const uint8_t *salt_str = salts + soff;

    int spos=3; int rounds=5000;
    if (slen_full>10 && salt_str[3]=='r' && salt_str[4]=='o' && salt_str[5]=='u' &&
        salt_str[6]=='n' && salt_str[7]=='d' && salt_str[8]=='s' && salt_str[9]=='=') {
        rounds=0; spos=10;
        while (spos<slen_full && salt_str[spos]>='0' && salt_str[spos]<='9') {
            rounds=rounds*10+(salt_str[spos]-'0'); spos++;
        }
        if (rounds<1000) rounds=1000;
        if (rounds>999999999) rounds=999999999;
        if (spos<slen_full && salt_str[spos]=='$') spos++;
    }
    uchar raw_salt[16]; int saltlen=0;
    for (int i=spos;i<slen_full && saltlen<16;i++) {
        if (salt_str[i]=='$') break;
        raw_salt[saltlen++]=salt_str[i];
    }
    if (saltlen==0) return;

    ulong state[8],ctx_buf[16]; int ctx_bufpos; ulong ctx_counter;
    uchar tmp[256],digest_a[64],digest_b[64];

    /* Step 1 */
    { int tlen=0; for (int i=0;i<plen;i++) tmp[tlen++]=pw[i];
      for (int i=0;i<saltlen;i++) tmp[tlen++]=raw_salt[i];
      for (int i=0;i<plen;i++) tmp[tlen++]=pw[i];
      sc_sha512_oneshot(tmp,tlen,digest_a); }

    /* Step 2 */
    sc_sha512_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,raw_salt,saltlen);
    for (int x=plen;x>64;x-=64) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,64);
    { int x=plen; while(x>64) x-=64; sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,x); }
    for (int x=plen;x!=0;x>>=1) {
        if (x&1) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,digest_a,64);
        else sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    }
    uchar curin[64];
    sc_sha512_final(state,ctx_buf,ctx_bufpos,ctx_counter,curin);

    /* Step 3: Hash P */
    sc_sha512_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    for (int x=0;x<plen;x++) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,pw,plen);
    sc_sha512_final(state,ctx_buf,ctx_bufpos,ctx_counter,digest_b);
    uchar p_bytes[256]; for (int i=0;i<plen;i++) p_bytes[i]=digest_b[i%64];

    /* Step 4: Hash S */
    sc_sha512_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
    int s_repeats=16+(uint)curin[0];
    for (int x=0;x<s_repeats;x++) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,raw_salt,saltlen);
    sc_sha512_final(state,ctx_buf,ctx_bufpos,ctx_counter,digest_b);
    uchar s_bytes[16]; for (int i=0;i<saltlen;i++) s_bytes[i]=digest_b[i];

    /* Step 5: Main loop */
    for (int r=0;r<rounds;r++) {
        sc_sha512_init(state); for (int j=0;j<16;j++) ctx_buf[j]=0; ctx_bufpos=0; ctx_counter=0;
        if (r&1) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        else sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,curin,64);
        if (r%3) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,s_bytes,saltlen);
        if (r%7) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        if (r&1) sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,curin,64);
        else sc_sha512_update(state,ctx_buf,&ctx_bufpos,&ctx_counter,p_bytes,plen);
        sc_sha512_final(state,ctx_buf,ctx_bufpos,ctx_counter,curin);
    }

    uint hx=(uint)curin[0]|((uint)curin[1]<<8)|((uint)curin[2]<<16)|((uint)curin[3]<<24);
    uint hy=(uint)curin[4]|((uint)curin[5]<<8)|((uint)curin[6]<<16)|((uint)curin[7]<<24);
    uint hz=(uint)curin[8]|((uint)curin[9]<<8)|((uint)curin[10]<<16)|((uint)curin[11]<<24);
    uint hw=(uint)curin[12]|((uint)curin[13]<<8)|((uint)curin[14]<<16)|((uint)curin[15]<<24);

    /* DBG: test block function with forced SHA512("a") */
    if (word_idx == 0 && salt_idx == params.salt_start) {
        ulong ts[8],tm[16];
        ts[0]=0x6a09e667f3bcc908UL;ts[1]=0xbb67ae8584caa73bUL;
        ts[2]=0x3c6ef372fe94f82bUL;ts[3]=0xa54ff53a5f1d36f1UL;
        ts[4]=0x510e527fade682d1UL;ts[5]=0x9b05688c2b3e6c1fUL;
        ts[6]=0x1f83d9abfb41bd6bUL;ts[7]=0x5be0cd19137e2179UL;
        tm[0]=0x6180000000000000UL;
        for (int i=1;i<15;i++) tm[i]=0;
        tm[14]=0; tm[15]=8;
        sc_sha512_block(ts,tm);
        uint slot=atomic_fetch_add_explicit(hit_count,1,memory_order_relaxed);
        if (slot<params.max_hits) {
            uint base=slot*7;
            hits[base]=0xDEAD; hits[base+1]=0xBEEF; hits[base+2]=0xFF;
            hits[base+3]=(uint)(ts[0]); hits[base+4]=(uint)(ts[0]>>32);
            hits[base+5]=(uint)(ts[1]); hits[base+6]=(uint)(ts[1]>>32);
        }
    }

    uint4 h = uint4(hx,hy,hz,hw);

    PROBE7_NOOVF(h, word_idx, salt_idx)
}
