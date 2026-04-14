/* metal_hmac_rmd160.metal -- HMAC-RIPEMD-160 GPU kernels
 * Hit stride: 7 (word_idx, salt_idx, iter, h0, h1, h2, h3)
 */

/* rmd160_block, SALTED_PARAMS, PROBE7_NOOVF all provided by metal_common.metal */
#define PROBE7(oh0,oh1,oh2,oh3,widx,sidx) { \
    uint4 _ph = uint4(oh0,oh1,oh2,oh3); \
    PROBE7_NOOVF(_ph, widx, sidx) }

/* HMAC helper: copy bytes into LE uint[] */
static void rmd_copy_g(thread uint *M, int off, device const uint8_t *src, int len) {
    thread uchar *mb = (thread uchar *)M;
    for (int i = 0; i < len; i++) mb[off + i] = src[i];
}
static void rmd_copy_p(thread uint *M, int off, thread const uchar *src, int len) {
    thread uchar *mb = (thread uchar *)M;
    for (int i = 0; i < len; i++) mb[off + i] = src[i];
}

kernel void hmac_rmd160_ksalt_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int plen=hexlens[word_idx]; device const uint8_t *pass=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int klen=salt_lens[salt_idx];
    device const uint8_t *key=salts+soff;
    uint key_block[16]; for(int i=0;i<16;i++) key_block[i]=0;
    if(klen>64) {
        uint kh[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
        uint KM[16]; for(int i=0;i<16;i++) KM[i]=0;
        rmd_copy_g(KM,0,key,64); rmd160_block(kh,KM);
        for(int i=0;i<16;i++) KM[i]=0;
        int rem=klen-64; if(rem>0) rmd_copy_g(KM,0,key+64,rem);
        ((thread uchar*)KM)[rem]=0x80; KM[14]=klen*8; rmd160_block(kh,KM);
        for(int i=0;i<5;i++) key_block[i]=kh[i]; klen=20;
    } else { rmd_copy_g(key_block,0,key,klen); }
    uint ipad[16],M[16];
    for(int i=0;i<16;i++) ipad[i]=key_block[i]^0x36363636u;
    for(int i=0;i<16;i++) M[i]=ipad[i];
    uint ih[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
    rmd160_block(ih,M);
    for(int i=0;i<16;i++) M[i]=0;
    if(plen<=55) { rmd_copy_g(M,0,pass,plen); ((thread uchar*)M)[plen]=0x80; M[14]=(64+plen)*8; rmd160_block(ih,M); }
    else { int c1=plen<64?plen:64; rmd_copy_g(M,0,pass,c1); if(plen<64)((thread uchar*)M)[plen]=0x80;
           rmd160_block(ih,M); for(int i=0;i<16;i++) M[i]=0;
           int rem=plen-64; if(rem>0) rmd_copy_g(M,0,pass+64,rem);
           if(plen>=64) ((thread uchar*)M)[rem>0?rem:0]=0x80; M[14]=(64+plen)*8; rmd160_block(ih,M); }
    uint opad_block[16]; for(int i=0;i<16;i++) opad_block[i]=key_block[i]^0x5c5c5c5cu;
    for(int i=0;i<16;i++) M[i]=opad_block[i];
    uint oh[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
    rmd160_block(oh,M);
    for(int i=0;i<16;i++) M[i]=0;
    M[0]=ih[0];M[1]=ih[1];M[2]=ih[2];M[3]=ih[3];M[4]=ih[4];
    M[5]=0x80; M[14]=(64+20)*8; rmd160_block(oh,M);
    PROBE7(oh[0],oh[1],oh[2],oh[3],word_idx,salt_idx)
}

kernel void hmac_rmd160_kpass_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int klen=hexlens[word_idx]; device const uint8_t *key=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int mlen=salt_lens[salt_idx];
    device const uint8_t *msg=salts+soff;
    uint key_block[16]; for(int i=0;i<16;i++) key_block[i]=0;
    if(klen>64) {
        uint kh[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
        uint KM[16]; for(int i=0;i<16;i++) KM[i]=0;
        rmd_copy_g(KM,0,key,64); rmd160_block(kh,KM);
        for(int i=0;i<16;i++) KM[i]=0;
        int rem=klen-64; if(rem>0) rmd_copy_g(KM,0,key+64,rem);
        ((thread uchar*)KM)[rem]=0x80; KM[14]=klen*8; rmd160_block(kh,KM);
        for(int i=0;i<5;i++) key_block[i]=kh[i]; klen=20;
    } else { rmd_copy_g(key_block,0,key,klen); }
    uint ipad[16],M[16];
    for(int i=0;i<16;i++) ipad[i]=key_block[i]^0x36363636u;
    for(int i=0;i<16;i++) M[i]=ipad[i];
    uint ih[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
    rmd160_block(ih,M);
    for(int i=0;i<16;i++) M[i]=0;
    if(mlen<=55) { rmd_copy_g(M,0,msg,mlen); ((thread uchar*)M)[mlen]=0x80; M[14]=(64+mlen)*8; rmd160_block(ih,M); }
    else { int c1=mlen<64?mlen:64; rmd_copy_g(M,0,msg,c1); if(mlen<64)((thread uchar*)M)[mlen]=0x80;
           rmd160_block(ih,M); for(int i=0;i<16;i++) M[i]=0;
           int rem=mlen-64; if(rem>0) rmd_copy_g(M,0,msg+64,rem);
           if(mlen>=64) ((thread uchar*)M)[rem>0?rem:0]=0x80; M[14]=(64+mlen)*8; rmd160_block(ih,M); }
    uint opad_block[16]; for(int i=0;i<16;i++) opad_block[i]=key_block[i]^0x5c5c5c5cu;
    for(int i=0;i<16;i++) M[i]=opad_block[i];
    uint oh[5]={0x67452301u,0xEFCDAB89u,0x98BADCFEu,0x10325476u,0xC3D2E1F0u};
    rmd160_block(oh,M);
    for(int i=0;i<16;i++) M[i]=0;
    M[0]=ih[0];M[1]=ih[1];M[2]=ih[2];M[3]=ih[3];M[4]=ih[4];
    M[5]=0x80; M[14]=(64+20)*8; rmd160_block(oh,M);
    PROBE7(oh[0],oh[1],oh[2],oh[3],word_idx,salt_idx)
}
