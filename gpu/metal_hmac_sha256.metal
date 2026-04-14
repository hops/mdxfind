/* metal_hmac_sha256.metal — HMAC-SHA256 GPU kernels
 * Hit stride: 11 (word_idx, salt_idx, iter, h[0..7])
 */

/* K256[], sha256_block, bswap32, S_copy_bytes, S_set_byte,
 * SALTED_PARAMS, PROBE11_NOOVF all provided by metal_common.metal */
#define PROBE11 PROBE11_NOOVF

#define HMAC_SHA256_BODY(kdata, klen_v, mdata, mlen_v) \
    uint key_block[16]; for(int i=0;i<16;i++) key_block[i]=0; \
    int klen=klen_v; \
    if(klen>64) { \
        uint kst[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au, \
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}; \
        uint KM[16]; for(int i=0;i<16;i++) KM[i]=0; \
        S_copy_bytes(KM,0,kdata,klen<64?klen:64); \
        if(klen<=55) { S_set_byte(KM,klen,0x80); KM[15]=klen*8; } \
        sha256_block(kst,KM); \
        if(klen>55) { for(int i=0;i<16;i++) KM[i]=0; int rem=klen-64; \
            if(rem>0) S_copy_bytes(KM,0,kdata+64,rem); S_set_byte(KM,rem,0x80); KM[15]=klen*8; sha256_block(kst,KM); } \
        for(int i=0;i<8;i++) key_block[i]=bswap32(kst[i]); klen=32; \
    } else { for(int i=0;i<klen;i++) key_block[i>>2]|=((uint)kdata[i])<<((i&3)<<3); } \
    uint ipad2[16],M2[16]; \
    for(int i=0;i<16;i++) ipad2[i]=key_block[i]^0x36363636u; \
    for(int i=0;i<16;i++) M2[i]=bswap32(ipad2[i]); \
    uint ist[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au, \
                 0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}; \
    sha256_block(ist,M2); \
    int mlen=mlen_v; \
    for(int i=0;i<16;i++) M2[i]=0; \
    if(mlen<=55) { S_copy_bytes(M2,0,mdata,mlen); S_set_byte(M2,mlen,0x80); M2[15]=(64+mlen)*8; sha256_block(ist,M2); } \
    else { S_copy_bytes(M2,0,mdata,mlen<64?mlen:64); if(mlen<64) S_set_byte(M2,mlen,0x80); \
           sha256_block(ist,M2); for(int i=0;i<16;i++) M2[i]=0; int rem=mlen-64; \
           if(rem>0) S_copy_bytes(M2,0,mdata+64,rem); if(mlen>=64) S_set_byte(M2,rem>0?rem:0,0x80); \
           M2[15]=(64+mlen)*8; sha256_block(ist,M2); } \
    uint op2[16]; for(int i=0;i<16;i++) op2[i]=key_block[i]^0x5c5c5c5cu; \
    for(int i=0;i<16;i++) M2[i]=bswap32(op2[i]); \
    uint ost[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au, \
                 0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}; \
    sha256_block(ost,M2); \
    for(int i=0;i<8;i++) M2[i]=ist[i]; M2[8]=0x80000000u; \
    for(int i=9;i<15;i++) M2[i]=0; M2[15]=(64+32)*8; sha256_block(ost,M2); \
    uint hout[8]; for(int i=0;i<8;i++) hout[i]=bswap32(ost[i]);

kernel void hmac_sha256_ksalt_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if(word_idx>=params.num_words) return;
    device const uint8_t *mdata=hexhashes+word_idx*256; int mlen_v=hexlens[word_idx];
    uint soff=salt_offsets[salt_idx]; device const uint8_t *kdata=salts+soff; int klen_v=salt_lens[salt_idx];
    HMAC_SHA256_BODY(kdata,klen_v,mdata,mlen_v)
    PROBE11(hout[0],hout[1],hout[2],hout[3],hout,word_idx,salt_idx)
}

kernel void hmac_sha256_kpass_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if(word_idx>=params.num_words) return;
    device const uint8_t *kdata=hexhashes+word_idx*256; int klen_v=hexlens[word_idx];
    uint soff=salt_offsets[salt_idx]; device const uint8_t *mdata=salts+soff; int mlen_v=salt_lens[salt_idx];
    HMAC_SHA256_BODY(kdata,klen_v,mdata,mlen_v)
    PROBE11(hout[0],hout[1],hout[2],hout[3],hout,word_idx,salt_idx)
}
