/*
 * gpu_opencl.c — OpenCL GPU acceleration for mdxfind
 *
 * Cross-vendor: NVIDIA, AMD, Intel, Apple (via OpenCL compatibility).
 * Kernel source compiled at runtime via clCreateProgramWithSource.
 */

#if defined(OPENCL_GPU)

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "opencl_dynload.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "gpu_opencl.h"
#include "job_types.h"
#include "gpujob.h"

/* ---- Multi-GPU device state ---- */
#define MAX_GPU_DEVICES 64

struct gpu_device {
    cl_context       ctx;
    cl_command_queue  queue;
    cl_program        prog;
    cl_device_id      dev;
    cl_kernel         kern_salt_iter; /* special: salted iteration override for Maxiter>1 */
    char              name[256];
    int               max_batch;     /* per-device batch limit (words per dispatch) */
    int               max_dispatch;  /* max work items per dispatch (0 = unlimited) */

    /* Compact table (read-only, uploaded once) */
    cl_mem b_compact_fp, b_compact_idx;
    cl_mem b_hash_data, b_hash_data_off, b_hash_data_len;

    /* Salt buffers (updated per-snapshot) */
    cl_mem b_salt_data, b_salt_off, b_salt_len;
    size_t salt_data_cap;
    int salts_count;

    /* Overflow (uploaded once) */
    cl_mem b_overflow_keys, b_overflow_hashes, b_overflow_offsets, b_overflow_lengths;

    /* Per-dispatch buffers */
    cl_mem b_hits, b_hit_count;
    cl_mem b_hexhashes, b_hexlens, b_params;
    size_t hexhash_cap;
    uint32_t *h_hits;    /* host-side hit buffer */
};

static struct gpu_device gpu_devs[MAX_GPU_DEVICES];
static int num_gpu_devs = 0;
static int ocl_ready = 0;

/* Shared state (same across all devices) */
static uint64_t _compact_mask = 0;
static uint32_t _hash_data_count = 0;
static int _overflow_count = 0;
static int _max_iter = 1;
static int _gpu_op = 0;

/* ---- Per-kernel autotune state ---- */
#define TUNE_CANDIDATES 4
static const size_t tune_sizes[TUNE_CANDIDATES] = { 64, 128, 256, 512 };

#define MAX_GPU_KERNELS 2048   /* indexed by op type, same as JOB_DONE */

struct gpu_kern {
    cl_kernel kernel;        /* NULL if no GPU kernel for this op */
    size_t local_size;       /* current work group size */
    size_t max_local;        /* from CL_KERNEL_WORK_GROUP_SIZE */
    int tuned;               /* 1 = done tuning */
    int tune_candidate;      /* which tune_sizes[] we're testing */
    int tune_samples;        /* dispatches at current candidate */
    double tune_best_time;   /* best avg ms so far */
    size_t tune_best_size;   /* size that produced best_time */
    double tune_cur_total;   /* accumulated ms for current candidate */
    int dev_idx;             /* device index for status messages */
};

/* Per-device kernel table — each device has its own compiled kernels */
struct gpu_kern_table {
    struct gpu_kern kerns[MAX_GPU_KERNELS];
};
static struct gpu_kern_table dev_kerns[MAX_GPU_DEVICES];

#define TUNE_SAMPLES 3  /* dispatches per candidate before moving on */

static void kern_register(int di, int op, cl_kernel kernel) {
    if (di < 0 || di >= MAX_GPU_DEVICES) return;
    if (op < 0 || op >= MAX_GPU_KERNELS || !kernel) return;
    struct gpu_kern *k = &dev_kerns[di].kerns[op];
    size_t max_wg = 0;
    clGetKernelWorkGroupInfo(kernel, gpu_devs[di].dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    k->kernel = kernel;
    k->max_local = max_wg;
    k->dev_idx = di;
    k->tuned = 0;
    k->tune_candidate = 0;
    k->tune_samples = 0;
    k->tune_best_time = 1e30;
    k->tune_best_size = 64;
    k->tune_cur_total = 0;
    int c = 0;
    while (c < TUNE_CANDIDATES && tune_sizes[c] > max_wg) c++;
    k->tune_candidate = c;
    k->local_size = (c < TUNE_CANDIDATES) ? tune_sizes[c] : max_wg;
}

static size_t kern_get_local_size(struct gpu_kern *k) {
    if (k->tuned) return k->local_size;
    if (k->tune_candidate >= TUNE_CANDIDATES) {
        k->tuned = 1;
        k->local_size = k->tune_best_size;
        fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        return k->local_size;
    }
    return tune_sizes[k->tune_candidate];
}

static void kern_record_time(struct gpu_kern *k, double ms) {
    if (k->tuned) return;
    k->tune_cur_total += ms;
    k->tune_samples++;
    if (k->tune_samples >= TUNE_SAMPLES) {
        double avg = k->tune_cur_total / k->tune_samples;
        if (avg < k->tune_best_time) {
            k->tune_best_time = avg;
            k->tune_best_size = tune_sizes[k->tune_candidate];
        }
        k->tune_candidate++;
        while (k->tune_candidate < TUNE_CANDIDATES && tune_sizes[k->tune_candidate] > k->max_local)
            k->tune_candidate++;
        k->tune_samples = 0;
        k->tune_cur_total = 0;
        if (k->tune_candidate >= TUNE_CANDIDATES) {
            k->tuned = 1;
            k->local_size = k->tune_best_size;
            fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        }
    }
}

#define GPU_MAX_HITS 32768

static uint32_t *h_hits = NULL;

/* ---- Params struct (must match kernel) ---- */
typedef struct {
    uint64_t compact_mask;
    uint32_t num_words;
    uint32_t num_salts;
    uint32_t salt_start;
    uint32_t max_probe;
    uint32_t hash_data_count;
    uint32_t max_hits;
    uint32_t overflow_count;
    uint32_t max_iter;
} OCLParams;

/* ---- Load kernel source from file ---- */
static char *load_kernel_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Try gpu/ prefix */
        char buf[512];
        snprintf(buf, sizeof(buf), "gpu/%s", path);
        f = fopen(buf, "r");
    }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char *)malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);
    return src;
}

/* ---- Embedded kernel source (auto-generated from gpu_kernels.cl) ---- */
#include "gpu_kernels_str.h"
static const char *kernel_source_embedded_unused =
"typedef struct {\n"
"    ulong compact_mask;\n"
"    uint num_words;\n"
"    uint num_salts;\n"
"    uint max_probe;\n"
"    uint hash_data_count;\n"
"    uint max_hits;\n"
"    uint overflow_count;\n"
"    uint max_iter;\n"
"} OCLParams;\n"
"\n"
"__constant uint K[64] = {\n"
"    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,\n"
"    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,\n"
"    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,\n"
"    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,\n"
"    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,\n"
"    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,\n"
"    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,\n"
"    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391\n"
"};\n"
"__constant uint S[64] = {\n"
"    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,\n"
"    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,\n"
"    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,\n"
"    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21\n"
"};\n"
"\n"
"#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotate(a,s); }\n"
"#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotate(a,s); }\n"
"#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + rotate(a,s); }\n"
"#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + rotate(a,s); }\n"
"\n"
"void md5_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {\n"
"    uint a = *h0, b = *h1, c = *h2, d = *h3;\n"
"    FF(a,b,c,d,M[0],(uint)7,0xd76aa478u);  FF(d,a,b,c,M[1],(uint)12,0xe8c7b756u);\n"
"    FF(c,d,a,b,M[2],(uint)17,0x242070dbu);  FF(b,c,d,a,M[3],(uint)22,0xc1bdceeeu);\n"
"    FF(a,b,c,d,M[4],(uint)7,0xf57c0fafu);   FF(d,a,b,c,M[5],(uint)12,0x4787c62au);\n"
"    FF(c,d,a,b,M[6],(uint)17,0xa8304613u);  FF(b,c,d,a,M[7],(uint)22,0xfd469501u);\n"
"    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);\n"
"    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);\n"
"    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);\n"
"    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);\n"
"    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);\n"
"    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);\n"
"    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);\n"
"    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);\n"
"    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);\n"
"    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);\n"
"    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);\n"
"    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);\n"
"    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);\n"
"    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);\n"
"    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);\n"
"    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);\n"
"    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);\n"
"    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);\n"
"    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);\n"
"    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);\n"
"    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);\n"
"    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);\n"
"    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);\n"
"    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);\n"
"    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);\n"
"    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);\n"
"    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);\n"
"    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);\n"
"    *h0 += a; *h1 += b; *h2 += c; *h3 += d;\n"
"}\n"
"\n"
"ulong compact_mix(ulong k) {\n"
"    k ^= k >> 33; k *= 0xff51afd7ed558ccdUL;\n"
"    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53UL;\n"
"    k ^= k >> 33; return k;\n"
"}\n"
"\n"
"int probe_compact(uint hx, uint hy, uint hz, uint hw,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    ulong compact_mask, uint max_probe, uint hash_data_count,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, uint overflow_count)\n"
"{\n"
"    ulong key = ((ulong)hy << 32) | hx;\n"
"    uint fp = (uint)(key >> 32);\n"
"    if (fp == 0) fp = 1;\n"
"    ulong pos = compact_mix(key) & compact_mask;\n"
"    for (int p = 0; p < (int)max_probe; p++) {\n"
"        uint cfp = compact_fp[pos];\n"
"        if (cfp == 0) break;\n"
"        if (cfp == fp) {\n"
"            uint idx = compact_idx[pos];\n"
"            if (idx < hash_data_count) {\n"
"                ulong off = hash_data_off[idx];\n"
"                __global const uint *ref = (__global const uint *)(hash_data_buf + off);\n"
"                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3])\n"
"                    return 1;\n"
"            }\n"
"        }\n"
"        pos = (pos + 1) & compact_mask;\n"
"    }\n"
"    if (overflow_count > 0) {\n"
"        int lo = 0, hi = (int)overflow_count - 1;\n"
"        while (lo <= hi) {\n"
"            int mid = (lo + hi) / 2;\n"
"            ulong mkey = overflow_keys[mid];\n"
"            if (key < mkey) hi = mid - 1;\n"
"            else if (key > mkey) lo = mid + 1;\n"
"            else {\n"
"                uint ooff = overflow_offsets[mid];\n"
"                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);\n"
"                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {\n"
"                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);\n"
"                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                }\n"
"                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {\n"
"                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);\n"
"                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                }\n"
"                break;\n"
"            }\n"
"        }\n"
"    }\n"
"    return 0;\n"
"}\n"
"\n"
"__kernel void md5salt_batch(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    uint hoff = word_idx * 256;\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + hoff);\n"
"    for (int i = 0; i < 8; i++) M[i] = mwords[i];\n"
"\n"
"    /* Prestate rounds 0-7 */\n"
"    uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;\n"
"    for (int i = 0; i < 8; i++) {\n"
"        uint f = (b & c) | (~b & d);\n"
"        f = f + a + K[i] + M[i];\n"
"        a = d; d = c; c = b;\n"
"        b = b + rotate(f, S[i]);\n"
"    }\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 32 + slen;\n"
"\n"
"    uint hx, hy, hz, hw;\n"
"    for (int i = 8; i < 16; i++) M[i] = 0;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++)\n"
"        mbytes[32 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"    /* Use full md5_block (prestate optimization deferred to tuning phase) */\n"
"    hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                      params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                      hash_data_buf, hash_data_off,\n"
"                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"        uint slot = atomic_add(hit_count, 1u);\n"
"        if (slot < params.max_hits) {\n"
"            uint base = slot * 6;\n"
"            hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"            hits[base+2] = hx; hits[base+3] = hy;\n"
"            hits[base+4] = hz; hits[base+5] = hw;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__kernel void md5salt_sub8_24(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);\n"
"    for (int i = 0; i < 4; i++) M[i] = mwords[i];\n"
"    for (int i = 4; i < 16; i++) M[i] = 0;\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 16 + slen;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++) mbytes[16 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"\n"
"    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                      params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                      hash_data_buf, hash_data_off,\n"
"                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"        uint slot = atomic_add(hit_count, 1u);\n"
"        if (slot < params.max_hits) {\n"
"            uint base = slot * 6;\n"
"            hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"            hits[base+2] = hx; hits[base+3] = hy;\n"
"            hits[base+4] = hz; hits[base+5] = hw;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__kernel void md5salt_iter(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);\n"
"    for (int i = 0; i < 8; i++) M[i] = mwords[i];\n"
"    for (int i = 8; i < 16; i++) M[i] = 0;\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 32 + slen;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++) mbytes[32 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"\n"
"    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    for (uint iter = 0; iter < params.max_iter; iter++) {\n"
"        if (iter > 0) {\n"
"            uint hwords[4]; hwords[0]=hx; hwords[1]=hy; hwords[2]=hz; hwords[3]=hw;\n"
"            uchar *hb = (uchar *)hwords;\n"
"            uint Mi[16];\n"
"            uchar *mb = (uchar *)Mi;\n"
"            for (int i = 0; i < 16; i++) {\n"
"                uchar hi = hb[i] >> 4, lo = hb[i] & 0xf;\n"
"                mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);\n"
"                mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);\n"
"            }\n"
"            Mi[8] = 0x80; Mi[9]=0; Mi[10]=0; Mi[11]=0;\n"
"            Mi[12]=0; Mi[13]=0; Mi[14]=256; Mi[15]=0;\n"
"            hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;\n"
"            md5_block(&hx, &hy, &hz, &hw, Mi);\n"
"        }\n"
"        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                          params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                          hash_data_buf, hash_data_off,\n"
"                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"            uint slot = atomic_add(hit_count, 1u);\n"
"            if (slot < params.max_hits) {\n"
"                uint base = slot * 7;\n"
"                hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"                hits[base+2] = iter + 1;\n"
"                hits[base+3] = hx; hits[base+4] = hy;\n"
"                hits[base+5] = hz; hits[base+6] = hw;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

/* ---- List GPU devices and exit (called early from option parsing) ---- */
void gpu_opencl_list_devices(void) {
    if (opencl_dynload_init() != 0) {
        fprintf(stderr, "  OpenCL library not found.\n");
        exit(0);
    }
    cl_platform_id plats[8];
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(8, plats, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "  No OpenCL platforms found.\n");
        exit(0);
    }
    int idx = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64, devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;
        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            fprintf(stderr, "  GPU[%d]: %s (%llu MB)\n", idx, dname,
                    (unsigned long long)(gmem / (1024*1024)));
            idx++;
        }
    }
    if (idx == 0) fprintf(stderr, "  No GPU devices found.\n");
    exit(0);
}

/* ---- GPU device filter ---- */
extern int gpu_device_filter_set;
extern int gpu_device_allowed[];

static int device_allowed(int idx) {
    if (!gpu_device_filter_set) return 1;
    if (idx < 0 || idx >= 64) return 0;
    return gpu_device_allowed[idx];
}

/* ---- Helper ---- */
#define OCL_CHECK(call, msg) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d: %s\n", _e, msg); return -1; } } while(0)

static cl_mem dev_buf(struct gpu_device *d, size_t size, cl_mem_flags flags) {
    cl_int err;
    return clCreateBuffer(d->ctx, flags, size ? size : 4, NULL, &err);
}

/* Kernel-to-op mapping table. Each entry: kernel function name + list of ops it serves.
 * Adding a new GPU algorithm = one line here + the kernel in gpu_kernels.cl. */
static const struct {
    const char *name;
    int ops[8];      /* -1 terminated */
} kernel_map[] = {
    {"md5salt_batch",       {JOB_MD5SALT, JOB_MD5UCSALT, JOB_MD5revMD5SALT, -1}},
    {"md5salt_sub8_24",     {JOB_MD5sub8_24SALT, -1}},
    {"md5salt_iter",        {-1}},  /* special: salted iteration override */
    {"md5saltpass_batch",   {JOB_MD5SALTPASS, -1}},
    {"md5passsalt_batch",   {JOB_MD5PASSSALT, -1}},
    {"md5_iter_lc",         {JOB_MD5, -1}},
    {"md5_iter_uc",         {JOB_MD5UC, -1}},
    {"sha256passsalt_batch", {JOB_SHA256PASSSALT, -1}},
    {"sha256saltpass_batch", {JOB_SHA256SALTPASS, -1}},
    {"md5_md5saltmd5pass_batch", {JOB_MD5_MD5SALTMD5PASS, -1}},
    {"md5crypt_batch",       {JOB_MD5CRYPT, -1}},
    {NULL, {-1}}
};
#define KERN_ITER_IDX 2  /* index of md5salt_iter in kernel_map (for Maxiter override) */

/* OpenCL async error callback — called by the driver on compute errors */
static void CL_CALLBACK ocl_error_callback(const char *errinfo,
    const void *private_info, size_t cb, void *user_data) {
    fprintf(stderr, "OpenCL GPU ASYNC ERROR: %s\n", errinfo);
}

/* Initialize one GPU device */
static int init_device(int di, cl_device_id dev_id, const char *kernel_source) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    d->dev = dev_id;
    clGetDeviceInfo(dev_id, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);

    d->ctx = clCreateContext(NULL, 1, &dev_id, ocl_error_callback, NULL, &err);
    if (!d->ctx) return -1;

    d->queue = clCreateCommandQueue(d->ctx, dev_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!d->queue) return -1;

    d->prog = clCreateProgramWithSource(d->ctx, 1, &kernel_source, NULL, &err);
    if (!d->prog) return -1;

    const char *build_opts = NULL;
    fprintf(stderr, "OpenCL GPU[%d]: kernel build\n", di);
    err = clBuildProgram(d->prog, 1, &dev_id, build_opts, NULL, NULL);
    { char log[4096]; size_t loglen = 0;
      clGetProgramBuildInfo(d->prog, dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &loglen);
      if (err != CL_SUCCESS) {
          fprintf(stderr, "OpenCL GPU[%d] kernel compile error:\n%s\n", di, log);
          return -1;
      }
      /* Only show build log on error — warnings are suppressed */
    }

    /* Create and register all kernels from the kernel_map table */
    memset(&dev_kerns[di], 0, sizeof(dev_kerns[di]));
    d->kern_salt_iter = NULL;
    for (int k = 0; kernel_map[k].name; k++) {
        cl_kernel kern = clCreateKernel(d->prog, kernel_map[k].name, &err);
        if (!kern) continue;
        if (k == KERN_ITER_IDX)
            d->kern_salt_iter = kern;  /* save salted iteration kernel */
        for (int j = 0; kernel_map[k].ops[j] >= 0; j++)
            kern_register(di, kernel_map[k].ops[j], kern);
    }

    /* Scale batch size to GPU capability */
    cl_uint compute_units = 0;
    clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    if (compute_units < 8)
        d->max_batch = compute_units * 8;   /* tiny GPU: 4 CU -> 32 words */
    else if (compute_units < 32)
        d->max_batch = compute_units * 16;  /* mid GPU: 16 CU -> 256 words */
    else
        d->max_batch = GPUBATCH_MAX;        /* big GPU: full 512 */
    if (d->max_batch < 16) d->max_batch = 16;
    if (d->max_batch > GPUBATCH_MAX) d->max_batch = GPUBATCH_MAX;
    fprintf(stderr, "OpenCL GPU[%d]: %u compute units, batch size %d\n",
            di, compute_units, d->max_batch);

    /* Per-device dispatch buffers */
    d->b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               GPU_MAX_HITS * 11 * sizeof(uint32_t), NULL, &err);
    d->b_hit_count = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t), NULL, &err);
    d->h_hits = (uint32_t *)malloc(GPU_MAX_HITS * 11 * sizeof(uint32_t));
    d->b_params = dev_buf(d, sizeof(OCLParams), CL_MEM_READ_ONLY);

    /* Dummy overflow buffers (replaced when overflow is loaded) */
    d->b_overflow_keys = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_hashes = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_offsets = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_lengths = dev_buf(d, 4, CL_MEM_READ_ONLY);

    d->salt_data_cap = 0;
    d->salts_count = 0;
    d->hexhash_cap = 0;

    return 0;
}

/* ---- Public API ---- */

/* Probe maximum reliable dispatch size for a GPU device.
 * Uses the actual md5salt_batch kernel with synthetic test data at increasing
 * power-of-2 salt counts. Each test uses 1 word with a known MD5SALT result;
 * the number of salts scales up. When the GPU misses the known hit, we've
 * found the dispatch limit.
 * Returns 0 if all sizes pass (no limit needed). */
static int probe_max_dispatch(int di) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    cl_kernel test_kern = clCreateKernel(d->prog, "md5salt_batch", &err);
    if (!test_kern) {
        fprintf(stderr, "OpenCL GPU[%d]: probe kernel not found (err=%d)\n", di, err);
        return 0;
    }

    /* Test word: "a" → hex = "61" (2 hex chars) padded to 256 bytes.
     * Test salt at index 0: "x" (1 byte).
     * MD5("61" + "x") = MD5("61x") — we'll compute this on the CPU to get the expected hash,
     * then build a tiny 2-slot compact table containing just this one hash.
     * The test: dispatch with 1 word and N salts (salt 0 = "x", salts 1..N-1 = "y").
     * If the GPU correctly processes all N work items, it finds exactly 1 hit (salt 0).
     * If the dispatch is too large and corrupts computation, it finds 0 hits. */

    /* Kernel computes MD5(hexhash[0..31] + salt) where hexhash = "61" + 30 NULs, salt = "x"
     * = MD5("61\0\0...\0x") (33 bytes) = 1d7f6bd1126bd87f5f5a4cdb6502cc5e */
    uint32_t exp_hx = 0xd16b7f1du, exp_hy = 0x7fd86b12u;

    /* Build a 4-slot compact table (mask=3) with just this one hash */
    uint32_t compact_fp[4] = {0, 0, 0, 0};
    uint32_t compact_idx[4] = {0, 0, 0, 0};
    uint32_t fp = exp_hy;
    if (fp == 0) fp = 1;
    uint32_t pos = (exp_hx ^ exp_hy) & 3;
    compact_fp[pos] = fp;
    compact_idx[pos] = 0;

    /* hash_data: 16-byte hash at offset 0 */
    uint32_t hash_data[4] = { exp_hx, exp_hy, 0xdb4c5a5fu, 0x5ecc0265u };
    uint64_t hash_off = 0;
    uint16_t hash_len = 16;

    /* Word: "61" in hex, padded to 256 bytes. First 2 bytes = 0x36,0x31 = "61" */
    /* 32 copies of the same word "61" in the hexhash buffer */
    unsigned char hexwords[32 * 256];
    uint16_t hexlens[32];
    memset(hexwords, 0, sizeof(hexwords));
    for (int w = 0; w < 32; w++) {
        hexwords[w * 256] = '6';
        hexwords[w * 256 + 1] = '1';
        hexlens[w] = 2;
    }

    /* Create GPU buffers for the test */
    cl_mem b_hexhash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexwords), hexwords, &err);
    cl_mem b_hexlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexlens), hexlens, &err);
    cl_mem b_cfp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_fp), compact_fp, &err);
    cl_mem b_cidx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_idx), compact_idx, &err);
    cl_mem b_hdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_data), hash_data, &err);
    cl_mem b_hoff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_off), &hash_off, &err);
    cl_mem b_hlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_len), &hash_len, &err);

    /* Params buffer */
    OCLParams params;
    memset(&params, 0, sizeof(params));
    params.compact_mask = 3;
    params.num_words = 32;
    params.salt_start = 0;
    params.max_probe = 4;
    params.hash_data_count = 1;
    params.max_hits = 256;
    params.overflow_count = 0;
    params.max_iter = 1;
    cl_mem b_params = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, sizeof(params), NULL, &err);

    /* Hits */
    cl_mem b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, 256 * 6 * sizeof(uint32_t), NULL, &err);
    cl_mem b_hitcnt = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &err);

    /* Dummy overflow buffers (empty) */
    uint64_t dummy64 = 0;
    uint32_t dummy32 = 0;
    uint16_t dummy16 = 0;
    cl_mem b_okeys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8, &dummy64, &err);
    cl_mem b_ohash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_ooff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_olen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2, &dummy16, &err);

    /* Build salt buffer: salt[0] = "x", salt[1..max] = "y" */
    int max_salts = 4194304;
    char *salt_data = (char *)malloc(max_salts);
    uint32_t *salt_off = (uint32_t *)malloc(max_salts * sizeof(uint32_t));
    uint16_t *salt_len = (uint16_t *)malloc(max_salts * sizeof(uint16_t));
    salt_data[0] = 'x';
    for (int i = 1; i < max_salts; i++) salt_data[i] = 'y';
    for (int i = 0; i < max_salts; i++) { salt_off[i] = i; salt_len[i] = 1; }

    cl_mem b_sdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts, NULL, &err);
    cl_mem b_soff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint32_t), NULL, &err);
    cl_mem b_slen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint16_t), NULL, &err);
    clEnqueueWriteBuffer(d->queue, b_sdata, CL_TRUE, 0, max_salts, salt_data, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_soff, CL_TRUE, 0, max_salts * sizeof(uint32_t), salt_off, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_slen, CL_TRUE, 0, max_salts * sizeof(uint16_t), salt_len, 0, NULL, NULL);
    clFinish(d->queue);
    free(salt_data); free(salt_off); free(salt_len);

    /* Set kernel args */
    int a = 0;
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexhash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_sdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_soff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_slen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cfp);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cidx);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_params);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hoff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hits);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hitcnt);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_okeys);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ohash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ooff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_olen);

    /* Test at increasing power-of-2 salt counts */
    static const int test_sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 0
    };

    int max_good = 0;
    for (int t = 0; test_sizes[t]; t++) {
        int nsalts = test_sizes[t];

        params.num_salts = nsalts;
        params.salt_start = 0;
        clEnqueueWriteBuffer(d->queue, b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);

        /* Zero hit counter */
        uint32_t zero = 0;
        clEnqueueWriteBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
        clFinish(d->queue);

        size_t global = (size_t)32 * nsalts;  /* 32 words * nsalts */
        size_t local = 128;
        global = ((global + local - 1) / local) * local;

        err = clEnqueueNDRangeKernel(d->queue, test_kern, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d]: probe dispatch error %d at %d salts\n", di, err, nsalts);
            break;
        }
        cl_int finish_err = clFinish(d->queue);

        /* Read hit count — should be exactly 32 (salt 0 matches for each of 32 words) */
        uint32_t nhits;
        clEnqueueReadBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(nhits), &nhits, 0, NULL, NULL);

        if (nhits == 32 && finish_err == CL_SUCCESS) {
            max_good = nsalts;
        } else {
            fprintf(stderr, "OpenCL GPU[%d]: probe FAIL at %d salts (%d items)"
                    " — got %u hits, expected 32, clFinish=%d\n",
                    di, nsalts, (int)global, nhits, finish_err);
            break;
        }
    }

    /* Cleanup */
    clReleaseKernel(test_kern);
    clReleaseMemObject(b_hexhash); clReleaseMemObject(b_hexlen);
    clReleaseMemObject(b_cfp); clReleaseMemObject(b_cidx);
    clReleaseMemObject(b_hdata); clReleaseMemObject(b_hoff); clReleaseMemObject(b_hlen);
    clReleaseMemObject(b_params); clReleaseMemObject(b_hits); clReleaseMemObject(b_hitcnt);
    clReleaseMemObject(b_sdata); clReleaseMemObject(b_soff); clReleaseMemObject(b_slen);
    clReleaseMemObject(b_okeys); clReleaseMemObject(b_ohash); clReleaseMemObject(b_ooff);
    clReleaseMemObject(b_olen);

    int last_tested = 0;
    for (int t = 0; test_sizes[t]; t++) last_tested = test_sizes[t];
    if (max_good >= last_tested)
        return 0;  /* all passed — no dispatch limit */
    return max_good;
}

int gpu_opencl_init(void) {
    if (opencl_dynload_init() != 0) return -1;

    cl_uint nplat = 0;
    cl_platform_id plats[8];
    cl_int err;

    clGetPlatformIDs(8, plats, &nplat);
    if (nplat == 0) { fprintf(stderr, "OpenCL: no platforms\n"); return -1; }

    /* Load kernel source */
#ifdef DEBUG
    const char *kernel_source = load_kernel_file("gpu_kernels.cl");
    if (kernel_source) {
        fprintf(stderr, "OpenCL GPU: loaded kernel from gpu_kernels.cl\n");
    } else {
        kernel_source = gpu_kernels_str;
        fprintf(stderr, "OpenCL GPU: using embedded kernel\n");
    }
#else
    const char *kernel_source = gpu_kernels_str;
#endif

    /* Enumerate all GPU devices across all platforms.
     * -G 0,2,4 or -G 0-2: select specific devices.
     * (-G list is handled earlier in main via gpu_opencl_list_devices) */
    int all_dev_idx = 0;

    num_gpu_devs = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64,
                             devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;

        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);

            if (!device_allowed(all_dev_idx)) {
                fprintf(stderr, "OpenCL GPU[%d]: %s - skipped\n", all_dev_idx, dname);
                all_dev_idx++;
                continue;
            }

            if (num_gpu_devs < MAX_GPU_DEVICES) {
                if (init_device(num_gpu_devs, devs[d], kernel_source) == 0) {
                    fprintf(stderr, "OpenCL GPU[%d]: %s (%llu MB)\n", all_dev_idx, dname,
                            (unsigned long long)(gmem / (1024*1024)));
                    int max_disp = probe_max_dispatch(num_gpu_devs);
                    gpu_devs[num_gpu_devs].max_dispatch = max_disp;
                    if (max_disp == 0)
                        fprintf(stderr, "OpenCL GPU[%d]: selftest passed all sizes (no dispatch limit)\n", all_dev_idx);
                    else
                        fprintf(stderr, "OpenCL GPU[%d]: dispatch limit = %d work items\n", all_dev_idx, max_disp);
                    num_gpu_devs++;
                }
            }
            all_dev_idx++;
        }
    }

    if (num_gpu_devs == 0) {
        fprintf(stderr, "OpenCL: no GPU devices found\n");
        return -1;
    }

    fprintf(stderr, "OpenCL GPU: %d device%s initialized\n",
            num_gpu_devs, num_gpu_devs > 1 ? "s" : "");
    ocl_ready = 1;
    return 0;
}

void gpu_opencl_shutdown(void) {
    if (!ocl_ready) return;
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->b_compact_fp) clReleaseMemObject(d->b_compact_fp);
        if (d->b_compact_idx) clReleaseMemObject(d->b_compact_idx);
        if (d->b_hash_data) clReleaseMemObject(d->b_hash_data);
        if (d->b_hash_data_off) clReleaseMemObject(d->b_hash_data_off);
        if (d->b_hash_data_len) clReleaseMemObject(d->b_hash_data_len);
        if (d->b_salt_data) clReleaseMemObject(d->b_salt_data);
        if (d->b_salt_off) clReleaseMemObject(d->b_salt_off);
        if (d->b_salt_len) clReleaseMemObject(d->b_salt_len);
        if (d->b_hits) clReleaseMemObject(d->b_hits);
        if (d->b_hit_count) clReleaseMemObject(d->b_hit_count);
        if (d->b_params) clReleaseMemObject(d->b_params);
        if (d->b_overflow_keys) clReleaseMemObject(d->b_overflow_keys);
        if (d->b_overflow_hashes) clReleaseMemObject(d->b_overflow_hashes);
        if (d->b_overflow_offsets) clReleaseMemObject(d->b_overflow_offsets);
        if (d->b_overflow_lengths) clReleaseMemObject(d->b_overflow_lengths);
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        /* Release all registered kernels */
        for (int k = 0; k < MAX_GPU_KERNELS; k++)
            if (dev_kerns[i].kerns[k].kernel) clReleaseKernel(dev_kerns[i].kerns[k].kernel);
        if (d->kern_salt_iter) clReleaseKernel(d->kern_salt_iter);
        if (d->prog) clReleaseProgram(d->prog);
        if (d->queue) clReleaseCommandQueue(d->queue);
        if (d->ctx) clReleaseContext(d->ctx);
        free(d->h_hits);
    }
    ocl_ready = 0;
}

int gpu_opencl_available(void) { return ocl_ready; }
int gpu_opencl_num_devices(void) { return num_gpu_devs; }
int gpu_opencl_max_batch(int dev_idx) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return GPUBATCH_MAX;
    return gpu_devs[dev_idx].max_batch;
}

int gpu_opencl_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;

    /* Check if GPU has enough memory for the compact table + hash data */
    size_t needed = compact_size * sizeof(uint32_t) * 2   /* compact_fp + compact_idx */
                  + hash_data_buf_size                      /* hash_data_buf */
                  + hash_data_count * sizeof(uint64_t)      /* hash_data_off */
                  + hash_data_count * sizeof(uint16_t)      /* hash_data_len */
                  + GPU_MAX_HITS * 11 * sizeof(uint32_t)     /* hits buffer */
                  + 512 * 256;                              /* hexhash buffer */
    cl_ulong gpu_mem = 0;
    clGetDeviceInfo(d->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gpu_mem), &gpu_mem, NULL);
    if (needed > (size_t)(gpu_mem * 0.8)) {
        fprintf(stderr, "OpenCL GPU[%d]: insufficient memory (%zuMB needed, %lluMB available) - skipping\n",
                dev_idx, needed / (1024*1024), (unsigned long long)(gpu_mem / (1024*1024)));
        return -1;
    }

    _compact_mask = compact_mask;
    _hash_data_count = hash_data_count;

    /* Allocate GPU buffers then upload in chunks */
#define GPU_CHUNK (1024 * 1024)  /* 1MB upload chunks */
    d->b_compact_fp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                     compact_size * sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_compact_idx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                      compact_size * sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_hash_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                    hash_data_buf_size, NULL, &err);
    if (err != CL_SUCCESS) goto alloc_fail;

    uint64_t *off64 = (uint64_t *)malloc(hash_data_count * sizeof(uint64_t));
    for (size_t i = 0; i < hash_data_count; i++) off64[i] = hash_data_off[i];
    d->b_hash_data_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                        hash_data_count * sizeof(uint64_t), NULL, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    d->b_hash_data_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                        hash_data_count * sizeof(uint16_t), NULL, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    /* Chunked upload — flush after each chunk for driver reliability */
    { struct { cl_mem buf; const void *src; size_t size; } uploads[] = {
        { d->b_compact_fp,    compact_fp,    compact_size * sizeof(uint32_t) },
        { d->b_compact_idx,   compact_idx,   compact_size * sizeof(uint32_t) },
        { d->b_hash_data,     hash_data_buf, hash_data_buf_size },
        { d->b_hash_data_off, off64,         hash_data_count * sizeof(uint64_t) },
        { d->b_hash_data_len, hash_data_len, hash_data_count * sizeof(uint16_t) },
        { 0, NULL, 0 }
      };
      for (int u = 0; uploads[u].buf; u++) {
        size_t off = 0;
        while (off < uploads[u].size) {
            size_t chunk = uploads[u].size - off;
            if (chunk > GPU_CHUNK) chunk = GPU_CHUNK;
            cl_int werr = clEnqueueWriteBuffer(d->queue, uploads[u].buf, CL_TRUE, off,
                                 chunk, (const char *)uploads[u].src + off, 0, NULL, NULL);
            if (werr != CL_SUCCESS)
                fprintf(stderr, "OpenCL GPU[%d]: upload[%d] chunk at off=%zu size=%zu failed: %d\n",
                        dev_idx, u, off, chunk, werr);
            off += chunk;
        }
        clFinish(d->queue);
      }
    }
    free(off64);

    fprintf(stderr, "OpenCL GPU[%d]: compact table registered (%llu slots, %u hashes, %zuMB)\n",
            dev_idx, (unsigned long long)compact_size, (unsigned)hash_data_count,
            needed / (1024*1024));
    return 0;

alloc_fail:
    fprintf(stderr, "OpenCL GPU[%d]: buffer allocation failed (err=%d) - skipping\n", dev_idx, err);
    if (d->b_compact_fp) { clReleaseMemObject(d->b_compact_fp); d->b_compact_fp = NULL; }
    if (d->b_compact_idx) { clReleaseMemObject(d->b_compact_idx); d->b_compact_idx = NULL; }
    if (d->b_hash_data) { clReleaseMemObject(d->b_hash_data); d->b_hash_data = NULL; }
    if (d->b_hash_data_off) { clReleaseMemObject(d->b_hash_data_off); d->b_hash_data_off = NULL; }
    if (d->b_hash_data_len) { clReleaseMemObject(d->b_hash_data_len); d->b_hash_data_len = NULL; }
    return -1;
}

int gpu_opencl_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_salts <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    size_t salts_size = salt_offsets[num_salts - 1] + salt_lens[num_salts - 1];

    if (salts_size > d->salt_data_cap) {
        if (d->b_salt_data) clReleaseMemObject(d->b_salt_data);
        d->b_salt_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, salts_size + 4096, NULL, &err);
        d->salt_data_cap = salts_size + 4096;
    }
    if (d->b_salt_off) clReleaseMemObject(d->b_salt_off);
    if (d->b_salt_len) clReleaseMemObject(d->b_salt_len);
    d->b_salt_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_salts * sizeof(uint32_t), NULL, &err);
    d->b_salt_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_salts * sizeof(uint16_t), NULL, &err);

    clEnqueueWriteBuffer(d->queue, d->b_salt_data, CL_TRUE, 0, salts_size, salts, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_off, CL_TRUE, 0, num_salts * sizeof(uint32_t), salt_offsets, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_len, CL_TRUE, 0, num_salts * sizeof(uint16_t), salt_lens, 0, NULL, NULL);
    clFinish(d->queue);
    d->salts_count = num_salts;
    return 0;
}

int gpu_opencl_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || count <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    size_t total = offsets[count - 1] + lengths[count - 1];

    clReleaseMemObject(d->b_overflow_keys);
    clReleaseMemObject(d->b_overflow_hashes);
    clReleaseMemObject(d->b_overflow_offsets);
    clReleaseMemObject(d->b_overflow_lengths);

    d->b_overflow_keys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        count * sizeof(uint64_t), (void*)keys, &err);
    d->b_overflow_hashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          total, (void*)hashes, &err);
    d->b_overflow_offsets = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           count * sizeof(uint32_t), (void*)offsets, &err);
    d->b_overflow_lengths = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            count * sizeof(uint16_t), (void*)lengths, &err);
    _overflow_count = count;
    fprintf(stderr, "OpenCL GPU[%d]: %d overflow entries loaded\n", dev_idx, count);
    return 0;
}

void gpu_opencl_set_max_iter(int max_iter) { _max_iter = (max_iter < 1) ? 1 : max_iter; }
void gpu_opencl_set_op(int op) { _gpu_op = op; }

uint32_t *gpu_opencl_dispatch_batch(int dev_idx,
    const char *hexhashes, const uint16_t *hexlens,
    int num_words, int *nhits_out)
{
    *nhits_out = 0;
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_words <= 0) return NULL;
    struct gpu_device *d = &gpu_devs[dev_idx];
    if (!d->b_compact_fp) return NULL; /* compact table not loaded (insufficient memory) */
    /* Look up kernel for current op */
    int op = _gpu_op;
    int cat = gpu_op_category(op);
    struct gpu_kern *gk = (op >= 0 && op < MAX_GPU_KERNELS) ? &dev_kerns[dev_idx].kerns[op] : NULL;
    if (!gk || !gk->kernel) return NULL;

    /* Salted types need salts uploaded; iteration types do not */
    if ((cat == GPU_CAT_SALTED || cat == GPU_CAT_SALTPASS) && d->salts_count <= 0) return NULL;

    /* For salted iteration, use the salt-iter kernel; otherwise use the registered kernel */
    cl_kernel kern = (cat == GPU_CAT_SALTED && _max_iter > 1 && d->kern_salt_iter) ? d->kern_salt_iter : gk->kernel;
    cl_int err;

    /* Upload words */
    size_t words_size = (size_t)num_words * 256;
    if (words_size > d->hexhash_cap) {
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        d->b_hexhashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, words_size, NULL, &err);
        d->b_hexlens = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_words * sizeof(uint16_t), NULL, &err);
        d->hexhash_cap = words_size;
    }
    clEnqueueWriteBuffer(d->queue, d->b_hexhashes, CL_TRUE, 0, words_size, hexhashes, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_hexlens, CL_TRUE, 0, num_words * sizeof(uint16_t), hexlens, 0, NULL, NULL);

    /* Params — base values, salt_start/num_salts updated per chunk for salted types */
    OCLParams params;
    params.compact_mask = _compact_mask;
    params.num_words = num_words;
    params.num_salts = d->salts_count;
    params.salt_start = 0;
    params.max_probe = 256;
    params.hash_data_count = _hash_data_count;
    params.max_hits = GPU_MAX_HITS;
    params.overflow_count = _overflow_count;
    params.max_iter = (cat == GPU_CAT_ITER) ? (_max_iter - 1) : _max_iter;

    /* Zero hit counter */
    uint32_t zero = 0;
    if (p_clEnqueueFillBuffer)
        clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero, sizeof(zero), 0, sizeof(zero), 0, NULL, NULL);
    else
        clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
    clFinish(d->queue);

    /* Set kernel args */
    {
        int a = 0;
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexhashes);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexlens);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_data);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_off);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_len);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_compact_fp);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_compact_idx);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_params);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data_off);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data_len);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hits);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hit_count);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_keys);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_hashes);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_offsets);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_lengths);
    }

    /* Dispatch — chunk salts in power-of-2 blocks for GPUs with limited work capacity */
    size_t local = kern_get_local_size(gk);
    int is_salted = (cat == GPU_CAT_SALTED || cat == GPU_CAT_SALTPASS);
    int total_salts = is_salted ? d->salts_count : 0;

    /* Chunk salts to stay within the device's max reliable dispatch size.
     * max_dispatch is determined by probe_max_dispatch() at init time.
     * If 0, the device passed all tests — no chunking needed. */
    int salt_chunk = total_salts;
    if (d->max_dispatch > 0 && is_salted && num_words > 0) {
        salt_chunk = d->max_dispatch / num_words;
        if (salt_chunk < 1024) salt_chunk = 1024;
    }
    int num_chunks = is_salted ? (total_salts + salt_chunk - 1) / salt_chunk : 1;

    struct timespec t0, t1;
    if (!gk->tuned) clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        if (is_salted) {
            params.salt_start = chunk * salt_chunk;
            params.num_salts = total_salts - params.salt_start;
            if (params.num_salts > salt_chunk) params.num_salts = salt_chunk;
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        } else if (chunk == 0) {
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        }

        size_t global = is_salted
            ? (size_t)num_words * params.num_salts
            : (size_t)num_words;
        global = ((global + local - 1) / local) * local;

        err = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d] dispatch error: %d (chunk %d/%d global=%zu)\n",
                    dev_idx, err, chunk, num_chunks, global);
            return NULL;
        }
        clFinish(d->queue);
    }

    if (!gk->tuned) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        kern_record_time(gk, ms);
    }

    /* Read back */
    uint32_t raw_nhits;
    clEnqueueReadBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(raw_nhits), &raw_nhits, 0, NULL, NULL);

    *nhits_out = (int)raw_nhits;
    if (raw_nhits > 0) {
        if (raw_nhits > GPU_MAX_HITS) raw_nhits = GPU_MAX_HITS;
        int is_sha256 = (_gpu_op == JOB_SHA256PASSSALT || _gpu_op == JOB_SHA256SALTPASS);
        int stride = is_sha256 ? 11 : (_max_iter > 1) ? 7 : 6;
        clEnqueueReadBuffer(d->queue, d->b_hits, CL_TRUE, 0, raw_nhits * stride * sizeof(uint32_t), d->h_hits, 0, NULL, NULL);
    }

    return d->h_hits;
}

#endif /* OPENCL_GPU */
