/*
 * opencl_md5salt.h — OpenCL GPU acceleration for MD5SALT
 *
 * Cross-vendor GPU support via OpenCL runtime.
 * Supports multiple GPU devices.
 */

#ifndef OPENCL_MD5SALT_H
#define OPENCL_MD5SALT_H

#if defined(OPENCL_GPU)

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int opencl_md5salt_init(void);
void opencl_md5salt_shutdown(void);
int opencl_md5salt_available(void);
int opencl_md5salt_num_devices(void);
void opencl_md5salt_list_devices(void);

/* Per-device APIs — dev_idx from 0 to num_devices-1 */
int opencl_md5salt_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len);

int opencl_md5salt_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts);

int opencl_md5salt_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count);

void opencl_md5salt_set_max_iter(int max_iter);
void opencl_md5salt_set_op(int op);

uint32_t *opencl_md5salt_dispatch_batch(int dev_idx,
    const char *hexhashes, const uint16_t *hexlens,
    int num_words, int *nhits_out);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_GPU */
#endif /* OPENCL_MD5SALT_H */
