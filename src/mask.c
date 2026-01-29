#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <omp.h>
#include <errno.h>   // for strerror(errno)
#include <unistd.h>  // for fsync

#include "mask.h"
#include "identification.h" // for IdentNSigmaMasks definition

// Write binary mask (0/1) as grayscale PNG (0 or 1 values)
void writeIndexMaskPNG(const bool *mask, int nsamp, int nchan, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "[Error] fopen('%s') failed: %s\n", filename, strerror(errno));
        return;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return; }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_write_struct(&png_ptr, (png_infopp)NULL); fclose(fp); return; }

    if (setjmp(png_jmpbuf(png_ptr))) { png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return; }

    png_init_io(png_ptr, fp);
    png_set_IHDR(
        png_ptr,
        info_ptr,
        nsamp,
        nchan,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_set_gAMA(png_ptr, info_ptr, 1.0);
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * nchan);
    int i, j;
    for (i = 0; i < nchan; i++)
    {
        row_pointers[nchan - 1 - i] = (png_bytep)malloc(nsamp);
        for (j = 0; j < nsamp; j++)
        {
            int val = mask[i * nsamp + j] ? 1 : 0;
            row_pointers[nchan - 1 - i][j] = (png_byte)val;  // 直接使用类别编号作为像素值
        }
    }

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    for (i = 0; i < nchan; i++) free(row_pointers[i]);
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fflush(fp);
    int fd = fileno(fp);
    if (fd >= 0) fsync(fd);
    fclose(fp);
}

// Write class-index mask (0..255) as 8-bit grayscale PNG
void writeClassIndexMaskPNG(const int *indexMask, int nsamp, int nchan, char *filename)
{
    if (!indexMask || nsamp <= 0 || nchan <= 0 || !filename) return;
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "[Error] fopen('%s') failed: %s\n", filename, strerror(errno));
        return;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return; }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_write_struct(&png_ptr, (png_infopp)NULL); fclose(fp); return; }
    if (setjmp(png_jmpbuf(png_ptr))) { png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return; }

    png_init_io(png_ptr, fp);
    png_set_IHDR(
        png_ptr,
        info_ptr,
        nsamp,
        nchan,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_set_gAMA(png_ptr, info_ptr, 1.0);
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * nchan);
    if (!row_pointers) { png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return; }
    for (int i = 0; i < nchan; i++)
    {
        row_pointers[nchan - 1 - i] = (png_bytep)malloc(nsamp);
        if (!row_pointers[nchan - 1 - i]) {
            for (int k = 0; k < i; ++k) free(row_pointers[nchan - 1 - k]);
            free(row_pointers);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            return;
        }
        for (int j = 0; j < nsamp; j++)
        {
            int idx = indexMask[i * nsamp + j];
            if (idx < 0) idx = 0;
            if (idx > 255) idx = 255;
            row_pointers[nchan - 1 - i][j] = (png_byte)idx;
        }
    }

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    for (int i = 0; i < nchan; i++) free(row_pointers[i]);
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fflush(fp);
    int fd = fileno(fp);
    if (fd >= 0) fsync(fd);
    fclose(fp);
}

void mergeMask2D(int *masks[], int nmasks, int nsamp, int nchan, int *result)
{
    int i, j;
    for (i = 0; i < nmasks; i++)
    {
        for (j = 0; j < nsamp * nchan; j++)
        {
            if (masks[i][j] == 1)
            {
                result[j] = i + 1;
            }
        }
    }
}

void expandChannelMask(const int *channelFlagged, bool *mask2D, int nsamp, int nchan)
{
    // #pragma omp parallel for
    for (int i = 0; i < nchan; i++) {
        if (channelFlagged[i]) {
            int base = i * nsamp;
            for (int j = 0; j < nsamp; j++) {
                mask2D[base + j] = 1;
            }
        }
    }
}

void logicalOR(bool *restrict globalMask, const bool *restrict mask, int nsamp, int nchan)
{
    if (!globalMask || !mask) return;
    int total = nsamp * nchan;
    // #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        if (mask[idx]) globalMask[idx] = true;
    }
}

void writeAllMasksPNG(const IdentNSigmaMasks *masks, int nsamp, int nchan,
                      const char *datasetPath, int index, int merge, const char *sourceName)
{
    if (!masks || !datasetPath) return;
    char filename[512];
    if (merge) {
        int total = nsamp * nchan;
        int *indexMask = (int *)calloc(total, sizeof(int));
        if (!indexMask) return;

        /*
         * 优先级逻辑调整：
         * 1. Point/Vertical/Block (较低优先级): 被脉冲覆盖以保留脉冲形态。
         * 2. Pulsar (中等优先级): 覆盖点状和块状干扰，但会被横向干扰覆盖。
         * 3. Horizontal (最高优先级): 覆盖脉冲，确保彻底消除通道干扰。
         *
         * 写入顺序决定覆盖（后写的盖住先写的）:
         */

        // 1. 第一优先级（最低）：写入各类点状、垂直、块状干扰
        // 这些干扰将被脉冲覆盖，以保护脉冲形态不被切割为碎片
        if (masks->verticalMask) {
            for (int i = 0; i < total; i++) if (masks->verticalMask[i]) indexMask[i] = 2;
        }
        if (masks->pointMask) {
            for (int i = 0; i < total; i++) if (masks->pointMask[i]) indexMask[i] = 6;
        }
        if (masks->periodicMask) {
            for (int i = 0; i < total; i++) if (masks->periodicMask[i]) indexMask[i] = 6; // 映射到 point
        }
        if (masks->blockMask) {
            for (int i = 0; i < total; i++) if (masks->blockMask[i]) indexMask[i] = 7;
        }

        // 2. 第二优先级：写入脉冲
        // 脉冲会覆盖 point/vertical/block，确保脉冲区域的标签是连续的 8
        if (masks->pulseMask) {
            for (int i = 0; i < total; i++) if (masks->pulseMask[i]) indexMask[i] = 8;
        }

        // 3. 第三优先级（最高）：写入横向干扰 (Horizontal)
        // 横向干扰会覆盖脉冲，确保强通道干扰被彻底标记并行清空，防止假阳性
        if (masks->horizontalMask) {
            for (int i = 0; i < total; i++) if (masks->horizontalMask[i]) indexMask[i] = 1;
        }

        int _n = snprintf(filename, sizeof(filename), "%s%s_block%d.png", datasetPath, sourceName, index);
        if (_n < 0) {
            fprintf(stderr, "[Error] snprintf failed building merged mask filename\n");
        } else if ((size_t)_n >= sizeof(filename)) {
            fprintf(stderr, "[Warn] merged mask filename truncated (len=%d, cap=%zu): '%s%s_block%d.png'\n", _n, sizeof(filename), datasetPath, sourceName, index);
        }
        writeClassIndexMaskPNG(indexMask, nsamp, nchan, filename);
        free(indexMask);
        return; // merged 模式下直接返回
    }

    // horizontal
    if (masks->horizontalMask) {
        int _n = snprintf(filename, sizeof(filename), "%s%s_block%d_horizontal.png", datasetPath, sourceName, index);
        if (_n < 0) {
            fprintf(stderr, "[Error] snprintf failed building horizontal mask filename\n");
        } else if ((size_t)_n >= sizeof(filename)) {
            fprintf(stderr, "[Warn] horizontal mask filename truncated (len=%d, cap=%zu)\n", _n, sizeof(filename));
        }
        writeIndexMaskPNG(masks->horizontalMask, nsamp, nchan, filename);
    }

    // // vertical
    // if (masks->verticalMask) {
    //     snprintf(filename, sizeof(filename), "%smask_vertical_%d.png", datasetPath, index);
    //     writeIndexMaskPNG(masks->verticalMask, nsamp, nchan, filename);
    // }

    // // global
    // if (masks->globalMask) {
    //     snprintf(filename, sizeof(filename), "%smask_global_%d.png", datasetPath, index);
    //     writeIndexMaskPNG(masks->globalMask, nsamp, nchan, filename);
    // }

    // block
    if (masks->blockMask) {
        int _n = snprintf(filename, sizeof(filename), "%s%s_block%d_blockRFI.png", datasetPath, sourceName, index);
        if (_n < 0) {
             fprintf(stderr, "[Error] snprintf failed building block mask filename\n");
        } else if ((size_t)_n >= sizeof(filename)) {
             fprintf(stderr, "[Warn] block mask filename truncated (len=%d, cap=%zu)\n", _n, sizeof(filename));
        }
        writeIndexMaskPNG(masks->blockMask, nsamp, nchan, filename);
    }

    // pulse
    if (masks->pulseMask) {
        int _n = snprintf(filename, sizeof(filename), "%s%s_block%d_pulse.png", datasetPath, sourceName, index);
        if (_n < 0) fprintf(stderr, "Error fmt pulse\n"); 
        writeIndexMaskPNG(masks->pulseMask, nsamp, nchan, filename);
    }
}

void allocIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan) {
    if (!m) return;
    m->horizontalMask = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->verticalMask   = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->blockMask      = (bool *)calloc(nsamp * nchan, sizeof(bool));  // New: allocate blockMask
    m->periodicMask   = (bool *)calloc(nsamp * nchan, sizeof(bool)); // New: allocate periodicMask
    m->pulseMask      = (bool *)calloc(nsamp * nchan, sizeof(bool)); // New: allocate pulseMask
    m->globalMask     = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->pointMask      = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->chanBrightMask = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->chanDarkMask   = (bool *)calloc(nsamp * nchan, sizeof(bool));
    m->chanComplexMask = (bool *)calloc(nsamp * nchan, sizeof(bool));
}

void clearIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan) {
    if (!m) return;
    memset(m->horizontalMask,    0, sizeof(bool)*nsamp*nchan);
    memset(m->verticalMask,      0, sizeof(bool)*nsamp*nchan);
    memset(m->blockMask,         0, sizeof(bool)*nsamp*nchan);  // New: clear blockMask
    memset(m->periodicMask,      0, sizeof(bool)*nsamp*nchan);  // New: clear periodicMask
    memset(m->pulseMask,         0, sizeof(bool)*nsamp*nchan);  // New: clear pulseMask
    memset(m->globalMask,        0, sizeof(bool)*nsamp*nchan);
    memset(m->pointMask,         0, sizeof(bool)*nsamp*nchan);
    memset(m->chanBrightMask,    0, sizeof(bool)*nsamp*nchan);
    memset(m->chanDarkMask,      0, sizeof(bool)*nsamp*nchan);
    memset(m->chanComplexMask,   0, sizeof(bool)*nsamp*nchan);
}

void freeIdentNSigmaMasks(IdentNSigmaMasks *m) {
    if (!m) return;
    free(m->horizontalMask);
    free(m->verticalMask);
    free(m->blockMask);  // New: free blockMask
    free(m->periodicMask); // New: free periodicMask
    free(m->pulseMask);    // New: free pulseMask
    free(m->globalMask);
    free(m->pointMask);
    free(m->chanBrightMask);
    free(m->chanDarkMask);
    free(m->chanComplexMask);
}

// Smooth outChannel mask by removing isolated flagged channels (single or pairs)
void smoothOutChanMask(int *channelFlagged, int nchan, int N) {
    if (!channelFlagged || nchan <= 0 || N <= 0) return;

    int *temp = (int *)malloc(nchan * sizeof(int));
    memcpy(temp, channelFlagged, nchan * sizeof(int));

    // Find and process contiguous flagged blocks
    int i = 0;
    while (i < nchan) {
        if (temp[i] == 1) {
            // Find the end of the block
            int j = i;
            while (j < nchan && temp[j] == 1) j++;
            int block_length = j - i;

            // Check if block is isolated (length <= 2 and no flagged channels in surrounding N)
            int isolated = 1;
            // Check left side: from i - N to i - 1
            for (int k = i - N; k < i; k++) {
                if (k >= 0 && temp[k] == 1) {
                    isolated = 0;
                    break;
                }
            }
            // Check right side: from j to j + N - 1
            for (int k = j; k < j + N; k++) {
                if (k < nchan && temp[k] == 1) {
                    isolated = 0;
                    break;
                }
            }

            if (isolated && block_length <= 2) {
                // Unflag the entire block
                for (int k = i; k < j; k++) {
                    channelFlagged[k] = 0;
                }
            }

            i = j;  // Skip to end of block
        } else {
            i++;
        }
    }

    free(temp);
}
