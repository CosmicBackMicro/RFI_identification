#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <omp.h>

#include "mask.h"
#include "identification.h" // for IdentNSigmaMasks definition

void writeIndexMaskPNG(int *mask, int nsamp, int nchan, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
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
            int val = mask[i * nsamp + j];
            row_pointers[nchan - 1 - i][j] = (png_byte)val;  // 直接使用类别编号作为像素值
        }
    }

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    for (i = 0; i < nchan; i++) free(row_pointers[i]);
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
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

void expandChannelMask(const int *channelFlagged, int *mask2D, int nsamp, int nchan)
{
    #pragma omp parallel for
    for (int i = 0; i < nchan; i++) {
        if (channelFlagged[i]) {
            int base = i * nsamp;
            for (int j = 0; j < nsamp; j++) {
                mask2D[base + j] = 1;
            }
        }
    }
}

void logicalOR(int *globalMask, const int *mask, int nsamp, int nchan)
{
    if (!globalMask || !mask) return;
    int total = nsamp * nchan;
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        if (mask[idx]) globalMask[idx] = 1;
    }
}

void writeAllMasksPNG(const IdentNSigmaMasks *masks, int nsamp, int nchan,
                      const char *datasetPath, int index, int merge)
{
    if (!masks || !datasetPath) return;

    char filename[512];

    if (merge) {
        int total = nsamp * nchan;
        int *indexMask = (int *)calloc(total, sizeof(int));
        if (!indexMask) return;

        /*
         * 合并优先级(低 -> 高): horizontal < complex < dark < bright < point
         * 低优先级先写入获得较小类号, 高优先级后写入覆盖并拥有较大类号。
         * 类号分配: 1:horizontal 2:complex 3:dark 4:bright 5:point
         */
        int classIndex = 0;

        if (masks->horizontalMask) {
            for (int i = 0; i < total; i++) if (masks->horizontalMask[i]) indexMask[i] = classIndex;
        }
        classIndex++;
        if (masks->chanComplexMask) {
            for (int i = 0; i < total; i++) if (masks->chanComplexMask[i]) indexMask[i] = classIndex;
        }
        classIndex++;
        if (masks->chanDarkMask) {
            for (int i = 0; i < total; i++) if (masks->chanDarkMask[i]) indexMask[i] = classIndex;
        }
        classIndex++;
        if (masks->chanBrightMask) {
            for (int i = 0; i < total; i++) if (masks->chanBrightMask[i]) indexMask[i] = classIndex;
        }
        classIndex++;
        if (masks->pointMask) {
            for (int i = 0; i < total; i++) if (masks->pointMask[i]) indexMask[i] = classIndex;
        }

        snprintf(filename, sizeof(filename), "%smask_merged_%d.png", datasetPath, index);
        writeIndexMaskPNG(indexMask, nsamp, nchan, filename);
        free(indexMask);
        return; // merged 模式下直接返回
    }

    // horizontal
    if (masks->horizontalMask) {
        snprintf(filename, sizeof(filename), "%smask_horizontal_%d.png", datasetPath, index);
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

    // point
    if (masks->pointMask) {
        snprintf(filename, sizeof(filename), "%smask_point_%d.png", datasetPath, index);
        writeIndexMaskPNG(masks->pointMask, nsamp, nchan, filename);
    }

    // // chanBright
    // if (masks->chanBrightMask) {
    //     snprintf(filename, sizeof(filename), "%smask_chanBright_%d.png", datasetPath, index);
    //     writeIndexMaskPNG(masks->chanBrightMask, nsamp, nchan, filename);
    // }

    // // chanDark
    // if (masks->chanDarkMask) {
    //     snprintf(filename, sizeof(filename), "%smask_chanDark_%d.png", datasetPath, index);
    //     writeIndexMaskPNG(masks->chanDarkMask, nsamp, nchan, filename);
    // }

    // // chanComplex
    // if (masks->chanComplexMask) {
    //     snprintf(filename, sizeof(filename), "%smask_chanComplex_%d.png", datasetPath, index);
    //     writeIndexMaskPNG(masks->chanComplexMask, nsamp, nchan, filename);
    // }
}

void allocIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan) {
    if (!m) return;
    m->horizontalMask = (int *)calloc(nsamp * nchan, sizeof(int));
    m->verticalMask   = (int *)calloc(nsamp * nchan, sizeof(int));
    m->globalMask     = (int *)calloc(nsamp * nchan, sizeof(int));
    m->pointMask      = (int *)calloc(nsamp * nchan, sizeof(int));
    m->chanBrightMask = (int *)calloc(nsamp * nchan, sizeof(int));
    m->chanDarkMask   = (int *)calloc(nsamp * nchan, sizeof(int));
    m->chanComplexMask = (int *)calloc(nsamp * nchan, sizeof(int));
}

void clearIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan) {
    if (!m) return;
    memset(m->horizontalMask,    0, sizeof(int)*nsamp*nchan);
    memset(m->verticalMask,      0, sizeof(int)*nsamp*nchan);
    memset(m->globalMask,        0, sizeof(int)*nsamp*nchan);
    memset(m->pointMask,         0, sizeof(int)*nsamp*nchan);
    memset(m->chanBrightMask,    0, sizeof(int)*nsamp*nchan);
    memset(m->chanDarkMask,      0, sizeof(int)*nsamp*nchan);
    memset(m->chanComplexMask,   0, sizeof(int)*nsamp*nchan);
}

void freeIdentNSigmaMasks(IdentNSigmaMasks *m) {
    if (!m) return;
    free(m->horizontalMask);
    free(m->verticalMask);
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
