#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <omp.h>

#include "mask.h"

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
            float val = (float)mask[i * nsamp + j];
            row_pointers[nchan - 1 - i][j] = (png_byte)(val * 255.0f);
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
