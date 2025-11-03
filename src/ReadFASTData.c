#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>
#include <float.h>
#include <nvtx3/nvToolsExt.h>

#include "omp.h"
#include "cpgplot.h"
#include "fitsio.h"
// #include "fftw3.h"

#include "ReadFASTData.h"
#include "identification.h"
#include "findStats.h"
#include "mask.h"
#include "plot.h"
#include "cmd.h"
#include "transpose.h"
#include "psrPalett.h"
#include "cuda_acceleration.h"
#include "include/alg_CLFD.h"
#include "include/alg_IQRM.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif
/* weak reference for optional CFITSIO API; if absent, pointer will be NULL */
extern int fits_is_reentrant(void) __attribute__((weak));
/* ===========================
 * Minimal async reader (single-slot prefetch)
 * =========================== */
typedef struct {
    char filename[512];
    int nchan;
    int blockSize;
    int blocksPerRead;
} ReaderCtx;

typedef struct {
    int requested;            // -1 none; >=0 block index
    unsigned char *outRaw;    // target raw buffer
    float *scaleRows;         // target scale rows buffer (blocksPerRead*nchan)
    float *offsetRows;        // target offset rows buffer (blocksPerRead*nchan)
    int result_ready;         // 0/1
    int result_status;        // CFITSIO status
} ReaderSlot;

static pthread_t g_reader_thread;
static pthread_mutex_t g_reader_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_reader_cv  = PTHREAD_COND_INITIALIZER;
static int g_reader_running = 0;
static ReaderCtx  g_reader_ctx;
static ReaderSlot g_reader_slot;

static void* reader_thread_main(void *arg) {
    (void)arg;
    fitsfile *rf = NULL;
    int st = 0;
    fits_open_file(&rf, g_reader_ctx.filename, READONLY, &st);
    if (st) {
        pthread_mutex_lock(&g_reader_mtx);
        g_reader_slot.result_ready = 1;
        g_reader_slot.result_status = st;
        pthread_cond_broadcast(&g_reader_cv);
        pthread_mutex_unlock(&g_reader_mtx);
        return NULL;
    }

    for (;;) {
        pthread_mutex_lock(&g_reader_mtx);
        while (g_reader_running && g_reader_slot.requested < 0) {
            pthread_cond_wait(&g_reader_cv, &g_reader_mtx);
        }
        if (!g_reader_running) { pthread_mutex_unlock(&g_reader_mtx); break; }
        int blockIndex = g_reader_slot.requested;
        unsigned char *outRaw = g_reader_slot.outRaw;
        float *scaleRows = g_reader_slot.scaleRows;
        float *offsetRows = g_reader_slot.offsetRows;
        g_reader_slot.requested = -1;
        pthread_mutex_unlock(&g_reader_mtx);

        int status = 0;
        readRawBlock(rf, blockIndex, g_reader_ctx.blocksPerRead, g_reader_ctx.nchan,
                     g_reader_ctx.blockSize, NULL, NULL, scaleRows, offsetRows, outRaw, &status);

        pthread_mutex_lock(&g_reader_mtx);
        g_reader_slot.result_ready = 1;
        g_reader_slot.result_status = status;
        pthread_cond_broadcast(&g_reader_cv);
        pthread_mutex_unlock(&g_reader_mtx);
    }

    if (rf) fits_close_file(rf, &st);
    return NULL;
}

static int start_reader_thread_prefetch(const Metadata *m) {
    int reent = 1;
    if (fits_is_reentrant) reent = fits_is_reentrant();
    if (!reent) {
        fprintf(stderr, "[CFITSIO] Non-reentrant build; async prefetch disabled.\n");
        return 0;
    }
    memset(&g_reader_ctx, 0, sizeof(g_reader_ctx));
    strncpy(g_reader_ctx.filename, m->filename, sizeof(g_reader_ctx.filename)-1);
    g_reader_ctx.nchan = m->nchan;
    g_reader_ctx.blockSize = m->blockSize;
    g_reader_ctx.blocksPerRead = m->blocksPerRead;

    pthread_mutex_lock(&g_reader_mtx);
    g_reader_running = 1;
    g_reader_slot.requested = -1;
    g_reader_slot.result_ready = 0;
    g_reader_slot.result_status = 0;
    pthread_mutex_unlock(&g_reader_mtx);
    pthread_create(&g_reader_thread, NULL, reader_thread_main, NULL);
    return 1;
}

static void stop_reader_thread_prefetch(void) {
    pthread_mutex_lock(&g_reader_mtx);
    g_reader_running = 0;
    pthread_cond_broadcast(&g_reader_cv);
    pthread_mutex_unlock(&g_reader_mtx);
    pthread_join(g_reader_thread, NULL);
}

static void submit_read_request_async(int blockIndex,
                                      unsigned char *outRaw,
                                      float *scaleRows,
                                      float *offsetRows) {
    pthread_mutex_lock(&g_reader_mtx);
    g_reader_slot.requested = blockIndex;
    g_reader_slot.outRaw = outRaw;
    g_reader_slot.scaleRows = scaleRows;
    g_reader_slot.offsetRows = offsetRows;
    g_reader_slot.result_ready = 0;
    pthread_cond_broadcast(&g_reader_cv);
    pthread_mutex_unlock(&g_reader_mtx);
}

static int wait_read_ready_async(int *status_out) {
    pthread_mutex_lock(&g_reader_mtx);
    while (!g_reader_slot.result_ready && g_reader_running) {
        pthread_cond_wait(&g_reader_cv, &g_reader_mtx);
    }
    int ready = g_reader_slot.result_ready;
    int st = g_reader_slot.result_status;
    pthread_mutex_unlock(&g_reader_mtx);
    if (status_out) *status_out = st;
    return ready;
}

#ifndef SWAP
#define SWAP(a, b)          \
    do                      \
    {                       \
        typeof(a) temp = a; \
        a = b;              \
        b = temp;           \
    } while (0)
#endif

void setup_openmp(int ncpus)
{
    if (ncpus > 1)
    {
        int maxcpus = omp_get_num_procs();
        int openmp_numthreads = (ncpus <= maxcpus) ? ncpus : maxcpus;
        omp_set_dynamic(0);
        omp_set_num_threads(openmp_numthreads);
        printf("Using %d threads with OpenMP\n\n", openmp_numthreads);
    }
    else
    {
        omp_set_num_threads(1);
    }
}

char *extractSourceName(const char *absolutePath)
{
    const char *last_slash = strrchr(absolutePath, '/');                         // Find last '/' of the absolute path
    const char *filename = (last_slash != NULL) ? last_slash + 1 : absolutePath; // Get filename after last '/'
    char *filename_copy = strdup(filename);                                      // Copy filename to a temporary string
    char *first_part = strtok(filename_copy, "_");                               // Separate underscore using `strtok`
    char *result = strdup(first_part);                                           // Copy the first part to a new string
    free(filename_copy);                                                         // Free the temporary string
    return result;
}

void wait_for_mouse_click()
{
    float x, y;
    char ch;
    int device = 1;
    printf("Click LEFT mouse button for next page, RIGHT to exit...\n");
    cpgband(device, 0, 0, 0, &x, &y, &ch);
    cpgeras();
    if (ch == 'X' || ch == 'x')
    {
        cpgend();
        exit(0);
    }
}

unsigned int next_power_of_two(unsigned int x)
{
    if (x == 0)
        return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

int readMetadata(Metadata *m)
{
    int status = 0;
    int nulval, anynul;
    fitsfile *fptr;
    fits_open_file(&fptr, m->filename, READONLY, &status);

    fits_movnam_hdu(fptr, BINARY_TBL, "SUBINT  ", 0, &status);             // move to hdu by name
    fits_read_key(fptr, TINT, "NCHAN", &m->nchan, NULL, &status);        // number of channels
    fits_read_key(fptr, TDOUBLE, "CHAN_BW", &m->chan_bw, NULL, &status); // channel bandwidth
    fits_read_key(fptr, TDOUBLE, "TBIN", &m->tbin, NULL, &status);       // time resolution
    fits_read_key(fptr, TINT, "NSBLK", &m->nsblk, NULL, &status);        // number of samples per subint block
    fits_read_key(fptr, TINT, "NAXIS2", &m->naxis2, NULL, &status);      // number of subint blocks in the Subint HDU
    fits_get_colnum(fptr, CASESEN, "DAT_FREQ", &m->colnumFreq, &status);
    fits_get_colnum(fptr, CASESEN, "DATA", &m->colnumData, &status);
    fits_read_key(fptr, TINT, "NPOL", &m->npol, NULL, &status); // number of polarizations

    float freqArray[m->nchan];
    fits_read_col(fptr, TFLOAT, m->colnumFreq, 1, 1, m->nchan, &nulval, freqArray, &anynul, &status);
    m->lofreq = freqArray[0];
    m->hifreq = freqArray[m->nchan - 1];

    fits_close_file(fptr, &status);

    /* Calculate secondary parameters */

    // Throw an error if both timeDuration and blocksPerRead are specified
    if (m->timeDuration > 0.0f && m->blocksPerRead > 0)
    {
        fprintf(stderr, "Error: Cannot specify both -d timeDuration and -n blocksPerRead. Advice using -n as it is accurate.\n");
        return -1;
    }
    // If one is specified, calculate the other
    if (m->timeDuration > 0.0f)
    {
        if (m->blocksPerRead <= 0)
        {
            m->blocksPerRead = (int)(m->timeDuration / (m->nsblk * m->tbin));
            if (m->blocksPerRead <= 0)
            {
                fprintf(stderr, "Error: Invalid blocksPerRead calculated from timeDuration. Please check your input.\n");
                return -1;
            }
            printf("Message: Calculated blocksPerRead from timeDuration: %d blocks.\n", m->blocksPerRead);
        }
    }
    else if (m->blocksPerRead > 0)
    {
        m->timeDuration = m->blocksPerRead * m->nsblk * m->tbin;
        printf("Message: Calculated timeDuration from blocksPerRead: %.2f seconds.\n", m->timeDuration);
    }

    m->nsamp = m->blocksPerRead * m->nsblk;
    m->nchanBinned = m->nchan / m->binFactorFreq;
    m->nsampBinned = m->nsamp / m->binFactorTime;
    m->tbinBinned = m->tbin * m->binFactorTime;
    m->chan_bwBinned = m->chan_bw * m->binFactorFreq;
    m->blockSize = m->nchan * m->nsblk;
    m->binnedBlockSize = m->blockSize / (m->binFactorTime * m->binFactorFreq);
    return status;
}

/// @brief Convert a PostScript file to PNG format.
/// @param saveName The name of the PostScript file to convert.
/// @return 0 on success, -1 on failure.
int convert_ps_to_png(char *saveName)
{
    char pngname[512];
    char cmd[1024];

    // Check input validity
    if (!saveName)
        return -1;

    /* === Remove Plot Device suffix from PS filename === */
    // size_t len = strlen(saveName);
    char *last_backslash = strrchr(saveName, '/');
    if (last_backslash == NULL)
        return -1;
    *last_backslash = '\0';

    /* === Generate PNG filename by replacing ".ps" with ".png" === */
    strncpy(pngname, saveName, sizeof(pngname) - 1);
    pngname[sizeof(pngname) - 1] = '\0'; // Ensure null-termination
    char *ext = strstr(pngname, ".ps");  // Find ".ps" extension
    if (!ext)
        return -1;       // Invalid input if no ".ps" found
    strcpy(ext, ".png"); // Replace with ".png"

    /* === Convert PS to PNG (white background, 7680x4320 resolution) === */
    snprintf(cmd, sizeof(cmd), "convert -density 600 %s -background white -flatten -depth 8 -resize 7680x4320 %s",
             saveName, pngname);
    if (system(cmd) != 0)
        return -1;

    // Remove original PS file (optional)
    snprintf(cmd, sizeof(cmd), "rm %s", saveName);
    system(cmd); // Ignore removal errors

    return 0;
}

/// @brief Perform 1D downsampling on a 1D array.
/// @param array Input array.
/// @param inputSize Size of `array`.
/// @param binFactor Downsampling factor.
/// @param binnedArray Output array.
void downsamp1D(float *array, int inputSize, int binFactor, float *binnedArray)
{
    if (binFactor == 1) {
        memcpy(binnedArray, array, sizeof(float) * inputSize);
        return;
    }

    int i, j;
    int dsSize = inputSize / binFactor;
    #pragma omp parallel for
    for (i = 0; i < dsSize; i++)
    {
        float sum = 0.0f;
        for (j = 0; j < binFactor; j++)
        {
            int idx = i * binFactor + j;
            sum += array[idx];
        }
        binnedArray[i] = sum / binFactor;
    }
}

/// @brief Perform 2D downsampling on a 2D array. Theoretically it doesn't care if the array is
/// transposed or not. Specifying `isTranspose` is because this function retains the transposition
/// status, i.e. input untransposed, output untransposed; input transposed, output transposed.
/// @param array Input undownsamped 2D array of data.
/// @param nsamp Original number of samples of `array`
/// @param nchan Original number of channels of `array`
/// @param binnedArray Output downsamped data.
/// @param binFactorTime Downsamp factor of time.
/// @param binFactorFreq Downsamp factor of frequency.
/// @param isTranspose Whether `array` is transposed after being read from FITS file.
void downsamp2D(float *array, int nsamp, int nchan,
                float *binnedArray, int binFactorTime, int binFactorFreq, int isTranspose) 
{
    if ((nsamp % binFactorTime) || (nchan % binFactorFreq)) {
        fprintf(stderr, "Error: nsamp (%d) must be divisible by binFactorTime (%d), "
                       "and nchan (%d) must be divisible by binFactorFreq (%d).\n",
                nsamp, binFactorTime, nchan, binFactorFreq);
        return;
    }

    if (binFactorFreq * binFactorTime == 1) {
        memcpy(binnedArray, array, sizeof(float) * nsamp * nchan);
        return;
    }
    int nsampBinned = nsamp / binFactorTime;
    int nchanBinned = nchan / binFactorFreq;
    if (isTranspose) {
        int i, j, ti, fj;
        #pragma omp parallel for collapse(2)
        for (i = 0; i < nsampBinned; i++) {
            for (j = 0; j < nchanBinned; j++) {
                float sum = 0.0f;
                for (ti = 0; ti < binFactorTime; ti++) {
                    for (fj = 0; fj < binFactorFreq; fj++) {
                        int samp_idx = i * binFactorTime + ti;
                        int freq_idx = j * binFactorFreq + fj;
                        sum += array[freq_idx * nsamp + samp_idx];
                    }
                }
                binnedArray[j * nsampBinned + i] = sum / (binFactorTime * binFactorFreq);
            }
        }
    } else {
        int i, j, ti, fj;
        #pragma omp parallel for collapse(2)
        for (i = 0; i < nsampBinned; i++) {
            for (j = 0; j < nchanBinned; j++) {
                float sum = 0.0f;
                for (ti = 0; ti < binFactorTime; ti++) {
                    for (fj = 0; fj < binFactorFreq; fj++) {
                        int samp_idx = i * binFactorTime + ti;
                        int freq_idx = j * binFactorFreq + fj;
                        sum += array[samp_idx * nchan + freq_idx];
                    }
                }
                binnedArray[i * nchanBinned + j] = sum / (binFactorTime * binFactorFreq);
            }
        }
    }
}

/// 使用每个 SUBINT 自身的 DAT_SCL/DAT_OFFS 将原始字节解压为浮点
/// raw:         输入原始字节，布局为 blocksPerRead 行，每行 nsblk*nchan 连续
/// out:         输出浮点数组，布局与 raw 相同（时间主序），总长度 blocksPerRead*nsblk*nchan
/// scaleRows:   每行的 DAT_SCL（长度 blocksPerRead*nchan），第 k 行起点为 k*nchan
/// offsetRows:  每行的 DAT_OFFS（长度 blocksPerRead*nchan）
/// blocksPerRead: 本次读取的 SUBINT 行数
/// nsblk:       每行的样本数（TBIN 方向）
/// nchan:       频率通道数
static inline void sclOffsToFloatPerRow(const unsigned char *raw, float *out, 
    const float *scaleRows, const float *offsetRows, int blocksPerRead, 
    int nsblk, int nchan)
{
    const size_t rowDataSize = (size_t)nsblk * (size_t)nchan;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < blocksPerRead; ++k) {
        for (int j = 0; j < nsblk; ++j) {
            const float *scl  = scaleRows  + (size_t)k * (size_t)nchan;
            const float *offs = offsetRows + (size_t)k * (size_t)nchan;
            const unsigned char *src = raw + (size_t)k * rowDataSize + (size_t)j * (size_t)nchan;
            float *dst = out + (size_t)k * rowDataSize + (size_t)j * (size_t)nchan;
            #pragma omp simd
            for (int i = 0; i < nchan; ++i) {
                dst[i] = (float)src[i] * scl[i] + offs[i];
            }
        }
    }
}

void getProfile(float *restrict array, int nsamp, int nchan, float *restrict freqProfile, float *restrict timeProfile, bool *restrict mask)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            int validCount = 0;
            for (int j = 0; j < nsamp; j++)
            {
                int maskIdx = i * nsamp + j;
                if (!mask || !mask[maskIdx])
                {
                    sum += array[i * nsamp + j];
                    validCount++;
                }
            }
            freqProfile[i] = (validCount > 0) ? sum / validCount : 0.0f;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            int validCount = 0;
            for (int j = 0; j < nchan; j++)
            {
                int maskIdx = j * nsamp + i;
                if (!mask || !mask[maskIdx])
                {
                    sum += array[j * nsamp + i];
                    validCount++;
                }
            }
            timeProfile[i] = (validCount > 0) ? sum / validCount : 0.0f;
        }
    }
}

void getProfileStd(float *restrict array, int nsamp, int nchan, float *restrict freqProfile, float *restrict timeProfile, bool *restrict mask)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;
            int validCount = 0;
            float *chanPtr = array + i * nsamp;

            for (int j = 0; j < nsamp; j++)
            {
                int maskIdx = i * nsamp + j;
                if (!mask || !mask[maskIdx])
                {
                    float value = chanPtr[j];
                    sum += value;
                    sumSq += value * value;
                    validCount++;
                }
            }
            if (validCount > 1)
            {
                float mean = sum / validCount;
                float variance = (sumSq - 2 * mean * sum + validCount * mean * mean) / validCount;
                freqProfile[i] = sqrt(variance);
            }
            else
            {
                freqProfile[i] = 0.0f;
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;
            int validCount = 0;

            for (int j = 0; j < nchan; j++)
            {
                int maskIdx = j * nsamp + i;
                if (!mask || !mask[maskIdx])
                {
                    float value = array[j * nsamp + i];
                    sum += value;
                    sumSq += value * value;
                    validCount++;
                }
            }
            if (validCount > 1)
            {
                float mean = sum / validCount;
                float variance = (sumSq - 2 * mean * sum + validCount * mean * mean) / validCount;
                timeProfile[i] = sqrt(variance);
            }
            else
            {
                timeProfile[i] = 0.0f;
            }
        }
    }
}

void calcCompress(float *data, int nchan, int nsamp, float *scale, float *offset)
{
    int ch;
    // #pragma omp parallel for
    for (ch = 0; ch < nchan; ch++)
    {
        float *channel_data = data + ch * nsamp;
        float min, max;
        findMinMax(channel_data, nsamp, &min, &max);

        float range = max - min;
        if (range == 0)
        {
            scale[ch] = 1.0f;
            offset[ch] = (max == 0) ? 0.0f : max;
        }
        else
        {
            scale[ch] = range / 255.0f;
            offset[ch] = min;
        }
    }
}

void applyCompress(float *outData, unsigned char *target_data, int nchan, int nsamp,
                   float *scale, float *offset)
{
    int i, j;
    // #pragma omp parallel for collapse(2)
    for (i = 0; i < nchan; i++)
    {
        for (j = 0; j < nsamp; j++)
        {
            float scaled_value = (outData[j * nchan + i] - offset[i]) / scale[i];
            target_data[j * nchan + i] = (unsigned char)(scaled_value + 0.5f);
        }
    }
}

void applyScaleOffset(float *data, float *scale, float *offset, int lenx, int nchanBinned) {
    int i, j;
    #pragma omp parallel for schedule(static)
    for (j = 0; j < lenx; j++) {
        float *row = data + (size_t)j * (size_t)nchanBinned;
        /* hint to vectorize inner loop */
        #pragma omp simd
        for (i = 0; i < nchanBinned; i++) {
            row[i] = row[i] * scale[i] + offset[i];
        }
    }
}


/// @brief Read raw data block from FITS file including scale and offset
/// @param fptr FITS file pointer
/// @param blockIndex Index of the current subint BLOCK (ii in the main loop)
/// @param blocksPerRead Number of blocks to read per iteration
/// @param nchan Number of channels
/// @param blockSize Size of each data block (nsblk * nchan)
/// @param scale Output scale array for the first row (size nchan). Kept for backward-compat.
/// @param offset Output offset array for the first row (size nchan). Kept for backward-compat.
/// @param scaleRows Output per-row scale buffer (size blocksPerRead * nchan). If NULL, will be ignored.
/// @param offsetRows Output per-row offset buffer (size blocksPerRead * nchan). If NULL, will be ignored.
/// @param outRawData Output raw data buffer
/// @param status FITS status pointer
void readRawBlock(fitsfile *fptr, int blockIndex, int blocksPerRead, int nchan, int blockSize,
                 float *scale, float *offset, float *scaleRows, float *offsetRows,
                 unsigned char *outRawData, int *status) {
    int anynul = 0;
    int col_scl = 0, col_offs = 0, col_data = 0;
    const long startRow = (long)(blockIndex * blocksPerRead + 1);

    // Resolve column indices once
    fits_get_colnum(fptr, CASEINSEN, "DAT_SCL",  &col_scl,  status);
    fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &col_offs, status);
    fits_get_colnum(fptr, CASEINSEN, "DATA",     &col_data, status);
    if (*status) {
        fprintf(stderr, "Error locating columns for reading block %d\n", blockIndex);
        fits_report_error(stderr, *status);
        return;
    }

    for (int k = 0; k < blocksPerRead; k++) {
        long row = startRow + k;
        // Read per-row scale/offset when buffers provided
        if (scaleRows) {
            fits_read_col(fptr, TFLOAT, col_scl,  row, 1, nchan, NULL, scaleRows  + (size_t)k * (size_t)nchan, &anynul, status);
            if (*status) break;
        }
        if (offsetRows) {
            fits_read_col(fptr, TFLOAT, col_offs, row, 1, nchan, NULL, offsetRows + (size_t)k * (size_t)nchan, &anynul, status);
            if (*status) break;
        }
        // Read raw DATA bytes for this row
        fits_read_col(fptr, TBYTE,  col_data, row, 1, blockSize, NULL, outRawData + (size_t)k * (size_t)blockSize, &anynul, status);
        if (*status) break;
    }

    // Also populate first-row scale/offset for backward compatibility if requested
    if (!*status && scale && scaleRows) {
        memcpy(scale, scaleRows, sizeof(float) * (size_t)nchan);
    }
    if (!*status && offset && offsetRows) {
        memcpy(offset, offsetRows, sizeof(float) * (size_t)nchan);
    }

    if (*status) {
        fprintf(stderr, "Error reading block %d (per-row SCL/OFFS/DATA)\n", blockIndex);
        fits_report_error(stderr, *status);
    }
}

void writeBlock(
    fitsfile *fptr, int blocksPerRead,
    int nchanBinned, int nsampBinned, int binnedBlockSize,
    float *offset, float *scale, unsigned char *outRawData, int naxis2, int blockIndex, int *status)
{
    int col1;
    int firstrow = blockIndex * blocksPerRead + 1;

    fits_movnam_hdu(fptr, BINARY_TBL, "SUBINT", 0, status);

    int k;
    for (k = 0; k < blocksPerRead; k++) {
        if (firstrow > naxis2) {
            printf("Reached end of file at block %d\n", firstrow);
            break;
        }

        fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &col1, status);
        fits_write_col(fptr, TFLOAT, col1, firstrow, 1, nchanBinned, offset, status);

        fits_get_colnum(fptr, CASEINSEN, "DAT_SCL ", &col1, status);
        fits_write_col(fptr, TFLOAT, col1, firstrow, 1, nchanBinned, scale, status);

        int writeSize = nsampBinned / blocksPerRead * nchanBinned;
        fits_get_colnum(fptr, CASEINSEN, "DATA", &col1, status);
        fits_write_col(fptr, TBYTE, col1, firstrow, 1, writeSize, outRawData + k * binnedBlockSize, status);

        firstrow++;
    }

    fits_flush_file(fptr, status);
    if(*status != 0) {
        fprintf(stderr, "Error writing block %d to FITS file!\n", blockIndex);
        fits_report_error(stderr, *status);
    }
}

void writeFITSDataset(unsigned char *outRawData, float *scale, float *offset, 
                      int nsampBinned, int nchanBinned, int blockIndex, Metadata *m, int *status) 
{
    fitsfile *fptr = NULL;
    char filename[256];
    
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char *sourceName = extractSourceName(m->filename);
    snprintf(filename, sizeof(filename), "%s/%s_%04d%02d%02d_block%d.fits",
             m->datasetPath, sourceName,
             t->tm_year+1900, t->tm_mon+1, t->tm_mday, blockIndex);

    // If a file with the same name already exists, terminate with a clear message
    struct stat st;
    if (stat(filename, &st) == 0) {
        fprintf(stderr, "### ERROR: File with name '%s' already exists! ###\n", filename);
        free(sourceName);
        exit(EXIT_FAILURE);
    }

    fits_create_file(&fptr, filename, status);
    if (*status) {
        fits_report_error(stderr, *status);
        free(sourceName);
        return;
    }

    /* --- Create binary table structure (consistent with original file) --- */
    char *ttype[] = {"DAT_OFFS", "DAT_SCL", "DATA"};
    char *tunit[] = {"", "", "RAW"};
    char *tform[3];
    char tform_offs[32], tform_scl[32], tform_data[32];
    snprintf(tform_offs, sizeof(tform_offs), "%dE", nchanBinned);
    snprintf(tform_scl, sizeof(tform_scl), "%dE", nchanBinned);
    snprintf(tform_data, sizeof(tform_data), "%dB", nchanBinned * nsampBinned);
    tform[0] = tform_offs;
    tform[1] = tform_scl;
    tform[2] = tform_data;
    
    // Create binary table HDU (SUBINT type)
    fits_create_tbl(fptr, BINARY_TBL, 0, 3, ttype, tform, tunit, "SUBINT", status);
    if (*status) {
        fits_report_error(stderr, *status);
        fits_close_file(fptr, status);
        free(sourceName);
        return;
    }

    /* --- Write header keywords --- */
    fits_write_key(fptr, TFLOAT, "TBIN", &m->tbinBinned, "Time per sample (s)", status);
    fits_write_key(fptr, TFLOAT, "CHAN_BW", &m->chan_bwBinned, "Channel bandwidth (MHz)", status);
    fits_write_key(fptr, TINT, "NSBLK", &nsampBinned, "Samples per block", status);
    fits_write_key(fptr, TINT, "NCHAN", &nchanBinned, "Frequency channels", status);
    fits_write_key(fptr, TSTRING, "ORIGIN", m->filename, "Source PSRFITS filename", status);
    fits_write_key(fptr, TINT, "BLOCKIDX", &blockIndex, "Original block index", status);
    fits_write_key(fptr, TINT, "NBLOCKS", &m->blocksPerRead, "Number of blocks per read", status);

    char dateStr[32];
    strftime(dateStr, sizeof(dateStr), "%Y-%m-%dT%H:%M:%S", t);
    fits_write_key(fptr, TSTRING, "DATE", dateStr, "File creation date", status);

    /* --- Write data columns (symmetric with readRawBlock) --- */
    int col_offs, col_scl, col_data;
    fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &col_offs, status);
    fits_get_colnum(fptr, CASEINSEN, "DAT_SCL", &col_scl, status);
    fits_get_colnum(fptr, CASEINSEN, "DATA", &col_data, status);
    fits_write_col(fptr, TFLOAT, col_offs, 1, 1, nchanBinned, offset, status);  // Write offset array
    fits_write_col(fptr, TFLOAT, col_scl, 1, 1, nchanBinned, scale, status);     // Write scale array
    fits_write_col(fptr, TBYTE, col_data, 1, 1, nsampBinned*nchanBinned, outRawData, status); // Write raw data

    // Close file
    fits_close_file(fptr, status);
    if (*status) {
        fits_report_error(stderr, *status);
    } else {
        printf("Successfully wrote block %d to %s\n", blockIndex, filename);
    }
    free(sourceName);
}

// ---------------------------
// Minimal async write thread pool
// ---------------------------

typedef struct WriteTask {
    unsigned char *outRawData; // deep-copied buffer (nsampBinned * nchanBinned)
    float *scaleBinned;        // deep-copied buffer (nchanBinned)
    float *offsetBinned;       // deep-copied buffer (nchanBinned)
    int nsampBinned;
    int nchanBinned;
    int blockIndex;
    struct WriteTask *next;
} WriteTask;

static pthread_t *g_writer_threads = NULL;
static int g_writer_nthreads = 0;
static pthread_mutex_t g_writer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_writer_cond = PTHREAD_COND_INITIALIZER;
static WriteTask *g_writer_head = NULL;
static WriteTask *g_writer_tail = NULL;
static int g_writer_running = 0;
static Metadata *g_writer_meta = NULL; // non-owning; valid until writer stops

static void enqueue_task(WriteTask *task) {
    task->next = NULL;
    if (g_writer_tail) {
        g_writer_tail->next = task;
        g_writer_tail = task;
    } else {
        g_writer_head = g_writer_tail = task;
    }
}

static WriteTask* dequeue_task() {
    WriteTask *t = g_writer_head;
    if (!t) return NULL;
    g_writer_head = t->next;
    if (!g_writer_head) g_writer_tail = NULL;
    return t;
}

static void* writer_thread_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_writer_mutex);
        while (g_writer_running && g_writer_head == NULL) {
            pthread_cond_wait(&g_writer_cond, &g_writer_mutex);
        }
        // When not running and no tasks, exit
        if (!g_writer_running && g_writer_head == NULL) {
            pthread_mutex_unlock(&g_writer_mutex);
            break;
        }
        WriteTask *task = dequeue_task();
        pthread_mutex_unlock(&g_writer_mutex);

        if (task) {
            int status = 0;
            writeFITSDataset(task->outRawData, task->scaleBinned, task->offsetBinned,
                             task->nsampBinned, task->nchanBinned, task->blockIndex,
                             g_writer_meta, &status);
            // Free task buffers
            free(task->outRawData);
            free(task->scaleBinned);
            free(task->offsetBinned);
            free(task);
        }
    }
    return NULL;
}

static void start_writer_threads(Metadata *m, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    pthread_mutex_lock(&g_writer_mutex);
    g_writer_meta = m;
    g_writer_running = 1;
    g_writer_nthreads = nthreads;
    g_writer_threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)g_writer_nthreads);
    pthread_mutex_unlock(&g_writer_mutex);
    for (int i = 0; i < g_writer_nthreads; ++i) {
        pthread_create(&g_writer_threads[i], NULL, writer_thread_main, NULL);
    }
}

static void stop_writer_threads_and_join() {
    pthread_mutex_lock(&g_writer_mutex);
    // Signal threads to stop after draining queue
    g_writer_running = 0;
    pthread_cond_broadcast(&g_writer_cond);
    pthread_mutex_unlock(&g_writer_mutex);

    for (int i = 0; i < g_writer_nthreads; ++i) {
        pthread_join(g_writer_threads[i], NULL);
    }
    free(g_writer_threads);
    g_writer_threads = NULL;
    g_writer_nthreads = 0;

    // Safety: free any remaining tasks if any (should be none)
    pthread_mutex_lock(&g_writer_mutex);
    WriteTask *t = g_writer_head;
    while (t) {
        WriteTask *next = t->next;
        free(t->outRawData);
        free(t->scaleBinned);
        free(t->offsetBinned);
        free(t);
        t = next;
    }
    g_writer_head = g_writer_tail = NULL;
    pthread_mutex_unlock(&g_writer_mutex);
}

static int submit_write_task(const unsigned char *outRawData,
                             const float *scaleBinned,
                             const float *offsetBinned,
                             int nsampBinned,
                             int nchanBinned,
                             int blockIndex) {
    // Deep copy buffers to decouple from producer lifetime
    size_t data_bytes = (size_t)nsampBinned * (size_t)nchanBinned;
    size_t vec_bytes = (size_t)nchanBinned * sizeof(float);

    WriteTask *task = (WriteTask*)malloc(sizeof(WriteTask));
    if (!task) return -1;
    task->outRawData = (unsigned char*)malloc(data_bytes);
    task->scaleBinned = (float*)malloc(vec_bytes);
    task->offsetBinned = (float*)malloc(vec_bytes);
    if (!task->outRawData || !task->scaleBinned || !task->offsetBinned) {
        free(task->outRawData); free(task->scaleBinned); free(task->offsetBinned); free(task);
        return -1;
    }
    memcpy(task->outRawData, outRawData, data_bytes);
    memcpy(task->scaleBinned, scaleBinned, vec_bytes);
    memcpy(task->offsetBinned, offsetBinned, vec_bytes);
    task->nsampBinned = nsampBinned;
    task->nchanBinned = nchanBinned;
    task->blockIndex = blockIndex;
    task->next = NULL;

    pthread_mutex_lock(&g_writer_mutex);
    enqueue_task(task);
    pthread_cond_signal(&g_writer_cond);
    pthread_mutex_unlock(&g_writer_mutex);
    return 0;
}

// --- Mask allocation/cleanup helpers are implemented in mask.c ---

int setup_cuda(Metadata *m) {
    m->cudaReady = 0; // Default: CUDA not ready
    if (m->enableCuda) {
        if (cuda_isAvailable()) {
            if (cuda_init() == 0) {
                printf("CUDA acceleration enabled.\n");
                m->cudaReady = 1; // CUDA is ready to use
                return 0;
            } else {
                printf("CUDA initialization failed, using CPU only.\n");
                m->enableCuda = 0; // Disable CUDA for this session
                return 0;
            }
        } else {
            fprintf(stderr, "Error: You requested CUDA acceleration, but CUDA is not available on this system.\n");
            fprintf(stderr, "Please run without --enableCuda option and check CUDA installation.\n");
            return -1;
        }
    } else {
        printf("Message: --enableCuda option not found, using CPU. \n");
        return 0;
    }
}

int main(int argc, char *argv[])
{
    double global_start_time = omp_get_wtime();

    Metadata m;
    int status = parseCommandLineArguments(argc, argv, &m);
    if (status != 0) return status;
    readMetadata(&m);

    // Initialize OpenMP and CUDA 
    setup_openmp(20);
    if (setup_cuda(&m) != 0) return -1;

    fitsfile *fptr = NULL;
    int fits_status = 0;
    fits_open_file(&fptr, m.filename, READWRITE, &fits_status);
    fits_movnam_hdu(fptr, BINARY_TBL, "SUBINT", 0, &fits_status);

    // Setup PGPLOT output
    if (m.plot)
    {
        char *sourceName = extractSourceName(m.filename);
        size_t requiredSize = strlen(m.datasetPath) + strlen(sourceName) + 64;
        char *saveName = (char*)malloc(requiredSize);
        memset(saveName, 0, requiredSize);
        snprintf(saveName, requiredSize, "%s/%s_%.2f_%d_%d.ps/VCPS",
                 m.datasetPath, sourceName, 0.0, m.binFactorTime, m.binFactorFreq);
        if (ensure_pgplot_device(saveName)) {
            // Set a 2x5 subdivision to mimic original cpgbeg layout
            cpgsubp(2, 5);
        } else {
            fprintf(stderr, "PGPLOT device open failed, plotting will be skipped.\n");
        }
        free(sourceName);
        free(saveName);
    }

    // Start background writer threads (for dataset writes)
    start_writer_threads(&m, 2);

    int nsamp, nchan, binFactorFreq, binFactorTime, nsampBinned, nchanBinned, blockSize, binnedBlockSize;
    int blocksPerRead, naxis2, colnumFreq;
    int startTime;
    int nulval, anynul;

    nsamp = m.nsamp;
    nchan = m.nchan;
    binFactorFreq = m.binFactorFreq;
    binFactorTime = m.binFactorTime;
    nsampBinned = m.nsampBinned;
    nchanBinned = m.nchanBinned;
    blockSize = m.blockSize;
    binnedBlockSize = m.binnedBlockSize;
    blocksPerRead = m.blocksPerRead;
    naxis2 = m.naxis2;
    colnumFreq = m.colnumFreq;
    startTime = m.startTime;

    // Read frequency array
    float *freqArray = malloc(sizeof(float) * nchan);
    float *dsFreqArray = malloc(sizeof(float) * nchanBinned);
    fits_read_col(fptr, TFLOAT, colnumFreq, 1, 1, nchan, &nulval, freqArray, &anynul, &fits_status);
    downsamp1D(freqArray, nchan, binFactorFreq, dsFreqArray);

    // Calculate buffer parameters
    int numReads = naxis2 / blocksPerRead;

    // Allocate output buffers
    unsigned char *outRawData = malloc(sizeof(unsigned char) * (size_t)nchan * (size_t)nsamp);
    float *outData = malloc(sizeof(float) * nchanBinned * nsampBinned);
    float *outDataT = malloc(sizeof(float) * nchanBinned * nsampBinned);

    // Allocate RFI-related buffers (only those actually used below)
    int *flaggedChans = (int *)calloc(nchanBinned, sizeof(int)); // Track fully flagged channels

    IdentNSigmaMasks maskSet;
    allocIdentNSigmaMasks(&maskSet, nsampBinned, nchanBinned);


    float *finalMedian = malloc(sizeof(float) * nsampBinned * nchanBinned);
    if (!finalMedian) {
        fprintf(stderr, "Memory allocation failed for finalMedian in ReadFASTData\n");
        return 1;
    }
    float *finalStd = malloc(sizeof(float) * nsampBinned * nchanBinned);
    if (!finalStd) {
        fprintf(stderr, "Memory allocation failed for finalStd in ReadFASTData\n");
        return 1;
    }
    float *scale = malloc(sizeof(float) * nchan);  // Fix: Use original nchan size
    float *offset = malloc(sizeof(float) * nchan); // Fix: Use original nchan size
    float *scaleBinned = malloc(sizeof(float) * nchanBinned); // For downsampled data
    float *offsetBinned = malloc(sizeof(float) * nchanBinned); // For downsampled data
    // Per-row (per SUBINT) scale/offset buffers, size = blocksPerRead x nchan
    float *scaleRows = (float *)malloc(sizeof(float) * (size_t)blocksPerRead * (size_t)nchan);
    if (!scaleRows) {
        fprintf(stderr, "Memory allocation failed for scaleRows in ReadFASTData\n");
        return 1;
    }
    float *offsetRows = (float *)malloc(sizeof(float) * (size_t)blocksPerRead * (size_t)nchan);
    if (!offsetRows) {
        fprintf(stderr, "Memory allocation failed for offsetRows in ReadFASTData\n");
        return 1;
    }
    
    // 解压后的全分辨率浮点缓冲（无论是否下采样都走统一流程，内部有早退）
    float *rawToFloatArray = (float *)malloc(sizeof(float) * (size_t)nsamp * (size_t)nchan);
    if (!rawToFloatArray) {
        fprintf(stderr, "Memory allocation failed for rawToFloatArray in ReadFASTData\n");
        return 1;
    }

    // Buffers for subChanMed to avoid repeated malloc/free
    float *subChanMed_medianBuf = (float *)malloc(sizeof(float) * (size_t)nchanBinned);
    float *subChanMed_tempDataBuf = (float *)malloc(sizeof(float) * (size_t)nsampBinned);

    // Buffers for identSubstNSigma to avoid per-iteration malloc/free
    int   *identSubst_goodSamps = (int *)malloc(sizeof(int) * (size_t)nsampBinned);
    int   *identSubst_randIdxs  = (int *)malloc(sizeof(int) * (size_t)nsampBinned);
    float *identSubst_medTemp   = (float *)malloc(sizeof(float) * (size_t)nsampBinned * (size_t)nchanBinned);
    if (!identSubst_goodSamps || !identSubst_randIdxs || !identSubst_medTemp) {
        fprintf(stderr, "Memory allocation failed for identSubst buffers in ReadFASTData\n");
        free(identSubst_goodSamps);
        free(identSubst_randIdxs);
        free(identSubst_medTemp);
        return 1;
    }

    int maxScratchThreads = omp_get_max_threads();
    if (maxScratchThreads < 1) maxScratchThreads = 1;
    size_t inChanScratchCount = (size_t)maxScratchThreads * (size_t)nsampBinned;
    float *inChanScratch = NULL;
    if (inChanScratchCount > 0) {
        inChanScratch = (float *)malloc(sizeof(float) * inChanScratchCount);
        if (!inChanScratch) {
            fprintf(stderr, "Warning: allocating inChanScratch buffer failed; falling back to per-channel allocation.\n");
            inChanScratchCount = 0;
        }
    }

    // Accumulators for loop timing stats
    double loop_total_time = 0.0;
    double loop_min_time = DBL_MAX;
    double loop_max_time = 0.0;

    int ii;
    int numiter = 0;
    
    // Note: cfitsio library handles I/O optimization internally
    // We rely on its buffering and caching mechanisms
    
    /*
     * Async prefetch setup: create double buffers and kick off first read if enabled
     */
    unsigned char *rawBufA = outRawData;
    unsigned char *rawBufB = (unsigned char*)malloc(sizeof(unsigned char) * (size_t)nchan * (size_t)nsamp);
    float *sclBufA = scaleRows;
    float *sclBufB = (float*)malloc(sizeof(float) * (size_t)blocksPerRead * (size_t)nchan);
    float *offBufA = offsetRows;
    float *offBufB = (float*)malloc(sizeof(float) * (size_t)blocksPerRead * (size_t)nchan);
    int use_async_prefetch = start_reader_thread_prefetch(&m);
    int useA = 1; // which buffer corresponds to current ii when ready
    if (use_async_prefetch && numReads > 0) {
        submit_read_request_async(0, rawBufA, sclBufA, offBufA);
    }

    for (ii = 0; ii < numReads; ii++)
    {
        double loop_start = omp_get_wtime();
        printf("Processing block %d of %d, %d subints per block, %.3f%% done.\n", ii, numReads, blocksPerRead, (ii * 100.0f / numReads));
        
        // Reset flaggedChans, finalMedian, and finalStd to avoid accumulation/residual data across loops
        memset(flaggedChans, 0, nchanBinned * sizeof(int));
        memset(finalMedian, 0, nsampBinned * nchanBinned * sizeof(float));
        memset(finalStd, 0, nsampBinned * nchanBinned * sizeof(float));
        
        // Read a raw data block of `blocksPerRead` subints with its scale and offset
        double read_start = omp_get_wtime();
        unsigned char *currRaw = outRawData;
        float *currScl = scaleRows;
        float *currOff = offsetRows;
        if (use_async_prefetch) {
            // wait for prefetch completion for current ii
            int st_ready = 0;
            (void)wait_read_ready_async(&st_ready);
            if (st_ready) {
                // Non-zero indicates error; fall back to sync read and disable prefetch
                fprintf(stderr, "Prefetch read error on block %d (status=%d); disabling async prefetch.\n", ii, st_ready);
                use_async_prefetch = 0;
            }
            // Select buffer based on toggle
            if (useA) { currRaw = rawBufA; currScl = sclBufA; currOff = offBufA; }
            else      { currRaw = rawBufB; currScl = sclBufB; currOff = offBufB; }
            // Fill first-row scale/offset to keep downstream behavior
            memcpy(scale,  currScl, sizeof(float) * (size_t)nchan);
            memcpy(offset, currOff, sizeof(float) * (size_t)nchan);
            // Submit next request ASAP to overlap with compute
            if (use_async_prefetch && (ii + 1) < numReads) {
                if (useA) submit_read_request_async(ii + 1, rawBufB, sclBufB, offBufB);
                else      submit_read_request_async(ii + 1, rawBufA, sclBufA, offBufA);
            }
        }
        if (!use_async_prefetch) {
            // Synchronous path
            readRawBlock(fptr, ii, blocksPerRead, nchan, blockSize,
                scale, offset, currScl, currOff,
                currRaw, &fits_status);
            if (fits_status) {
                fits_report_error(stderr, fits_status);
            }
        }
        double read_time = omp_get_wtime() - read_start;
        printf("Read block time: %.4f seconds\n", read_time);
        
        // Start timing
        double convert_start = omp_get_wtime();
        // Decode (per-SUBINT DAT_SCL/DAT_OFFS)
        sclOffsToFloatPerRow(currRaw, rawToFloatArray, currScl, currOff,
            blocksPerRead, m.nsblk, nchan);
        // Downsample data (time, freq)
        downsamp2D(rawToFloatArray, nsamp, nchan, outData, binFactorTime, binFactorFreq, 0);
        // Downsample scale/offset (freq)
        downsamp1D(scale, nchan, binFactorFreq, scaleBinned);
        downsamp1D(offset, nchan, binFactorFreq, offsetBinned);
        // Report timing
        double convert_time = omp_get_wtime() - convert_start;
        printf("Convert/downsamp time: %.4f seconds\n", convert_time);
        


        // CUDA-accelerated transpose (with fallback to CPU)
        printf("Performing matrix transpose (%d x %d)...\n", nsampBinned, nchanBinned);
        double transpose_start = omp_get_wtime();
        if (m.cudaReady) {
            cuda_transpose(outData, outDataT, nsampBinned, nchanBinned);
        } else {
            transpose(outData, nsampBinned, nchanBinned, outDataT);
        }
        double transpose_time = omp_get_wtime() - transpose_start;
        printf("Transpose time: %.4f seconds\n", transpose_time);
        

        
        if (m.generateMasks)
        {
            clearIdentNSigmaMasks(&maskSet, nsampBinned, nchanBinned);

            // Plot the unprocessed raw data
            if (m.plot)
            { 
                cpgpage(); // Create new graphics page
                cpgmtxt("T", 3.0, 0.35, 0.5, "Raw Data");
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, NULL);
            }


            // =======================Subtract Channel Median========================================================
            int subtractChanMed = 0; 
            if (subtractChanMed)
            {
                double chanMed_start = omp_get_wtime();
                subChanMed(outDataT, nsampBinned, nchanBinned,
                                       subChanMed_medianBuf, subChanMed_tempDataBuf);
                double chanMed_time = omp_get_wtime() - chanMed_start;
                printf("Channel median subtraction time: %.4f seconds\n", chanMed_time);
            } else {
                printf("Warning: Median subtraction disabled. If channel RFI is severe, consider lowering outChanNSigma threshold.\n");
            }

            // 在这里写出
            // {
            //     // 设置固定的 scale 和 offset
            //     float *scale_const = malloc(sizeof(float) * nchanBinned);
            //     float *offset_const = malloc(sizeof(float) * nchanBinned);
            //     for (int i = 0; i < nchanBinned; i++) {
            //         scale_const[i] = 1.0f;
            //         offset_const[i] = 0.0f;
            //     }
                
            //     // 压缩数据
            //     // calcCompress(outDataT, nchanBinned, nsampBinned, scale_const, offset_const);
            //     transpose(outDataT, nchanBinned, nsampBinned, outData);
            //     // applyCompress(outData, outRawData, nchanBinned, nsampBinned, scale_const, offset_const);
                
            //     // 写出 FITS 数据集
            //     writeFITSDataset(outRawData, scale_const, offset_const, nsampBinned, nchanBinned, ii, &m, &fits_status);
                
            //     // 释放内存
            //     free(scale_const);
            //     free(offset_const);
            // }

            // if (m.plot)
            // {
            //     cpgpage();
            //     char text2[100];
            //     snprintf(text2, sizeof(text2), "Result after subtracting channel median");
            //     cpgmtxt("T", 3.5, 0.5, 0.5, text2);
            //     plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, NULL);
            // }

            // =====================Channel IQR + IQRM（合并，一次可视化，封装调用）============================
            // {
            //     float q_chan = 1.5f;       // Tukey 系数
            //     float thr = 3.0f;          // IQRM 阈值
            //     int *mask_chan2d = NULL;   // 输出 2D 掩码（需 free）

            //     int flagged_channels = IQRM(
            //         outDataT, nsampBinned, nchanBinned,
            //         q_chan, thr,
            //         &mask_chan2d);

            //     if (m.plot) {
            //         cpgpage();
            //         char t_before[160]; snprintf(t_before, sizeof(t_before), "Before Channel-IQR + IQRM (q=%.2f, thr=%.1f)", q_chan, thr);
            //         cpgmtxt("T", 3.0, 0.35, 0.5, t_before);
            //         plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, NULL);

            //         cpgpage();
            //         char t_after[200]; snprintf(t_after, sizeof(t_after), "After Channel-IQR + IQRM flagged %d/%d (q=%.2f, thr=%.1f)", flagged_channels, nchanBinned, q_chan, thr);
            //         cpgmtxt("T", 3.0, 0.35, 0.5, t_after);
            //         plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, mask_chan2d, NULL);
            //     }

            //     if (writeMasks && mask_chan2d) {
            //         char mask_CHAN_filename[256];
            //         sprintf(mask_CHAN_filename, "%smask_CHAN_%d.png", m.datasetPath, ii);
            //         writeIndexMaskPNG(mask_chan2d, nsampBinned, nchanBinned, mask_CHAN_filename);
            //     }

            //     free(mask_chan2d);
            // }

            // =====================CLFD (启用)====================================================
            // CLFD(outDataT, nsampBinned, nchanBinned, 3.0f, NULL, 0, mask_CLFD);
            // for (int idx = 0; idx < nsampBinned * nchanBinned; idx++) {
            //     if (mask_SPIKE[idx]) mask_CLFD[idx] = 1; // 如果有 spike 掩码，合并
            // }
            // substPixels2D(outDataT, nsampBinned, nchanBinned, mask_CLFD);
            // if (writeMasks) {
            //     char mask_CLFD_filename[256];
            //     sprintf(mask_CLFD_filename, "%smask_CLFD_%d.png", m.datasetPath, ii);
            //     writeIndexMaskPNG(mask_CLFD, nsampBinned, nchanBinned, mask_CLFD_filename);
            // }
            // if (m.plot) {
            //     cpgpage(); cpgmtxt("T", 3.0, 0.35, 0.5, "Before CLFD");
            //     plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, NULL);
            //     cpgpage(); cpgmtxt("T", 3.0, 0.35, 0.5, "After CLFD");
            //     plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, mask_CLFD, NULL);
            // }



            float NSigmaInChan = 3.0f; // Updated to use 3-sigma threshold for iterative outlier detection
            // float NSigmaOutChan = 2.0f; // When data is severely affected and subtractMedian is turned off, a lower threshold is favored
            float NSigmaOutChan = 3.0f; // Use 3-sigma threshold for out-of-channel detection as well
            if (m.doSubstitution)
            {
                double rfi_start = omp_get_wtime();
                printf("=== Using CPU RFI detection ===\n");
                identSubstNSigma(outDataT, nsampBinned, nchanBinned, NSigmaInChan, NSigmaOutChan, ii, m.plot,
                                &maskSet, finalMedian, finalStd, m.cudaReady, flaggedChans,
                                identSubst_goodSamps, identSubst_randIdxs, identSubst_medTemp,
                                inChanScratch, inChanScratchCount);
                double rfi_time = omp_get_wtime() - rfi_start;
                printf("RFI detection time: %.4f seconds\n", rfi_time);

            }
            if (m.writeMasks)
            {
                // merge=0 默认分别输出；如需合并成索引图传 1
                int merge = 1;
                writeAllMasksPNG(&maskSet, nsampBinned, nchanBinned, m.datasetPath, ii, merge);
            }
            if (m.plot)
            { // Plot result after NSigma substitution
                cpgpage();
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, maskSet.globalMask, flaggedChans);
            }

            // Plot all individual masks except globalMask
            if (m.plot)
            {
                plotAllMasks(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, &maskSet, flaggedChans);
            }

            // --- Final visualization: show data after all pixel substitutions ---
            if (m.plot)
            {
                cpgpage();
                cpgmtxt("T", 3.0, 0.35, 0.5, "Final result after all substitutions");
                // Display processed data without mask overlay; stats panels remain enabled (top/right)
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, flaggedChans);
            }


            // =================================SumThreshold===================================================
            // float timesOfSigma = 3.0f;
            // int M_len = 6;
            // int win_samp = 3, win_chan = 3;
            // float thrup = 0.5f, thrdown = 0.5f;
            // memset(mask_ST, 0, sizeof(int) * nsampBinned * nchanBinned);
            // if (m.doSumThreshold)
            // {
            //     sumthreshold_2d(outDataT, nsampBinned, nchanBinned,
            //                     mask_chanRFI, mask_ST, timesOfSigma, M_len);
            // }
            // if (writeMasks)
            // {
            //     char mask_ST_filename[256];
            //     sprintf(mask_ST_filename, "%smask_ST_%d.png", m.datasetPath, ii);
            //     writeIndexMaskPNG(mask_ST, nsampBinned, nchanBinned, mask_ST_filename);
            // }
            // if (m.plot)
            // {
            //     cpgpage();
            //     char text3[100];
            //     snprintf(text3, sizeof(text3), "Result of SumThreshold RFI detection with chi=%.2f", timesOfSigma);
            //     cpgmtxt("T", 3.5, 0.5, 0.5, text3);
            //     plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, mask_ST, flaggedChans);
            // }
            // if (m.doSumThreshold)
            // {
            //     substPixels2D(outDataT, nsampBinned, nchanBinned, mask_ST);
            // }
            // if (m.plot)
            // {
            //     cpgpage();
            //     cpgmtxt("T", 3.5, 0.5, 0.5, "Result after pixel substitution");
            //     // 顶部和右侧都显示标准差，无掩码
            //     plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL, flaggedChans);
            //     cpgpage();
            // }
        }

        if (m.write)
        {
            double write_start = omp_get_wtime();
            calcCompress(outDataT, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            transpose(outDataT, nchanBinned, nsampBinned, outData);
            applyCompress(outData, outRawData, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            
            int writeBack = m.writeBack, writeDastset = 1;
            
            if (writeBack) {
                writeBlock(fptr, blocksPerRead, nchanBinned, nsampBinned, binnedBlockSize,
                        offsetBinned, scaleBinned, outRawData, m.naxis2, ii, &fits_status);
                if (fits_status)
                {
                    fits_report_error(stderr, fits_status);
                    printf("Writing subint block %d, failed!\n", ii);
                }
                else
                {
                    printf("Writing subint block %d, OK.\n", ii);
                }    
            }

            if (writeDastset) {
                // Offload dataset writing to background thread
                if (submit_write_task(outRawData, scaleBinned, offsetBinned, nsampBinned, nchanBinned, ii) != 0) {
                    fprintf(stderr, "Warning: submit_write_task failed, fallback to synchronous write.\n");
                    writeFITSDataset(outRawData, scaleBinned, offsetBinned, nsampBinned, nchanBinned, ii, &m, &fits_status);
                }
            }
            double write_time = omp_get_wtime() - write_start;
            printf("Write operations time: %.4f seconds\n", write_time);
        }


        numiter++;
        if (numiter % 200 == 0) {
            m.plot = 1; 
        } else if (numiter % 200 == 1) {
            m.plot = 0; 
        }
        if (numiter == 200) {
            status = 0; // keep original behavior but allow cleanup and summary
            break;
        }

        // if (numiter == 1) {
        //     m.plot = 0;
        // } else if (numiter == 7) {
        //     m.plot = 1;
        // } else if (numiter == 9) {
        //     m.plot = 0;
        // } else if (numiter == 60) {
        //     m.plot = 1;
        // } else if (numiter == 63) {
        //     m.plot = 0;
        //     return 0; 
        // }
        
        double loop_time = omp_get_wtime() - loop_start;
        loop_total_time += loop_time;
        if (loop_time < loop_min_time) loop_min_time = loop_time;
        if (loop_time > loop_max_time) loop_max_time = loop_time;
        printf("Total loop %d time: %.4f seconds\n", ii, loop_time);
        printf("iteration %d done.\n\n", numiter);

        // toggle buffer for next iteration
        if (use_async_prefetch) useA = !useA;

    }

    // Stop writer threads and cleanup
    stop_writer_threads_and_join();
    if (g_reader_running) {
        stop_reader_thread_prefetch();
    }
    // Cleanup
    if (m.plot) cpgend();
    fits_close_file(fptr, &fits_status);
    free(freqArray);
    free(dsFreqArray);
    free(outData);
    free(outDataT);
    
    freeIdentNSigmaMasks(&maskSet);
    free(flaggedChans);
    free(outRawData);
    free(rawBufB);
    free(scale);
    free(offset);
    free(scaleBinned);
    free(offsetBinned);
    free(scaleRows);
    free(offsetRows);
    free(sclBufB);
    free(offBufB);
    free(rawToFloatArray);
    free(finalMedian);
    free(finalStd);
    free(subChanMed_medianBuf);
    free(subChanMed_tempDataBuf);
    free(identSubst_goodSamps);
    free(identSubst_randIdxs);
    free(identSubst_medTemp);
    free(inChanScratch);

    // Print loop timing summary (if any iteration executed)
    if (numiter > 0) {
        double loop_avg_time = loop_total_time / (double)numiter;
        printf("Loop summary: %d iterations, total %.2f s, avg %.4f s, min %.4f s, max %.4f s\n",
               numiter, loop_total_time, loop_avg_time, loop_min_time, loop_max_time);
    }

    double global_end_time = omp_get_wtime();
    printf("All done, time taken: %.2f seconds.\n", global_end_time - global_start_time);
    return status;
}

/// @brief Extract mask for specific frequency channels only, set all other pixels to zero
/// @param inputMask Input mask array (nchan x nsamp layout)
/// @param outputMask Output mask array (will be modified)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param channelIndices Array of channel indices to keep (0-based)
/// @param numChannels Number of channels in channelIndices array
/// @param isTranspose Whether the mask is transposed (nsamp x nchan layout)
void extractChannelMask(int *inputMask, int *outputMask, int nsamp, int nchan, 
                       int *channelIndices, int numChannels, int isTranspose)
{
    // Initialize output mask to all zeros
    memset(outputMask, 0, nsamp * nchan * sizeof(int));
    
    int k, i, j;
    if (isTranspose) {
        // For transposed layout: nsamp x nchan
        for (k = 0; k < numChannels; k++) {
            int chanIdx = channelIndices[k];
            if (chanIdx >= 0 && chanIdx < nchan) {
                for (i = 0; i < nsamp; i++) {
                    outputMask[i * nchan + chanIdx] = inputMask[i * nchan + chanIdx];
                }
            }
        }
    } else {
        // For normal layout: nchan x nsamp
        for (k = 0; k < numChannels; k++) {
            int chanIdx = channelIndices[k];
            if (chanIdx >= 0 && chanIdx < nchan) {
                for (j = 0; j < nsamp; j++) {
                    outputMask[chanIdx * nsamp + j] = inputMask[chanIdx * nsamp + j];
                }
            }
        }
    }
}

/// @brief Extract mask for a range of frequency channels
/// @param inputMask Input mask array 
/// @param outputMask Output mask array (will be modified)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param startChannel Start channel index (inclusive, 0-based)
/// @param endChannel End channel index (inclusive, 0-based)
/// @param isTranspose Whether the mask is transposed (nsamp x nchan layout)
void extractChannelRangeMask(int *inputMask, int *outputMask, int nsamp, int nchan,
                            int startChannel, int endChannel, int isTranspose)
{
    // Initialize output mask to all zeros
    memset(outputMask, 0, nsamp * nchan * sizeof(int));
    
    // Validate range
    if (startChannel < 0) startChannel = 0;
    if (endChannel >= nchan) endChannel = nchan - 1;
    if (startChannel > endChannel) return;
    
    int chanIdx, i, j;
    if (isTranspose) {
        // For transposed layout: nsamp x nchan
        for (chanIdx = startChannel; chanIdx <= endChannel; chanIdx++) {
            for (i = 0; i < nsamp; i++) {
                outputMask[i * nchan + chanIdx] = inputMask[i * nchan + chanIdx];
            }
        }
    } else {
        // For normal layout: nchan x nsamp  
        for (chanIdx = startChannel; chanIdx <= endChannel; chanIdx++) {
            for (j = 0; j < nsamp; j++) {
                outputMask[chanIdx * nsamp + j] = inputMask[chanIdx * nsamp + j];
            }
        }
    }
}