#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>
#include <float.h>
#include <ctype.h>
#include <nvtx3/nvToolsExt.h>

#include "omp.h"
#include "cpgplot.h"
#include "fitsio.h"

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
 * Thread-local CFITSIO handle helpers (per-thread FITS handles)
 * =========================== */
#define SLICE_PTR(base, elems_per_thread, tid) ((base) + ((size_t)(tid) * (size_t)(elems_per_thread)))

static inline int slice_index_from_iter(int iter, int maxThreads) {
    if (omp_in_parallel()) {
        int tid = omp_get_thread_num();
        return (tid >= 0) ? tid : 0;
    }
    if (maxThreads < 1) maxThreads = 1;
    return iter % maxThreads;
}

static _Thread_local fitsfile *tls_read_fptr = NULL; // per-thread read handle
static _Thread_local int tls_read_hdu_ready = 0;     // per-thread SUBINT selected

/* Mutex to serialize all CFITSIO calls when library is not reentrant
    or to force serial I/O (方案A). This ensures correctness even if
    libcfitsio is not built with thread-safety. */
static pthread_mutex_t fits_io_mutex = PTHREAD_MUTEX_INITIALIZER;

#define FITS_IO_LOCK()   pthread_mutex_lock(&fits_io_mutex)
#define FITS_IO_UNLOCK() pthread_mutex_unlock(&fits_io_mutex)

static inline fitsfile* get_thread_read_fptr(const char *filename, int *status) {
    if (!tls_read_fptr) {
        int st = 0;
        FITS_IO_LOCK();
        fits_open_file(&tls_read_fptr, filename, READONLY, &st);
        FITS_IO_UNLOCK();
        if (status) *status = st;
    } else if (status) {
        *status = 0;
    }
    return tls_read_fptr;
}

static inline fitsfile* ensure_thread_read_ready(const char *filename, int *status) {
    int st = 0;
    fitsfile *f = get_thread_read_fptr(filename, &st);
    if (status) *status = st;
    if (st) return f;
    if (!tls_read_hdu_ready) {
        FITS_IO_LOCK();
        fits_movnam_hdu(f, BINARY_TBL, "SUBINT", 0, &st);
        FITS_IO_UNLOCK();
        if (st) {
            if (status) *status = st;
        } else {
            tls_read_hdu_ready = 1;
        }
    }
    return f;
}

static inline void close_thread_read_fptr(void) {
    if (tls_read_fptr) {
        int st = 0;
        FITS_IO_LOCK();
        fits_close_file(tls_read_fptr, &st);
        FITS_IO_UNLOCK();
        tls_read_fptr = NULL;
    }
}
/* ------------------------------
 * PostScript -> PDF conversion helper (later use)
 * ------------------------------ */
static char g_last_plot_device[1024] = {0};

int convert_ps_to_pdf(const char *deviceName, int remove_ps)
{
    if (!deviceName || !*deviceName) return -1;
    char psname[512];
    char pdfname[512];
    memset(psname, 0, sizeof(psname));
    strncpy(psname, deviceName, sizeof(psname)-1);
    // Strip trailing "/VCPS" (or other) after .ps
    char *last_slash = strrchr(psname, '/');
    if (!last_slash) return -1;
    *last_slash = '\0';
    char *ext = strstr(psname, ".ps");
    if (!ext) return -1;
    strncpy(pdfname, psname, sizeof(pdfname)-1);
    pdfname[sizeof(pdfname)-1] = '\0';
    char *pdfext = strstr(pdfname, ".ps");
    if (!pdfext) return -1;
    strcpy(pdfext, ".pdf");
    int have_ps2pdf = (system("command -v ps2pdf >/dev/null 2>&1") == 0);
    char cmd[1200];
    if (have_ps2pdf) {
        snprintf(cmd, sizeof(cmd), "ps2pdf %s %s", psname, pdfname);
    } else {
        snprintf(cmd, sizeof(cmd), "gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=%s %s", pdfname, psname);
    }
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "Warning: ps->pdf conversion failed for '%s' (rc=%d)\n", psname, rc);
        return -1;
    }
    if (remove_ps) {
        snprintf(cmd, sizeof(cmd), "rm -f %s", psname);
        system(cmd);
    }
    printf("Converted PostScript to PDF: %s -> %s\n", psname, pdfname);
    return 0;
}
/* Monotonic timer helper to avoid negative durations when system clock slews */
#ifdef CLOCK_MONOTONIC
static inline double mono_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#else
#include <sys/time.h>
static inline double mono_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
#endif
/* ===========================
 * Human-readable generation summary writer
 * =========================== */
static void write_generation_summary(
    const Metadata *m,
    const char *sourceName,
    int blocksPerRead,
    int nsampBinned,
    int nchanBinned,
    int numReads,
    int remainderRows,
    float NSigmaInChan,
    float NSigmaOutChan)
{
    if (!m || !sourceName || !m->datasetPath) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s_summary.txt", m->datasetPath, sourceName);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[Warn] Failed to write summary file: %s\n", path);
        return;
    }

    // Derive friendly time lengths
    double tbin_b = (double)m->tbinBinned;
    double block_obs_sec = (double)nsampBinned * tbin_b;
    double block_obs_min = block_obs_sec / 60.0;
    double block_obs_hr  = block_obs_sec / 3600.0;

    time_t now = time(NULL);
    struct tm *tmv = localtime(&now);
    char timestr[64] = {0};
    if (tmv) strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", tmv);

    // Note: We now report enabled output mask classes for PNG merge instead of internal algorithms count.

    fprintf(fp,
        "# deRFI generation summary (human-readable)\n"
        "# Created at: %s\n\n",
        timestr[0] ? timestr : "(unknown)"
    );

    fprintf(fp, "[Source]\n");
    fprintf(fp, "  Input PSRFITS: %s\n", m->filename);
    fprintf(fp, "  Output directory: %s\n", m->datasetPath);
    fprintf(fp, "\n");

    fprintf(fp, "[Downsampling & Geometry]\n");
    fprintf(fp, "  binFactorTime: %d\n", m->binFactorTime);
    fprintf(fp, "  binFactorFreq: %d\n", m->binFactorFreq);
    fprintf(fp, "  blocksPerRead: %d\n", blocksPerRead);
    fprintf(fp, "  Input NCHAN: %d\n", m->nchan);
    fprintf(fp, "  Input NSBLK: %d\n", m->nsblk);
    fprintf(fp, "  Input TBIN(s): %.9f\n", m->tbin);
    fprintf(fp, "  Output NCHAN (binned): %d\n", nchanBinned);
    fprintf(fp, "  Output nsamp per file (nsampBinned): %d\n", nsampBinned);
    fprintf(fp, "  Output TBIN_Binned(s): %.9f\n", m->tbinBinned);
    fprintf(fp, "  Wall-clock time per output FITS: %.6f s (%.3f min, %.3f hr)\n",
            block_obs_sec, block_obs_min, block_obs_hr);
    fprintf(fp, "\n");

    fprintf(fp, "[Production]\n");
    fprintf(fp, "  Number of samples (files) generated: %d\n", numReads);
    fprintf(fp, "  Tail SUBINT rows dropped: %d\n", remainderRows);
    fprintf(fp, "\n");

    fprintf(fp, "[Masking / Detection]\n");
    fprintf(fp, "  Generate masks: %s\n", m->generateMasks ? "yes" : "no");
    fprintf(fp, "  Write mask PNGs: %s\n", m->writeMasks ? "yes" : "no");
    fprintf(fp, "  Enable NSigma substitution (doSubstitution): %s\n", m->doSubstitution ? "yes" : "no");
    fprintf(fp, "  Enable SumThreshold: %s\n", m->doSumThreshold ? "yes" : "no");
    fprintf(fp, "  NSigma threshold (in-channel): %.3f\n", NSigmaInChan);
    fprintf(fp, "  NSigma threshold (cross-channel): %.3f\n", NSigmaOutChan);
    // Enabled output classes: horizontal, vertical, point, block (periodic disabled)
    fprintf(fp, "  Enabled mask classes (PNG merge): %d\n", 4);
    fprintf(fp, "    - horizontal\n");
    fprintf(fp, "    - vertical\n");
    fprintf(fp, "    - point\n");
    fprintf(fp, "    - block\n");
    // Keep users aware of PNG index mapping as implemented in writeAllMasksPNG
    fprintf(fp, "  PNG class index mapping: horizontal=1, vertical=2, point=6, block=7\n");
    fprintf(fp, "  Periodic point RFI detection: disabled\n");
    fprintf(fp, "\n");

    fprintf(fp, "[Acceleration / Misc]\n");
    fprintf(fp, "  CUDA requested: %s\n", m->enableCuda ? "yes" : "no");
    fprintf(fp, "  CUDA actually used: %s\n", m->cudaReady ? "yes" : "no");

    fclose(fp);
    fprintf(stderr, "Summary written to %s\n", path);
}
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
    char *first_part = strtok(filename_copy, "_");                               // G120.82-21.31
    char *second_part = strtok(NULL, "_");                                       // 20240812
    char *result = NULL;
    if (first_part && second_part) {
        size_t len = strlen(first_part) + 1 + strlen(second_part) + 1;
        result = malloc(len);
        sprintf(result, "%s_%s", first_part, second_part);
    } else if (first_part) {
        result = strdup(first_part);
    }
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
    FITS_IO_LOCK();
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
    FITS_IO_UNLOCK();

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
    // #pragma omp parallel for
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
    // #pragma omp parallel for collapse(2)
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
    // #pragma omp parallel for collapse(2)
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

    // #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < blocksPerRead; ++k) {
        for (int j = 0; j < nsblk; ++j) {
            const float *scl  = scaleRows  + (size_t)k * (size_t)nchan;
            const float *offs = offsetRows + (size_t)k * (size_t)nchan;
            const unsigned char *src = raw + (size_t)k * rowDataSize + (size_t)j * (size_t)nchan;
            float *dst = out + (size_t)k * rowDataSize + (size_t)j * (size_t)nchan;
            // #pragma omp simd
            for (int i = 0; i < nchan; ++i) {
                dst[i] = (float)src[i] * scl[i] + offs[i];
            }
        }
    }
}

void getProfile(float *restrict array, int nsamp, int nchan, float *restrict freqProfile, float *restrict timeProfile, bool *restrict mask)
{
    // #pragma omp parallel
    {
    // #pragma omp for schedule(static)
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

    // #pragma omp for schedule(static)
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
    // #pragma omp parallel
    {
    // #pragma omp for schedule(static)
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

    // #pragma omp for schedule(static)
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
    // #pragma omp parallel for schedule(static)
    for (j = 0; j < lenx; j++) {
        float *row = data + (size_t)j * (size_t)nchanBinned;
        /* hint to vectorize inner loop */
    // #pragma omp simd
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
    // Resolve column indices once and read rows under I/O lock
    FITS_IO_LOCK();
    fits_get_colnum(fptr, CASEINSEN, "DAT_SCL",  &col_scl,  status);
    fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &col_offs, status);
    fits_get_colnum(fptr, CASEINSEN, "DATA",     &col_data, status);
    if (*status) {
        fprintf(stderr, "Error locating columns for reading block %d\n", blockIndex);
        fits_report_error(stderr, *status);
        FITS_IO_UNLOCK();
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
    FITS_IO_UNLOCK();

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
    /* Serialize FITS writes to be safe with non-reentrant CFITSIO */
    FITS_IO_LOCK();
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
    FITS_IO_UNLOCK();
}

void writeFITSDataset(unsigned char *outRawData, float *scale, float *offset, 
                      int nsampBinned, int nchanBinned, int blockIndex, Metadata *m, int *status) 
{
    fitsfile *fptr = NULL;
    char filename[256];
    
    char *sourceName = extractSourceName(m->filename);
    snprintf(filename, sizeof(filename), "%s/%s_block%d.fits",
             m->datasetPath, sourceName, blockIndex);

    // If a file with the same name already exists, terminate with a clear message
    struct stat st;
    if (stat(filename, &st) == 0) {
        fprintf(stderr, "### ERROR: File with name '%s' already exists! ###\n", filename);
        free(sourceName);
        exit(EXIT_FAILURE);
    }

    /* Serialize all CFITSIO calls in this function */
    FITS_IO_LOCK();
    fits_create_file(&fptr, filename, status);
    if (*status) {
        fits_report_error(stderr, *status);
        FITS_IO_UNLOCK();
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
        FITS_IO_UNLOCK();
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

    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char creationDateStr[32];
    strftime(creationDateStr, sizeof(creationDateStr), "%Y-%m-%dT%H:%M:%S", t);
    fits_write_key(fptr, TSTRING, "DATE", creationDateStr, "File creation date", status);

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
    FITS_IO_UNLOCK();
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
    double global_start_time = mono_time_sec();

    Metadata m;
    int status = parseCommandLineArguments(argc, argv, &m);
    if (status != 0) return status;
    readMetadata(&m);

    char *sourceName = extractSourceName(m.filename);

    // Hard check: CFITSIO reentrancy
    if (fits_is_reentrant) {
        int reent = fits_is_reentrant();
        if (!reent) {
            fprintf(stderr, "[CFITSIO] ERROR: CFITSIO is not built as re-entrant/thread-safe. \
                Please reinstall or use a thread-safe build.\n");
            return EXIT_FAILURE;
        }
    }

    // Initialize OpenMP and CUDA (use configured CPU count, default 20)
    setup_openmp(m.ncpus);
    if (setup_cuda(&m) != 0) return -1;

    /* Propagate CLI toggles to identification module */
    setUseIQRM(m.enableIQRM);
    setUseCLFD(m.enableCLFD);

    fitsfile *wfptr = NULL; // optional writer handle (when writeBack)
    int fits_status = 0;

    // Setup PGPLOT output
    if (m.plot)
    {
        char *sourceName = extractSourceName(m.filename);
        size_t requiredSize = strlen(m.datasetPath) + strlen(sourceName) + 64;
        char *saveName = (char*)malloc(requiredSize);
        memset(saveName, 0, requiredSize);
        snprintf(saveName, requiredSize, "%s/%s_%.2f_%d_%d.ps/VCPS",
                 m.datasetPath, sourceName, 0.0, m.binFactorTime, m.binFactorFreq);
        strncpy(g_last_plot_device, saveName, sizeof(g_last_plot_device)-1); // store for later ps->pdf
        if (ensure_pgplot_device(saveName)) {
            // Set a 2x5 subdivision to mimic original cpgbeg layout
            cpgsubp(2, 5);
        } else {
            fprintf(stderr, "PGPLOT device open failed, plotting will be skipped.\n");
        }
        free(sourceName);
        free(saveName);
    }

    // // Start background writer threads (disabled for parallel-safe sync I/O)
    // start_writer_threads(&m, 2);

    int nsamp, nchan, binFactorFreq, binFactorTime, nsampBinned, nchanBinned, blockSize, binnedBlockSize;
    int blocksPerRead, naxis2, colnumFreq;
    int startTime;
    int nulval, anynul;

    // Effective per-iteration geometry: ensure each output file keeps the same time-pixel count
    // Setup per-iteration geometry. Keep user's blocksPerRead as rows-per-iteration,
    // and let binFactorTime only affect the time downsampling.
    nchan = m.nchan;
    binFactorFreq = m.binFactorFreq;
    binFactorTime = m.binFactorTime;
    // rows read per iteration equals user-specified blocksPerRead
    if (m.blocksPerRead < 1) {
        fprintf(stderr, "Error: blocksPerRead must be >= 1, got %d\n", m.blocksPerRead);
        return EXIT_FAILURE;
    }
    if (binFactorTime < 1) {
        fprintf(stderr, "Error: binFactorTime must be >= 1, got %d\n", binFactorTime);
        return EXIT_FAILURE;
    }

    blocksPerRead = m.blocksPerRead;
    nsamp = blocksPerRead * m.nsblk;                 // samples per iteration before downsampling
    if ((nsamp % binFactorTime) != 0) {
        fprintf(stderr,
                "Error: blocksPerRead*NSBLK (%d) must be divisible by binFactorTime (%d) for time downsampling.\n",
                nsamp, binFactorTime);
        return EXIT_FAILURE;
    }
    nsampBinned = nsamp / binFactorTime;             // samples per iteration after time downsampling
    nchanBinned = m.nchanBinned;                     // frequency downsampling unchanged
    blockSize = m.blockSize;                         // per SUBINT row size remains nsblk*nchan
    binnedBlockSize = m.binnedBlockSize;             // per-row binned size remains consistent
    naxis2 = m.naxis2;
    colnumFreq = m.colnumFreq;
    startTime = m.startTime;

    // Prepare read handle (ensure per-thread handle selects SUBINT HDU)
    fitsfile *rf0 = ensure_thread_read_ready(m.filename, &fits_status);
    if (fits_status) { fits_report_error(stderr, fits_status); return fits_status; }

    // Read frequency array
    float *freqArray = malloc(sizeof(float) * nchan);
    float *dsFreqArray = malloc(sizeof(float) * nchanBinned);
    /* serialize this CFITSIO call in case libcfitsio is not reentrant */
    FITS_IO_LOCK();
    fits_read_col(rf0, TFLOAT, colnumFreq, 1, 1, nchan, &nulval, freqArray, &anynul, &fits_status);
    FITS_IO_UNLOCK();
    downsamp1D(freqArray, nchan, binFactorFreq, dsFreqArray);

    // Determine thread count for per-thread scratch slicing
    int maxScratchThreads = omp_get_max_threads();
    if (maxScratchThreads < 1) maxScratchThreads = 1;

    // Sizes per iteration
    const size_t BYTES_per_iter_raw      = (size_t)nchan * (size_t)nsamp;             // unsigned char
    const size_t FLOATS_per_iter_full    = (size_t)nchan * (size_t)nsamp;             // rawToFloatArray
    const size_t FLOATS_per_iter_binned  = (size_t)nchanBinned * (size_t)nsampBinned; // outData/outDataT/finalMedian/finalStd
    const size_t FLOATS_per_iter_rows    = (size_t)blocksPerRead * (size_t)nchan;     // scaleRows/offsetRows
    const size_t FLOATS_per_iter_scale   = (size_t)nchan;                              // scale/offset
    const size_t FLOATS_per_iter_scale_b = (size_t)nchanBinned;                        // scaleBinned/offsetBinned

    // Allocate per-thread-slice buffers
    unsigned char *outRawData_all   = (unsigned char*)malloc(BYTES_per_iter_raw * (size_t)maxScratchThreads);
    float *outData_all              = (float*)malloc(sizeof(float) * FLOATS_per_iter_binned * (size_t)maxScratchThreads);
    float *outDataT_all             = (float*)malloc(sizeof(float) * FLOATS_per_iter_binned * (size_t)maxScratchThreads);
    float *finalMedian_all          = (float*)malloc(sizeof(float) * FLOATS_per_iter_binned * (size_t)maxScratchThreads);
    float *finalStd_all             = (float*)malloc(sizeof(float) * FLOATS_per_iter_binned * (size_t)maxScratchThreads);
    float *scale_all                = (float*)malloc(sizeof(float) * FLOATS_per_iter_scale   * (size_t)maxScratchThreads);
    float *offset_all               = (float*)malloc(sizeof(float) * FLOATS_per_iter_scale   * (size_t)maxScratchThreads);
    float *scaleBinned_all          = (float*)malloc(sizeof(float) * FLOATS_per_iter_scale_b * (size_t)maxScratchThreads);
    float *offsetBinned_all         = (float*)malloc(sizeof(float) * FLOATS_per_iter_scale_b * (size_t)maxScratchThreads);
    float *scaleRows_all            = (float*)malloc(sizeof(float) * FLOATS_per_iter_rows    * (size_t)maxScratchThreads);
    float *offsetRows_all           = (float*)malloc(sizeof(float) * FLOATS_per_iter_rows    * (size_t)maxScratchThreads);
    float *rawToFloatArray_all      = (float*)malloc(sizeof(float) * FLOATS_per_iter_full    * (size_t)maxScratchThreads);

    // Per-thread flagged channels and masks
    int *flaggedChans_all = (int *)calloc((size_t)nchanBinned * (size_t)maxScratchThreads, sizeof(int));
    IdentNSigmaMasks *maskSets = (IdentNSigmaMasks*)malloc(sizeof(IdentNSigmaMasks) * (size_t)maxScratchThreads);
    for (int t = 0; t < maxScratchThreads; t++) {
        allocIdentNSigmaMasks(&maskSets[t], nsampBinned, nchanBinned);
    }

    // Buffers for subChanMed to avoid repeated malloc/free (per-thread slices)
    float *subChanMed_medianBuf = (float *)malloc(sizeof(float) * (size_t)nchanBinned * (size_t)maxScratchThreads);
    float *subChanMed_tempDataBuf = (float *)malloc(sizeof(float) * (size_t)nsampBinned * (size_t)maxScratchThreads);

    // Buffers for identSubstNSigma to avoid per-iteration malloc/free (per-thread slices)
    int   *identSubst_goodSamps = (int *)malloc(sizeof(int) * (size_t)nsampBinned * (size_t)maxScratchThreads);
    int   *identSubst_randIdxs  = (int *)malloc(sizeof(int) * (size_t)nsampBinned * (size_t)maxScratchThreads);
    float *identSubst_medTemp   = (float *)malloc(sizeof(float) * (size_t)nsampBinned * (size_t)nchanBinned * (size_t)maxScratchThreads);

    size_t inChanScratchCount = (size_t)maxScratchThreads * (size_t)nsampBinned;
    float *inChanScratch = NULL;
    if (inChanScratchCount > 0) {
        inChanScratch = (float *)malloc(sizeof(float) * inChanScratchCount);
    }

    // Scratch buffers for vertical stripe detection (moved out of inner functions)
    float *vs_time_means_all = (float *)malloc(sizeof(float) * (size_t)nsampBinned * (size_t)maxScratchThreads);
    unsigned char *vs_flag_time_all = (unsigned char *)malloc((size_t)nsampBinned * (size_t)maxScratchThreads);
    // Fallback channel means buffer (per-thread slices) for outChanDetection 保底均值检测，避免循环内 malloc
    float *fallback_chan_means_all = (float *)malloc(sizeof(float) * (size_t)nchanBinned * (size_t)maxScratchThreads);

     /* Preallocate channel-level masks to avoid per-iteration malloc/free inside the loop
         Each thread gets a slice of size (nchanBinned * nsampBinned) */
     int *mask_chan2d_all = (int *)malloc(sizeof(int) * (size_t)nchanBinned * (size_t)nsampBinned * (size_t)maxScratchThreads);
     int *mask_CLFD_all    = (int *)malloc(sizeof(int) * (size_t)nchanBinned * (size_t)nsampBinned * (size_t)maxScratchThreads);

     /* Optional pulse mask (per-thread slice) */
     bool *mask_pulse_all = (bool *)calloc((size_t)nchanBinned * (size_t)nsampBinned * (size_t)maxScratchThreads, sizeof(bool));

    // Accumulators for loop timing stats
    double loop_total_time = 0.0;
    double loop_min_time = DBL_MAX;
    double loop_max_time = 0.0;

    // Optional write-back handle (open once if needed)
    int enable_writeback = (m.write && m.writeBack) ? 1 : 0;
    if (enable_writeback) {
        FITS_IO_LOCK();
        fits_open_file(&wfptr, m.filename, READWRITE, &fits_status);
        if (fits_status) { fits_report_error(stderr, fits_status); FITS_IO_UNLOCK(); return fits_status; }
        fits_movnam_hdu(wfptr, BINARY_TBL, "SUBINT", 0, &fits_status);
        if (fits_status) { fits_report_error(stderr, fits_status); FITS_IO_UNLOCK(); return fits_status; }
        FITS_IO_UNLOCK();
    }

    int numReads = naxis2 / blocksPerRead; // number of iterations (files)
    int remainderRows = naxis2 % blocksPerRead;
    if (remainderRows != 0) {
        fprintf(stderr,
                "Warning: dropping tail %d SUBINT rows (naxis2=%d not divisible by blocksPerRead=%d).\n",
                remainderRows, naxis2, blocksPerRead);
    }
    // numRead = 200; // Limit to 200 iterations for profiling
    const int plotEvery = 200; // base interval for plotting

    int ii;
    // thresholds used for NSigma (kept here to also record into summary)
    const float NSigmaInChan = m.NSigmaInChan;   // configurable via CLI (default 3.0)
    const float NSigmaOutChan = m.NSigmaOutChan; // configurable via CLI (default 3.0)
    // 配置保底均值检测 σ 倍数（默认 2.0，可由命令行 -F 指定）
    setFallbackMeanNSigma(m.FallbackMeanNSigma);
    setNoBlock(m.noBlock);
    setNoVertical(m.noVertical);
    #pragma omp parallel for schedule(static) reduction(+:loop_total_time) reduction(min:loop_min_time) reduction(max:loop_max_time)
    for (ii = 0; ii < numReads; ii++)
    {
    double loop_start = mono_time_sec();
        printf("Processing block %d of %d, %d subints per block, %.3f%% done.\n",
             ii, numReads, blocksPerRead, (ii * 100.0f / (numReads ? numReads : 1)));
        int tid = slice_index_from_iter(ii, maxScratchThreads);
        // All plotting handled by a single thread for correctness
        const int plotterTid = 0;
        int doPlotThisIter = (m.plot != 0) && ((ii % plotEvery) == 0) && (tid == plotterTid);
        // Per-iteration slices
        unsigned char *outRawData   = SLICE_PTR(outRawData_all,   BYTES_per_iter_raw,      tid);
        float *outData              = SLICE_PTR(outData_all,      FLOATS_per_iter_binned,  tid);
        float *outDataT             = SLICE_PTR(outDataT_all,     FLOATS_per_iter_binned,  tid);
        float *finalMedian          = SLICE_PTR(finalMedian_all,  FLOATS_per_iter_binned,  tid);
        float *finalStd             = SLICE_PTR(finalStd_all,     FLOATS_per_iter_binned,  tid);
        float *scale                = SLICE_PTR(scale_all,        FLOATS_per_iter_scale,   tid);
        float *offset               = SLICE_PTR(offset_all,       FLOATS_per_iter_scale,   tid);
        float *scaleBinned          = SLICE_PTR(scaleBinned_all,  FLOATS_per_iter_scale_b, tid);
        float *offsetBinned         = SLICE_PTR(offsetBinned_all, FLOATS_per_iter_scale_b, tid);
        float *scaleRows            = SLICE_PTR(scaleRows_all,    FLOATS_per_iter_rows,    tid);
        float *offsetRows           = SLICE_PTR(offsetRows_all,   FLOATS_per_iter_rows,    tid);
        float *rawToFloatArray      = SLICE_PTR(rawToFloatArray_all, FLOATS_per_iter_full, tid);
        int   *flaggedChans         = SLICE_PTR(flaggedChans_all, nchanBinned, tid);
        IdentNSigmaMasks *maskSetPtr = &maskSets[tid];
        
        // Reset flaggedChans, finalMedian, and finalStd to avoid accumulation/residual data across loops
        memset(flaggedChans, 0, nchanBinned * sizeof(int));
        memset(finalMedian,  0, nsampBinned * nchanBinned * sizeof(float));
        memset(finalStd,     0, nsampBinned * nchanBinned * sizeof(float));
        
        // Read a raw data block of `blocksPerRead` subints with its scale and offset (synchronous)
    double read_start = mono_time_sec();
        int fits_status_local = 0;
        fitsfile *rf_local = ensure_thread_read_ready(m.filename, &fits_status_local);
        if (fits_status_local) { fits_report_error(stderr, fits_status_local); continue; }
        unsigned char *currRaw = outRawData;
        float *currScl = scaleRows;
        float *currOff = offsetRows;
        readRawBlock(rf_local, ii, blocksPerRead, nchan, blockSize,
                     scale, offset, currScl, currOff,
                     currRaw, &fits_status_local);
        if (fits_status_local) {
            fits_report_error(stderr, fits_status_local);
        }
    double read_time = mono_time_sec() - read_start; if (read_time < 0) read_time = 0;
        printf("Read block time: %.4f seconds\n", read_time);
        
        // Start timing
    double convert_start = mono_time_sec();
        // Decompress and downsample data (time, freq) and scale/offset
        sclOffsToFloatPerRow(currRaw, rawToFloatArray, currScl, currOff,
            blocksPerRead, m.nsblk, nchan);
        downsamp2D(rawToFloatArray, nsamp, nchan, outData, binFactorTime, binFactorFreq, 0);
        downsamp1D(scale, nchan, binFactorFreq, scaleBinned);
        downsamp1D(offset, nchan, binFactorFreq, offsetBinned);
    double convert_time = mono_time_sec() - convert_start; if (convert_time < 0) convert_time = 0;
        printf("Convert/downsamp time: %.4f seconds\n", convert_time);
        
        // CUDA-accelerated transpose (with fallback to CPU)
        printf("Performing matrix transpose (%d x %d)...\n", nsampBinned, nchanBinned);
    double transpose_start = mono_time_sec();
        transpose(outData, nsampBinned, nchanBinned, outDataT);
    double transpose_time = mono_time_sec() - transpose_start; if (transpose_time < 0) transpose_time = 0;
        printf("Transpose time: %.4f seconds\n", transpose_time);
        

        
        if (m.generateMasks)
        {
            clearIdentNSigmaMasks(maskSetPtr, nsampBinned, nchanBinned);

            /* ---------------- Pulse mask (experimental) ----------------
             * We generate it in window-local coordinates (t=0 at this block start),
             * consistent with src/experiment_pulse_mask.py.
             * NOTE: Currently OR into globalMask only (so pulse isn't treated as RFI).
             */
            if (m.hasPulse) {
                /* Calculate block start time to adjust global T0 to block-local time */
                double block_duration = (double)nsampBinned * (double)m.tbinBinned;
                double current_block_start_time = (double)ii * block_duration;
                
                /* T0_local passed to identPulse needs to be: time of a pulse arrival relative to this block's t=0.
                   Since m.pulseT0Local is effectively global T0, we subtract current block's start time. */
                float relativeT0 = m.pulseT0Local - (float)current_block_start_time;

                bool *pulseMask = SLICE_PTR(mask_pulse_all, (size_t)nchanBinned * (size_t)nsampBinned, tid);
                /* Ensure it's clear for this iteration (since it's a pre-allocated scratch buffer) */
                memset(pulseMask, 0, (size_t)nchanBinned * (size_t)nsampBinned * sizeof(bool));

                identPulse(
                    1,
                    m.pulseDM,
                    m.pulseP0,
                    m.pulseWidth,
                    relativeT0, 
                    m.pulselofreq,
                    m.pulsehifreq,
                    dsFreqArray,
                    nchanBinned,
                    m.tbinBinned,
                    nsampBinned,
                    pulseMask
                );

                if (m.interpulse) {
                    float rel_it0;
                    if (m.interpulseT0 >= 0.0f) {
                        rel_it0 = m.interpulseT0 - (float)current_block_start_time;
                    } else {
                        rel_it0 = relativeT0 + 0.5f * m.pulseP0;
                    }
                    identPulse(
                        1,
                        m.pulseDM,
                        m.pulseP0,
                        m.interpulseWidth,
                        rel_it0, 
                        m.pulselofreq,
                        m.pulsehifreq,
                        dsFreqArray,
                        nchanBinned,
                        m.tbinBinned,
                        nsampBinned,
                        pulseMask
                    );
                }
                // logicalOR(maskSetPtr->globalMask, pulseMask, nsampBinned, nchanBinned); // Moved downstream
            }

            // Plot the unprocessed raw data
            if (doPlotThisIter)
            { 
                #pragma omp critical(pgplot)
                {
                    cpgpage(); // Create new graphics page
                    cpgmtxt("T", 3.0, 0.35, 0.5, "Raw Data");
                    plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, NULL, NULL);
                }
            }


            // =======================Subtract Channel Median========================================================
            int subtractChanMed = 0; 
            if (subtractChanMed)
            {
                double chanMed_start = mono_time_sec();
                int tid = slice_index_from_iter(ii, maxScratchThreads);
                float *scm_median = SLICE_PTR(subChanMed_medianBuf, nchanBinned, tid);
                float *scm_temp   = SLICE_PTR(subChanMed_tempDataBuf, nsampBinned, tid);
                subChanMed(outDataT, nsampBinned, nchanBinned,
                                       scm_median, scm_temp);
                double chanMed_time = mono_time_sec() - chanMed_start; if (chanMed_time < 0) chanMed_time = 0;
                printf("Channel median subtraction time: %.4f seconds\n", chanMed_time);
            } else {
                printf("Warning: Median subtraction disabled. If channel RFI is severe, consider lowering outChanNSigma threshold.\n");
            }

            // =====================Channel IQR + IQRM（合并，一次可视化）============================
            {
                float q_chan = 1.5f;       // Tukey 系数
                float thr = 3.0f;          // IQRM 阈值
                int *mask_chan2d_ptr = SLICE_PTR(mask_chan2d_all, nchanBinned * nsampBinned, tid);
                /* Ensure the preallocated buffer is zeroed */
                memset(mask_chan2d_ptr, 0, sizeof(int) * (size_t)nchanBinned * (size_t)nsampBinned);

                if (m.enableIQRM) {
                    /* IQRM currently allocates a mask and returns it; call it then copy into our preallocated slice */
                    int *tmp_mask = NULL;
                    int flagged_channels = IQRM(outDataT, nsampBinned, nchanBinned,
                                                q_chan, thr,
                                                &tmp_mask);
                    if (tmp_mask) {
                        memcpy(mask_chan2d_ptr, tmp_mask, sizeof(int) * (size_t)nchanBinned * (size_t)nsampBinned);
                        free(tmp_mask);
                    }

                    if (m.plot) {
                        cpgpage();
                        char t_before[160]; snprintf(t_before, sizeof(t_before), "Before Channel-IQR + IQRM (q=%.2f, thr=%.1f)", q_chan, thr);
                        cpgmtxt("T", 3.0, 0.35, 0.5, t_before);
                        plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, NULL, NULL);

                        cpgpage();
                        char t_after[200]; snprintf(t_after, sizeof(t_after), "After Channel-IQR + IQRM flagged %d/%d (q=%.2f, thr=%.1f)", flagged_channels, nchanBinned, q_chan, thr);
                        cpgmtxt("T", 3.0, 0.35, 0.5, t_after);
                        plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, (bool*)mask_chan2d_ptr, NULL);
                    }

                    if (m.writeMasks) {
                        char mask_CHAN_filename[256];
                        snprintf(mask_CHAN_filename, sizeof(mask_CHAN_filename), "%smask_CHAN_%d.png", m.datasetPath, ii);
                        writeIndexMaskPNG((const bool*)mask_chan2d_ptr, nsampBinned, nchanBinned, mask_CHAN_filename);
                    }
                }
            }

            // =====================CLFD (启用)====================================================
            {
                if (m.enableCLFD) {
                    int *mask_CLFD = SLICE_PTR(mask_CLFD_all, nchanBinned * nsampBinned, tid);
                    memset(mask_CLFD, 0, sizeof(int) * (size_t)nchanBinned * (size_t)nsampBinned);
                    CLFD(outDataT, nsampBinned, nchanBinned, 3.0f, NULL, 0, mask_CLFD);
                    for (int idx = 0; idx < nsampBinned * nchanBinned; idx++) {
                        if (/* mask_SPIKE may be present elsewhere */ 0) mask_CLFD[idx] = 1; // 如果有 spike 掩码，合并
                    }
                    substPixels2D(outDataT, nsampBinned, nchanBinned, mask_CLFD);
                    if (m.writeMasks) {
                        char mask_CLFD_filename[256];
                        snprintf(mask_CLFD_filename, sizeof(mask_CLFD_filename), "%smask_CLFD_%d.png", m.datasetPath, ii);
                        writeIndexMaskPNG((const bool*)mask_CLFD, nsampBinned, nchanBinned, mask_CLFD_filename);
                    }
                    if (m.plot) {
                        cpgpage(); cpgmtxt("T", 3.0, 0.35, 0.5, "Before CLFD");
                        plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, NULL, NULL);
                        cpgpage(); cpgmtxt("T", 3.0, 0.35, 0.5, "After CLFD");
                        plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, (bool*)mask_CLFD, NULL);
                    }
                }
            }

            // =======================NSigma Identification + Substitution=============================================
            {
                double rfi_start = mono_time_sec();
                printf("=== Using CPU RFI detection ===\n");
                int tid = slice_index_from_iter(ii, maxScratchThreads);
                int   *goodSamps = SLICE_PTR(identSubst_goodSamps, nsampBinned, tid);
                int   *randIdxs  = SLICE_PTR(identSubst_randIdxs,  nsampBinned, tid);
                float *medTemp   = SLICE_PTR(identSubst_medTemp,   (size_t)nsampBinned * (size_t)nchanBinned, tid);
                float *inChanSlice = SLICE_PTR(inChanScratch, nsampBinned, tid);
                size_t inChanSliceCount = (size_t)nsampBinned;
                float *vs_time_means = SLICE_PTR(vs_time_means_all, nsampBinned, tid);
                unsigned char *vs_flag_time = SLICE_PTR(vs_flag_time_all, nsampBinned, tid);
                setFallbackChannelMeansBuffer(SLICE_PTR(fallback_chan_means_all, nchanBinned, tid), (size_t)nchanBinned);
                identSubstNSigma(outDataT, nsampBinned, nchanBinned, NSigmaInChan, NSigmaOutChan, ii, doPlotThisIter,
                                m.doSubstitution, maskSetPtr, finalMedian, finalStd, m.cudaReady, flaggedChans,
                                goodSamps, randIdxs, medTemp, inChanSlice, inChanSliceCount,
                                SLICE_PTR(mask_CLFD_all, nchanBinned * nsampBinned, tid),
                                vs_time_means, vs_flag_time);
                double rfi_time = mono_time_sec() - rfi_start; if (rfi_time < 0) rfi_time = 0;
                printf("RFI detection time: %.4f seconds\n", rfi_time);

                /* Merge pulse mask (if enabled) so it appears in final output. 
                   Done here because identSubstNSigma clears maskSetPtr masks. */
                if (m.hasPulse) {
                   bool *pulseBuf = SLICE_PTR(mask_pulse_all, nchanBinned * nsampBinned, tid);
                   memcpy(maskSetPtr->pulseMask, pulseBuf, sizeof(bool) * nchanBinned * nsampBinned);
                   
                   /* 也将脉冲或到 globalMask 中，以便在绘图中视觉上阻断 RFI (可选) */
                   logicalOR(maskSetPtr->globalMask, pulseBuf, nsampBinned, nchanBinned);

                   /* 移除这里的强制优先级逻辑，让 mask.c 中的写入顺序决定最终颜色 */
                   /* 
                   size_t total_pix = (size_t)nchanBinned * (size_t)nsampBinned;
                   for (size_t idx = 0; idx < total_pix; idx++) {
                       if (pulseBuf[idx]) {
                           maskSetPtr->horizontalMask[idx] = false;
                           maskSetPtr->verticalMask[idx] = false;
                           maskSetPtr->pointMask[idx] = false;
                           maskSetPtr->chanBrightMask[idx] = false;
                           maskSetPtr->chanDarkMask[idx] = false;
                           maskSetPtr->chanComplexMask[idx] = false;
                           maskSetPtr->blockMask[idx] = false;
                           maskSetPtr->periodicMask[idx] = false;
                       }
                   }
                   */
                }
            }
            if (m.writeMasks)
            {
                // merge=0 默认分别输出；如需合并成索引图传 1
                int merge = 1;
                writeAllMasksPNG(maskSetPtr, nsampBinned, nchanBinned, m.datasetPath, ii, merge, sourceName);
            }
            if (doPlotThisIter)
            { // Plot result after NSigma substitution
                #pragma omp critical(pgplot)
                {
                    cpgpage();
                    plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, maskSetPtr->globalMask, flaggedChans);
                }
            }

            // Plot all individual masks except globalMask
            if (doPlotThisIter)
            {
                #pragma omp critical(pgplot)
                {
                    plotAllMasks(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, maskSetPtr, flaggedChans);
                }
            }

            // --- Final visualization: show data after all pixel substitutions ---
            if (doPlotThisIter)
            {
                #pragma omp critical(pgplot)
                {
                    cpgpage();
                    cpgmtxt("T", 3.0, 0.35, 0.5, "Final result after all substitutions");
                    // Display processed data without mask overlay; stats panels remain enabled (top/right)
                    plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, ii, NULL, 1, 1, NULL, flaggedChans);
                }
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
            double write_start = mono_time_sec();
            calcCompress(outDataT, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            transpose(outDataT, nchanBinned, nsampBinned, outData);
            applyCompress(outData, outRawData, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            
            int writeBack = m.writeBack, writeDastset = 1;

            if (writeBack) {
                int fits_status_wb = 0;
                #pragma omp critical(fits_writeback)
                {
                    writeBlock(wfptr, blocksPerRead, nchanBinned, nsampBinned, binnedBlockSize,
                               offsetBinned, scaleBinned, outRawData, m.naxis2, ii, &fits_status_wb);
                    if (fits_status_wb)
                    {
                        fits_report_error(stderr, fits_status_wb);
                        printf("Writing subint block %d, failed!\n", ii);
                    }
                    else
                    {
                        printf("Writing subint block %d, OK.\n", ii);
                    }
                }
            }

            if (writeDastset) {
                // Synchronous dataset writing (async writer disabled)
                int fits_status_ds = 0;
                writeFITSDataset(outRawData, scaleBinned, offsetBinned,
                                 nsampBinned, nchanBinned, ii, &m, &fits_status_ds);
            }
            double write_time = mono_time_sec() - write_start; if (write_time < 0) write_time = 0;
            printf("Write operations time: %.4f seconds\n", write_time);
        }

        int iter1 = ii + 1; // 1-based iteration count

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
        
    double loop_time = mono_time_sec() - loop_start; if (loop_time < 0) loop_time = 0;
        loop_total_time += loop_time;
        if (loop_time < loop_min_time) loop_min_time = loop_time;
        if (loop_time > loop_max_time) loop_max_time = loop_time;
    printf("Total loop %d time: %.4f seconds\n", ii, loop_time);
    printf("iteration %d done.\n\n", iter1);

    // toggle buffer for next iteration (not used in sync mode)
    // if (use_async_prefetch) useA = !useA;

    }

    // Stop writer threads and cleanup (disabled for sync I/O)
    // stop_writer_threads_and_join();
    // if (g_reader_running) {
    //     stop_reader_thread_prefetch();
    // }
    // Cleanup
    if (m.plot) {
        cpgend();
        if (g_last_plot_device[0]) {
            convert_ps_to_pdf(g_last_plot_device, 1);
        }
    }
    if (wfptr) { int st=0; fits_close_file(wfptr, &st); wfptr=NULL; }
    if (wfptr) { int st=0; FITS_IO_LOCK(); fits_close_file(wfptr, &st); FITS_IO_UNLOCK(); wfptr=NULL; }
    close_thread_read_fptr();
    free(freqArray);
    free(dsFreqArray);
    free(outData_all);
    free(outDataT_all);

    if (maskSets) {
        for (int t=0; t<maxScratchThreads; ++t) {
            freeIdentNSigmaMasks(&maskSets[t]);
        }
        free(maskSets);
    }
    free(flaggedChans_all);
    free(outRawData_all);
    // free(rawBufB);
    free(scale_all);
    free(offset_all);
    free(scaleBinned_all);
    free(offsetBinned_all);
    free(scaleRows_all);
    free(offsetRows_all);
    // free(sclBufB);
    // free(offBufB);
    free(rawToFloatArray_all);
    free(finalMedian_all);
    free(finalStd_all);
    free(subChanMed_medianBuf);
    free(subChanMed_tempDataBuf);
    free(identSubst_goodSamps);
    free(identSubst_randIdxs);
    free(identSubst_medTemp);
    free(inChanScratch);
    free(vs_time_means_all);
    free(vs_flag_time_all);
    free(fallback_chan_means_all);
    free(mask_chan2d_all);
    free(mask_CLFD_all);

    // Print loop timing summary (if any iteration executed)
    {
        int iterations_executed = numReads;
        if (iterations_executed > 0) {
            double loop_avg_time = loop_total_time / (double)iterations_executed;
            printf("Loop summary: %d iterations, total %.2f s, avg %.4f s, min %.4f s, max %.4f s\n",
                   iterations_executed, loop_total_time, loop_avg_time, loop_min_time, loop_max_time);
        }
    }

    // Write human-readable generation summary
    write_generation_summary(&m, sourceName,
                             blocksPerRead, nsampBinned, nchanBinned,
                             numReads, remainderRows,
                             NSigmaInChan, NSigmaOutChan);

    double global_end_time = mono_time_sec();
    printf("All done, time taken: %.2f seconds.\n", global_end_time - global_start_time);
    free(sourceName);
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