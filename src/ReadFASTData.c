#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>

#include "omp.h"
#include "cpgplot.h"
#include "fitsio.h"
#include "fftw3.h"

#include "ReadFASTData.h"
#include "identification.h"
#include "findStats.h"
#include "plot.h"
#include "cmd.h"
#include "transpose.h"
#include "psrPalett.h"
#include "cuda_acceleration.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

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

    fits_movnam_hdu(fptr, BINARY_TBL, "SUBINT", 0, &status);             // move to hdu by name
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

void convertToFloat(unsigned char *rawData, float *data, int size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        data[i] = (float)rawData[i];
    }
}

float gaussianSample(float mu, float sigma)
{
    // Generate two uniformly distributed random numbers in the range [0, 1)
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);

    // Box-Muller transformation to get a standard normal random variable
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);

    // Return the transformed variable with mean and standard deviation
    return mu + sigma * z0;
}


void getProfile(float *array, int nsamp, int nchan, float *freqProfile, float *timeProfile, int isTranspose)
{
    int i, j;

    if (isTranspose)
    {
        #pragma omp parallel for
        for (i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            for (j = 0; j < nsamp; j++)
            {
                sum += array[i * nsamp + j];
            }
            freqProfile[i] = sum / nsamp;
        }

        #pragma omp parallel for
        for (i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            for (j = 0; j < nchan; j++)
            {
                sum += array[j * nsamp + i];
            }
            timeProfile[i] = sum / nchan;
        }
    }
    else
    {
        #pragma omp parallel for
        for (i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            for (j = 0; j < nsamp; j++)
            {
                sum += array[j * nchan + i];
            }
            freqProfile[i] = sum / nsamp;
        }

        #pragma omp parallel for
        for (i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            for (j = 0; j < nchan; j++)
            {
                sum += array[i * nchan + j];
            }
            timeProfile[i] = sum / nchan;
        }
    }
}

void getProfileStd(float *array, int nsamp, int nchan, float *freqProfile, float *timeProfile, int isTranspose)
{
    int i, j;

    if (isTranspose)
    {
        #pragma omp parallel for
        for (i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;
            float *chanPtr = array + i * nsamp;

            for (j = 0; j < nsamp; j++)
            {
                float value = chanPtr[j];
                sum += value;
                sumSq += value * value;
            }
            float mean = sum / nsamp;
            float variance = (sumSq - 2 * mean * sum + nsamp * mean * mean) / nsamp;
            freqProfile[i] = sqrt(variance);
        }

        #pragma omp parallel for
        for (i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            for (j = 0; j < nchan; j++)
            {
                float value = array[j * nsamp + i];
                sum += value;
                sumSq += value * value;
            }
            float mean = sum / nchan;
            float variance = (sumSq - 2 * mean * sum + nchan * mean * mean) / nchan;
            timeProfile[i] = sqrt(variance);
        }
    }
    else
    {
        #pragma omp parallel for
        for (i = 0; i < nchan; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            for (j = 0; j < nsamp; j++)
            {
                float value = array[j * nchan + i];
                sum += value;
                sumSq += value * value;
            }
            float mean = sum / nsamp;
            float variance = (sumSq - 2 * mean * sum + nsamp * mean * mean) / nsamp;
            freqProfile[i] = sqrt(variance);
        }

        #pragma omp parallel for
        for (i = 0; i < nsamp; i++)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;
            float *sampPtr = array + i * nchan;

            for (j = 0; j < nchan; j++)
            {
                float value = sampPtr[j];
                sum += value;
                sumSq += value * value;
            }
            float mean = sum / nchan;
            float variance = (sumSq - 2 * mean * sum + nchan * mean * mean) / nchan;
            timeProfile[i] = sqrt(variance);
        }
    }
}

void calcCompress(float *data, int nchan, int nsamp, float *scale, float *offset)
{
    int ch;
    #pragma omp parallel for
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
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for collapse(2)
    for (i = 0; i < nchanBinned; i++) {
        for (j = 0; j < lenx; j++) {
            data[j * nchanBinned + i] = 
                data[j * nchanBinned + i] * scale[i] + offset[i];
        }
    }
}


/// @brief Read raw data block from FITS file including scale and offset
/// @param fptr FITS file pointer
/// @param blockIndex Index of the current subint BLOCK (ii in the main loop)
/// @param blocksPerRead Number of blocks to read per iteration
/// @param nchan Number of channels
/// @param blockSize Size of each data block
/// @param scale Output scale array (sized for nchan)
/// @param offset Output offset array (sized for nchan)
/// @param outRawData Output raw data buffer
/// @param status FITS status pointer
void readRawBlock(fitsfile *fptr, int blockIndex, int blocksPerRead, int nchan, int blockSize,
                 float *scale, float *offset, unsigned char *outRawData, int *status) {
    int col, nulval = 0, anynul = 0;

    // Read the first subint's scale and offset (assuming they're the same for all subints in this block)
    fits_get_colnum(fptr, CASESEN, "DAT_OFFS", &col, status);
    fits_read_col(fptr, TFLOAT, col, blockIndex * blocksPerRead + 1, 1, nchan, 
                 &nulval, offset, &anynul, status);

    fits_get_colnum(fptr, CASESEN, "DAT_SCL", &col, status);
    fits_read_col(fptr, TFLOAT, col, blockIndex * blocksPerRead + 1, 1, nchan, 
                 &nulval, scale, &anynul, status);

    // Read all data blocks
    int k;
    for (k = 0; k < blocksPerRead; k++) {
        fits_get_colnum(fptr, CASESEN, "DATA", &col, status);
        fits_read_col(fptr, TBYTE, col, blockIndex * blocksPerRead + k + 1, 1, blockSize, 
                     &nulval, outRawData + k * blockSize, &anynul, status);
    }
    
    if (*status) {
        fprintf(stderr, "Error reading block %d\n", blockIndex);
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

        fits_get_colnum(fptr, CASEINSEN, "DAT_SCL", &col1, status);
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
    snprintf(filename, sizeof(filename), "%s/%s_%04d%02d%02d_block%d.fits",
             m->datasetPath, extractSourceName(m->filename),
             t->tm_year+1900, t->tm_mon+1, t->tm_mday, blockIndex);

    fits_create_file(&fptr, filename, status);
    if (*status) {
        fits_report_error(stderr, *status);
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
}
int main(int argc, char *argv[])
{
    setup_openmp(20);
    
    Metadata m;

    int status = parseCommandLineArguments(argc, argv, &m);
    if (status != 0) {
        return status;
    }
    readMetadata(&m);

    // Initialize CUDA if requested by user
    m.cudaReady = 0; // Default: CUDA not ready
    if (m.enableCuda) {
        if (cuda_isAvailable()) {
            if (cuda_init() == 0) {
                printf("CUDA acceleration enabled.\n");
                m.cudaReady = 1; // CUDA is ready to use
            } else {
                printf("CUDA initialization failed, using CPU only.\n");
                m.enableCuda = 0; // Disable CUDA for this session
            }
        } else {
            fprintf(stderr, "Error: You requested CUDA acceleration, but CUDA is not available on this system.\n");
            fprintf(stderr, "Please run without --enableCuda option and check CUDA installation.\n");
            return -1;
        }
    } else {
        printf("Message: --enableCuda option not found, using CPU. \n");
    }

    fitsfile *fptr = NULL;
    int fits_status = 0;
    fits_open_file(&fptr, m.filename, READWRITE, &fits_status);
    fits_movnam_hdu(fptr, BINARY_TBL, "SUBINT", 0, &fits_status);

    // Setup PGPLOT output
    if (m.plot)
    {
        char *sourceName = extractSourceName(m.filename);
        size_t requiredSize = strlen(m.datasetPath) + strlen(sourceName) + 64;
        char *saveName = malloc(requiredSize);
        memset(saveName, 0, requiredSize);
        snprintf(saveName, requiredSize, "%s/%s_%.2f_%d_%d.ps/VCPS",
                 m.datasetPath, sourceName, 0.0, m.binFactorTime, m.binFactorFreq);
        // Initialize PGPLOT graphics system, set output device to PostScript file
        // Parameters: device count=1, filename with path and device type, subplot layout 2x3
        cpgbeg(1, saveName, 2, 5);
        free(sourceName);
        free(saveName);
    }

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
    // unsigned char *outRawData = malloc(sizeof(unsigned char) * nchanBinned * nsampBinned);
    // unsigned char *outRawDataT = malloc(sizeof(unsigned char) * nchanBinned * nsampBinned);
    unsigned char *outRawData = malloc(sizeof(unsigned char) * nchan * nsamp);
    unsigned char *outRawDataT = malloc(sizeof(unsigned char) * nchan * nsamp);
    float *outData = fftwf_malloc(sizeof(float) * nchanBinned * nsampBinned);
    float *outDataT = fftwf_malloc(sizeof(float) * nchanBinned * nsampBinned);

    // Allocate RFI mask buffers
    int *mask_chanRFI = calloc(nsampBinned * nchanBinned, sizeof(int));
    int *mask_ST = calloc(nsampBinned * nchanBinned, sizeof(int));
    int *horizontalMask = (int *)calloc(nsampBinned * nchanBinned, sizeof(int));
    int *verticalMask = (int *)calloc(nsampBinned * nchanBinned, sizeof(int));
    int *globalMask = (int *)calloc(nsampBinned * nchanBinned, sizeof(int));

    float *finalMedian = malloc(sizeof(float) * nsampBinned * nchanBinned);
    float *finalStd = malloc(sizeof(float) * nsampBinned * nchanBinned);
    float *scale = malloc(sizeof(float) * nchan);  // Fix: Use original nchan size
    float *offset = malloc(sizeof(float) * nchan); // Fix: Use original nchan size
    float *scaleBinned = malloc(sizeof(float) * nchanBinned); // New: for downsampled data
    float *offsetBinned = malloc(sizeof(float) * nchanBinned); // New: for downsampled data
    
    float *rawToFloatArray = NULL;
    int needDownsamp = (binFactorTime * binFactorFreq != 1);
    if (needDownsamp) {
        rawToFloatArray = (float *)malloc(sizeof(float) * nsamp * nchan);
    }   

    int numiter = 0;
    int ii;
    int writeMasks = 1;
    for (ii = 0; ii < numReads; ii++)
    {
        printf("Processing block %d of %d, %d subints per block, %.3f%% done.\n", ii, numReads, blocksPerRead, (ii * 100.0f / numReads));
        // Read a raw data block of `blocksPerRead` subints with its scale and offset
        readRawBlock(fptr, ii, blocksPerRead, nchan, blockSize, scale, offset, outRawData, &fits_status);
        
        // Convert raw data to float and apply scale/offset BEFORE downsampling
        if (needDownsamp) {
            convertToFloat(outRawData, rawToFloatArray, nsamp * nchan);
            applyScaleOffset(rawToFloatArray, scale, offset, nsamp, nchan);
            downsamp2D(rawToFloatArray, nsamp, nchan, outData, binFactorTime, binFactorFreq, 0);
            // Create downsampled scale/offset arrays for later use
            downsamp1D(scale, nchan, binFactorFreq, scaleBinned);
            downsamp1D(offset, nchan, binFactorFreq, offsetBinned);
        } else {
            convertToFloat(outRawData, outData, nsamp * nchan);
            applyScaleOffset(outData, scale, offset, nsamp, nchan);
            // For no downsampling case, just copy the arrays
            memcpy(scaleBinned, scale, sizeof(float) * nchanBinned);
            memcpy(offsetBinned, offset, sizeof(float) * nchanBinned);
        }
        
        // CUDA-accelerated transpose (with fallback to CPU)
        printf("Performing matrix transpose (%d x %d)...\n", nsampBinned, nchanBinned);
        if (m.cudaReady) {
            double start_time = omp_get_wtime();
            cuda_transpose(outData, outDataT, nsampBinned, nchanBinned);
            double cuda_time = omp_get_wtime() - start_time;
            printf("CUDA transpose completed in %.4f seconds\n", cuda_time);
        } else {
            double start_time = omp_get_wtime();
            transpose(outData, nsampBinned, nchanBinned, outDataT);
            double cpu_time = omp_get_wtime() - start_time;
            printf("CPU transpose completed in %.4f seconds\n", cpu_time);
        }
        
        if (m.generateMasks)
        {
            memset(mask_ST, 0, sizeof(int) * nsampBinned * nchanBinned);

            // Plot the unprocessed raw data
            if (m.plot)
            { 
                cpgpage(); // Create new graphics page
                cpgmtxt("T", 3.0, 0.35, 0.5, "Raw Data");
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL);
            }



            int subtractChanMed = 1;
            if (subtractChanMed)
            {
                subtractChannelMedians(outDataT, nsampBinned, nchanBinned);
            }
            if (m.plot)
            {
                cpgpage();
                char text2[100];
                snprintf(text2, sizeof(text2), "Result after subtracting channel median");
                cpgmtxt("T", 3.5, 0.5, 0.5, text2);
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL);
            }



            float NSigmaInChan = 2.0f;
            float NSigmaOutChan = 3.0f;
            if (m.doSubstitution)
            {
                identSubstNSigma(outDataT, nsampBinned, nchanBinned, NSigmaInChan, NSigmaOutChan, ii, m.plot,
                                 horizontalMask, verticalMask, globalMask, finalMedian, finalStd, m.cudaReady);
            }
            if (writeMasks)
            {
                char mask_Subst_filename[256];
                sprintf(mask_Subst_filename, "%smask_Subst_%d.png", m.datasetPath, ii);
                writeIndexMaskPNG(globalMask, nsampBinned, nchanBinned, mask_Subst_filename);
            }
            if (m.plot)
            { // Plot result after NSigma substitution
                cpgpage();
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, globalMask);
            }



            float timesOfSigma = 20.0f;
            // float timesOfSigma = 8.0f;
            int M_len = 6;
            int win_samp = 3, win_chan = 3;
            float thrup = 0.5f, thrdown = 0.5f;
            if (m.doSumThreshold)
            {
                // sumthreshold_2d(outDataT, nsampBinned, nchanBinned,
                //                 mask_chanRFI, mask_ST, timesOfSigma, M_len);
                // if (m.cudaReady) {
                //     printf("Using CUDA-accelerated binarySIR filtering...\n");
                //     cuda_binarySIR(mask_ST, nsampBinned, nchanBinned, win_samp, win_chan, thrup, thrdown);
                // } else {
                //     binarySIR(mask_ST, nsampBinned, nchanBinned, win_samp, win_chan, thrup, thrdown);
                // }
                // binarySIR(mask_ST, nsampBinned, nchanBinned, win_samp, win_chan, thrup, thrdown);
            }
            if (writeMasks)
            {
                char mask_ST_filename[256];
                sprintf(mask_ST_filename, "%smask_ST_%d.png", m.datasetPath, ii);
                writeIndexMaskPNG(mask_ST, nsampBinned, nchanBinned, mask_ST_filename);
            }
            if (m.plot)
            {
                cpgpage();
                char text3[100];
                snprintf(text3, sizeof(text3), "Result of SumThreshold RFI detection with chi=%.2f", timesOfSigma);
                cpgmtxt("T", 3.5, 0.5, 0.5, text3);
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, mask_ST);
            }
            if (m.doSumThreshold)
            {
                substPixels2D(outDataT, nsampBinned, nchanBinned, mask_ST);
            }
            if (m.plot)
            {
                cpgpage();
                cpgmtxt("T", 3.5, 0.5, 0.5, "Result after pixel substitution");
                // 顶部和右侧都显示标准差，无掩码
                plotTimeFreqSED(&m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, NULL);
                cpgpage();
            }            
        }

        if (m.write)
        {
            calcCompress(outDataT, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            transpose(outDataT, nchanBinned, nsampBinned, outData);
            applyCompress(outData, outRawData, nchanBinned, nsampBinned, scaleBinned, offsetBinned);
            
            int writeBack = 0, writeDastset = 1;
            
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
                writeFITSDataset(outRawData, scaleBinned, offsetBinned, nsampBinned, nchanBinned, ii, &m, &fits_status);
            }
        }


        numiter++;
        if (numiter >= 2) {
            m.plot = 0; 
            return 0;
        }
        // if (numiter >= 20)   
        // {
        //     printf("Debug mode: Reached maximum iterations of 20, exiting.\n");
        //     return 0;
        // }

        // if (m.savePlot)
        // {
        //     // Optional: convert_ps_to_png(saveName);
        // }
        // else
        // {
        //     // wait_for_mouse_click();
        // }

        /*
            // Example usage of channel-specific mask extraction:
            
            // Method 1: Extract specific channels by indices
            int targetChannels[] = {10, 15, 20, 25}; // Example: channels with known interference
            int numTargetChannels = 4;
            int *channelSpecificMask = (int *)calloc(nsampBinned * nchanBinned, sizeof(int));
            extractChannelMask(globalMask, channelSpecificMask, nsampBinned, nchanBinned, 
                             targetChannels, numTargetChannels, 1); // isTranspose=1 for transposed data
            
            // Method 2: Extract a range of channels
            int *rangeSpecificMask = (int *)calloc(nsampBinned * nchanBinned, sizeof(int));
            extractChannelRangeMask(globalMask, rangeSpecificMask, nsampBinned, nchanBinned,
                                  10, 30, 1); // Extract channels 10-30
            
            // Write channel-specific masks
            if (writeMasks) {
                char channelMask_filename[256];
                sprintf(channelMask_filename, "%smask_Channels_%d.png", m.datasetPath, ii);
                writeIndexMaskPNG(channelSpecificMask, nsampBinned, nchanBinned, channelMask_filename);
                
                sprintf(channelMask_filename, "%smask_Range_%d.png", m.datasetPath, ii);
                writeIndexMaskPNG(rangeSpecificMask, nsampBinned, nchanBinned, channelMask_filename);
            }
            
            // Clean up
            free(channelSpecificMask);
            free(rangeSpecificMask);
            */

    }

    // Cleanup
    // Close PGPLOT graphics system, complete all graphics file writing and resource cleanup
    if (m.plot) cpgend();
    fits_close_file(fptr, &fits_status);
    free(freqArray);
    free(dsFreqArray);
    free(outData);
    free(outDataT);
    free(mask_ST);
    free(mask_chanRFI);
    free(horizontalMask);
    free(verticalMask);
    free(globalMask);
    free(outRawData);
    free(outRawDataT);
    free(scale);
    free(offset);
    free(scaleBinned);
    free(offsetBinned);
    if (needDownsamp) {
        free(rawToFloatArray);
    }
    free(finalMedian);
    free(finalStd);
    
    // Cleanup CUDA resources
    if (m.cudaReady) {
        cuda_cleanup();
        printf("CUDA resources cleaned up.\n");
    }
    
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