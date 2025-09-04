#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include <png.h>
#include <omp.h>
#include "cpgplot.h"

// GSL headers for nonlinear fitting
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include "ReadFASTData.h"
#include "findStats.h"
#include "identification.h"
#include "transpose.h"
#include "plot.h"
#include "cuda_acceleration.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

float gaus(float x, float med, float sigma)
{
    return expf(-(x - med) * (x - med) / (2 * sigma * sigma)) / (sqrtf(2 * PI) * sigma);
}

// New function for amplitude-included Gaussian
float gaus_with_amplitude(float x, float amplitude, float mean, float sigma)
{
    return amplitude * expf(-(x - mean) * (x - mean) / (2 * sigma * sigma));
}

/**
 * @brief Residual function for two-parameter Gaussian fitting (fixed amplitude)
 * Parameters: [0]=mean, [1]=sigma
 * Data is passed as void pointer to array of [x_array, y_array, &n, &fixed_amplitude]
 */
int gaussian_residual_f_fixed_amp(const gsl_vector *params, void *data, gsl_vector *f) {
    void **data_array = (void **)data;
    float *x = (float *)data_array[0];
    float *y = (float *)data_array[1];
    int n = *(int *)data_array[2];
    float fixed_amplitude = *(float *)data_array[3];  // Fixed amplitude from histogram max
    
    double mu = gsl_vector_get(params, 0);     // mean
    double sigma = gsl_vector_get(params, 1);  // standard deviation
    
    // Prevent negative or zero sigma
    if (sigma <= 0) {
        sigma = 1e-6;
    }
    
    int i;
    for (i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double model = fixed_amplitude * exp(-(xi - mu) * (xi - mu) / (2.0 * sigma * sigma));
        // Residual = observed - predicted
        gsl_vector_set(f, i, yi - model);
    }
    
    return GSL_SUCCESS;
}

/**
 * @brief Two-parameter Gaussian fitting using GSL Levenberg-Marquardt (fixed amplitude)
 * @param x X data points
 * @param y Y data points
 * @param n Number of data points
 * @param fixed_amplitude Fixed amplitude value (e.g., histogram max bin value)
 * @param fitted_mu Output: fitted mean
 * @param fitted_sigma Output: fitted standard deviation
 * @return 1 for success, 0 for failure
 */
int gsl_gaussian_fit(float *x, float *y, int n, float fixed_amplitude,
    float *fitted_mu, float *fitted_sigma)
{
    // Essential check: sufficient data points for 2-parameter fitting
    if (n < 3) {
        printf("GSL Gaussian fit (fixed amp): Insufficient data points (%d < 3)\n", n);
        return 0;
    }
    
    // Find initial parameter estimates
    float min_x = x[0], max_x = x[0];
    float sum_x_weighted = 0.0f, sum_y = 0.0f;
    
    int i;
    for (i = 0; i < n; i++) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        sum_x_weighted += x[i] * y[i];
        sum_y += y[i];
    }
    
    // Calculate initial parameters
    float initial_mu = (sum_y > 0) ? sum_x_weighted / sum_y : (min_x + max_x) / 2.0f;
    
    float sum_weighted_var = 0.0f;
    for (i = 0; i < n; i++) {
        float dx = x[i] - initial_mu;
        sum_weighted_var += y[i] * dx * dx;
    }
    float initial_sigma = (sum_y > 0) ? sqrtf(sum_weighted_var / sum_y) : (max_x - min_x) / 6.0f;
    
    // Basic safety bounds
    if (initial_sigma <= 0) initial_sigma = (max_x - min_x) / 10.0f;
    
    // Set up GSL fitting
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();    
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    
    // Prepare data as pointer array for function parameters
    void *data_ptrs[4];
    data_ptrs[0] = x;
    data_ptrs[1] = y;
    data_ptrs[2] = &n;
    data_ptrs[3] = &fixed_amplitude;
    
    // Set up function
    fdf.f = gaussian_residual_f_fixed_amp;
    fdf.df = NULL;   // Use numerical differentiation
    fdf.fvv = NULL;  // Use numerical second derivatives
    fdf.n = n;       // number of data points
    fdf.p = 2;       // number of parameters (mu, sigma)
    fdf.params = data_ptrs;
    
    // Allocate workspace
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, 2);
    
    // Initial parameter vector (only mu and sigma)
    gsl_vector *params_init = gsl_vector_alloc(2);
    gsl_vector_set(params_init, 0, initial_mu);
    gsl_vector_set(params_init, 1, initial_sigma);
    
    // Essential check: GSL initialization
    int status = gsl_multifit_nlinear_init(params_init, &fdf, w);
    if (status != GSL_SUCCESS) {
        printf("GSL Gaussian fit (fixed amp): Initialization failed\n");
        gsl_vector_free(params_init);
        gsl_multifit_nlinear_free(w);
        return 0;
    }
    
    // Iterate to find solution
    int info;
    const int max_iterations = 100;
    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 1e-8;
    
    int iter;
    for (iter = 0; iter < max_iterations; iter++) {
        status = gsl_multifit_nlinear_iterate(w);
        
        if (status == GSL_ENOPROG) break;
        if (status != GSL_SUCCESS) break;
        
        // Test for convergence
        status = gsl_multifit_nlinear_test(xtol, gtol, ftol, &info, w);
        if (status == GSL_SUCCESS) break;
    }
    
    // Get final results
    gsl_vector *final_params = gsl_multifit_nlinear_position(w);
    
    *fitted_mu = (float)gsl_vector_get(final_params, 0);
    *fitted_sigma = (float)gsl_vector_get(final_params, 1);
    
    // Essential check: validate final results
    int success = (*fitted_sigma > 0) ? 1 : 0;
    
    // Clean up
    gsl_vector_free(params_init);
    gsl_multifit_nlinear_free(w);
    
    return success;
}

float simple_curve_fit(float *x, float *y, int n, float med)
{
    int i;
    float best_sigma = 1.0f;
    float min_error = FLT_MAX;
    float sigma;

    for (sigma = 0.1f; sigma <= 5.0f; sigma += 0.1f)
    {
        float error = 0.0f;
        for (i = 0; i < n; i++)
        {
            float diff = y[i] - gaus(x[i], med, sigma);
            error += diff * diff;
        }

        if (error < min_error)
        {
            min_error = error;
            best_sigma = sigma;
        }
    }

    return best_sigma;
}

/**
 * Subtract the median value from each frequency channel
 * @param data: Input data array (nsamp * nchan)
 * @param nsamp: Number of time samples per channel
 * @param nchan: Number of frequency channels
 */
void subtractChannelMedians(float *data, int nsamp, int nchan)
{
    printf("=== Subtracting channel medians from data ===\n");
    
    float *channel_medians = (float *)malloc(nchan * sizeof(float));
    float *temp_data = (float *)malloc(nsamp * sizeof(float));
    
    // Calculate median for each channel
    int i, j;
    for (i = 0; i < nchan; i++) {
        // Copy channel data for median calculation
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        channel_medians[i] = median(temp_data, nsamp);
    }
    
    // Subtract median from each channel
    #pragma omp parallel for
    for (i = 0; i < nchan; i++) {
        for (j = 0; j < nsamp; j++) {
            data[i * nsamp + j] -= channel_medians[i];
        }
    }
    
    printf("Channel median subtraction completed for %d channels\n", nchan);
    
    // Clean up temporary arrays
    free(channel_medians);
    free(temp_data);
}

float ksigma_1d(float *data, int n, int bins, float *hist, float *x_val, float *median_temp)
{
    if (n <= 0 || bins <= 0 || !data || !hist || !x_val || !median_temp) {
        return 0.0f;
    }

    int i;
    memcpy(median_temp, data, n * sizeof(float));
    float med = median(median_temp, n);

    // find range of data
    float min_val = data[0], max_val = data[0];
    for (i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    // Handle case where all values are the same
    if (min_val == max_val) {
        memset(hist, 0, bins * sizeof(float));
        hist[bins/2] = n;
        for (i = 0; i < bins; i++) {
            x_val[i] = min_val;
        }
        return 0.0f; // or some default value
    }

    float bin_width = (max_val - min_val) / bins;
    if (bin_width <= 0.0f) {
        memset(hist, 0, bins * sizeof(float));
        hist[bins/2] = n;
        for (i = 0; i < bins; i++) {
            x_val[i] = min_val;
        }
        return 0.0f;
    }

    // Initialize hist
    memset(hist, 0, bins * sizeof(float));

    // Fill histogram with bounds checking
    for (i = 0; i < n; i++) {
        float normalized = (data[i] - min_val) / bin_width;
        int bin = (int)normalized;
        if (bin < 0) bin = 0;
        else if (bin >= bins) bin = bins - 1;
        hist[bin] += 1.0f;
    }

    // Normalize hist
    float sum = 0.0f;
    for (i = 0; i < bins; i++) {
        sum += hist[i];
    }
    if (sum > 0.0f) {
        for (i = 0; i < bins; i++) {
            hist[i] /= (sum * bin_width);
        }
    }

    // calc x_val (center of every bin)
    for (i = 0; i < bins; i++) {
        x_val[i] = min_val + (i + 0.5f) * bin_width;
    }

    // fit gaussian
    float sigma = simple_curve_fit(x_val, hist, bins, med);

    return sigma;
}

float ksigma_2d(const float *dataT, const int *mask_chanRFI, int nsamp, int nchan)
{
    // Flatten the unmasked data
    int i;
    int total_size = nsamp * nchan;
    float *unmasked_data = (float *)malloc(total_size * sizeof(float));
    int unmasked_count = 0;
    for (i = 0; i < total_size; i++)
    {
        if (mask_chanRFI == NULL || mask_chanRFI[i] == 0)
        {
            unmasked_data[unmasked_count++] = dataT[i];
        }
    }

    // Use `ksigma_1d` on the flattened array
    int bins = 50; // Number of bins for histogram
    float *hist = (float *)calloc(bins, sizeof(float));
    float *x_val = (float *)malloc(bins * sizeof(float));
    float *median_temp = (float *)malloc(unmasked_count * sizeof(float));
    float sigma = ksigma_1d(unmasked_data, unmasked_count, bins, hist, x_val, median_temp);
    
    free(unmasked_data);
    free(hist);
    free(x_val);
    free(median_temp);
    return sigma;
}

// Original function name kept, with optimized memory management
void sumthreshold_1d(
    const float *data,
    int length,
    int *mask,
    float chi_1,
    int M_len,
    float *temp_data,  // Pre-allocated temp buffer
    int *local_mask,   // Pre-allocated mask buffer
    float *M,          // Pre-allocated M array
    float *chi_i)      // Pre-allocated chi_i array
{
    const float p = 1.5f;
    const int eta_len = 1;
    const float eta_i[] = {1.0f};

    // Pre-compute M and chi_i values
    int i, j, e, m;
    for (i = 0; i < M_len; i++) {
        M[i] = powf(2.0f, (float)i);
        chi_i[i] = chi_1 / powf(p, log2f(M[i]));
    }

    memcpy(temp_data, data, length * sizeof(float));
    memset(local_mask, 0, length * sizeof(int));

    // Main thresholding logic
    for (e = 0; e < eta_len; e++) {
        float current_eta = eta_i[e];
        for (m = 0; m < M_len; m++) {
            int window = (int)M[m];
            float threshold = chi_i[m] / current_eta;

            // Window processing
            for (i = 0; i <= length - window; i++) {
                float sum = 0.0f;
                int count = 0;

                // Calculate sum and count
                for (j = 0; j < window; j++) {
                    if (!mask[i + j]) {
                        sum += fabsf(temp_data[i + j]);
                        count++;
                    }
                }

                // Apply threshold
                if (count > 0 && (sum / count) > threshold) {
                    for (j = 0; j < window; j++) {
                        local_mask[i + j] = 1;
                    }
                }
            }
        }
    }

    // Merge masks
    for (i = 0; i < length; i++) {
        mask[i] |= local_mask[i];
    }
}

// Original 2D function with optimized memory allocation
void sumthreshold_2d(
    const float *dataT,
    int nsamp,
    int nchan,
    int *mask_chanRFI,
    int *mask,
    float timesOfSigma,
    int M_len)
{
    // Determine max dimension needed
    int max_dim = (nsamp > nchan) ? nsamp : nchan;
    
    // Allocate reusable buffers
    float *temp_data_1d = (float*)malloc(max_dim * sizeof(float));
    int *local_mask_1d = (int*)malloc(max_dim * sizeof(int));
    float *M = (float*)malloc(M_len * sizeof(float));
    float *chi_i = (float*)malloc(M_len * sizeof(float));
    
    // Original 2D buffers
    float *temp_dataT = (float *)malloc(nsamp * nchan * sizeof(float));
    int *temp_maskT = (int *)malloc(nsamp * nchan * sizeof(int));

    // Copy input data
    memcpy(temp_dataT, dataT, nsamp * nchan * sizeof(float));
    memset(mask, 0, nsamp * nchan * sizeof(int));
    memset(temp_maskT, 0, nsamp * nchan * sizeof(int));

    // Calculate channel statistics
    float means[nchan], stds[nchan];
    int i, j;
    for (i = 0; i < nchan; i++) {
        findMeanStd(&temp_dataT[i * nsamp], nsamp, &means[i], &stds[i]);
    }

    // Global normalization
    float global_mean, global_std;
    findMeanStd(temp_dataT, nsamp * nchan, &global_mean, &global_std);
    float chi_1 = timesOfSigma * ksigma_2d(temp_dataT, mask_chanRFI, nsamp, nchan);

    // Normalize data
    #pragma omp parallel for collapse(2)
    for (j = 0; j < nchan; j++) {
        for (i = 0; i < nsamp; i++) {
            temp_dataT[j * nsamp + i] = (temp_dataT[j * nsamp + i] - global_mean) / (global_std + 1e-6f);
        }
    }

    // Time-axis processing with optimized 1D
    #pragma omp parallel for
    for (j = 0; j < nchan; j++) {
        sumthreshold_1d(&temp_dataT[j * nsamp], nsamp, &mask[j * nsamp], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Transpose for frequency processing
    float *transposed_data = (float *)malloc(nsamp * nchan * sizeof(float));
    transpose(temp_dataT, nchan, nsamp, transposed_data);

    // Frequency-axis processing
    #pragma omp parallel for
    for (i = 0; i < nsamp; i++) {
        sumthreshold_1d(&transposed_data[i * nchan], nchan, &temp_maskT[i * nchan], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Merge masks
    #pragma omp parallel for collapse(2)
    for (i = 0; i < nsamp; i++) {
        for (j = 0; j < nchan; j++) {
            mask[j * nsamp + i] |= temp_maskT[i * nchan + j];
        }
    }

    // Cleanup
    free(temp_dataT);
    free(temp_maskT);
    free(transposed_data);
    free(temp_data_1d);
    free(local_mask_1d);
    free(M);
    free(chi_i);
}

void writeIndexMaskPNG(int *mask, int nsamp, int nchan, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

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
            // Convert mask value to byte
            float val = (float)mask[i * nsamp + j];
            row_pointers[nchan - 1 - i][j] = (png_byte)(val * 255.0f);
        }
    }

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

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

/// @brief Substitute masked elements in a 1D array with random samples from unmasked elements.
/// Core implementation that handles the actual pixel substitution logic.
/// @param data Pointer to the data array.
/// @param size Number of elements in the array.
/// @param mask Mask array indicating which elements are masked (1 for masked, 0 for good).
/// @param goodSamps Pre-allocated array of size `size` to hold indices of unmasked elements.
/// @param randIdx Pre-allocated array of size `size` to hold random indices for replacement.
void substPixels(float *data, int size, int *mask, int *goodSamps, int *randIdx) {
    int i, goodCnt = 0;
    
    // Collect indices of good samples
    for (i = 0; i < size; i++) {
        if (!mask[i]) {
            goodSamps[goodCnt] = i;
            goodCnt++;
        }
    }
    
    if (goodCnt == 0) {
        // No good samples found, nothing to substitute
        return;
    }

    // Prepare random indices for replacement
    // Use thread-safe random number generation with unique seed per thread
    unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 1000 + size);
    for (i = 0; i < size; i++) {
        randIdx[i] = rand_r(&seed) % goodCnt;
    }

    // Perform substitution
    for (i = 0; i < size; i++) {
        if (mask[i]) {
            data[i] = data[goodSamps[randIdx[i]]];
        }
    }
}

/// @brief Substitute masked pixels in each channel with random samples from good pixels in the same channel.
/// This is a wrapper function that uses the core 1D implementation for each channel.
/// @param data Data array to be processed, time samples from same channel are stored contiguously.
/// @param nsamp Number of time samples in each channel.
/// @param nchan Number of frequency channels.
/// @param mask Mask array indicating which pixels are masked (1 for masked, 0 for good).
void substPixels2D(float *data, int nsamp, int nchan, int *mask)
{
    int i;
    
    // Process each channel separately using the core 1D implementation
    // Each thread will have its own private temporary arrays
    #pragma omp parallel for private(i)
    for (i = 0; i < nchan; i++)
    {
        // Allocate thread-private temporary arrays
        int *good_samples = (int *)malloc(nsamp * sizeof(int));
        int *random_indices = (int *)malloc(nsamp * sizeof(int));
        
        // Check memory allocation
        if (!good_samples || !random_indices) {
            fprintf(stderr, "Error: Memory allocation failed in thread %d\n", omp_get_thread_num());
            free(good_samples);
            free(random_indices);
            continue; // Skip this channel and continue with others
        }

        int chan_offset = i * nsamp;
        substPixels(data + chan_offset, nsamp, mask + chan_offset,
                            good_samples, random_indices);
        
        // Free thread-private arrays
        free(good_samples);
        free(random_indices);
    }
}



void binarySIR(
    int *mask, int nsamp, int nchan,
    int win_samp, int win_chan, float thr_up, float thr_down) 
{
    if (((win_samp | win_chan) & 1) == 0) return;
    
    // Count pixels before filtering
    int pixelsBefore = 0;
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (mask[idx] != 0) pixelsBefore++;
    }
    
    const int rad_samp = win_samp / 2;
    const int rad_chan = win_chan / 2;
    int i, j, di, dj;

    #pragma omp parallel for collapse(2)
    for (i = 0; i < nsamp; i++) {
        for (j = 0; j < nchan; j++) {
            int count = 0, total = 0;
            for (dj = -rad_chan; dj <= rad_chan; dj++) {
                int jj = j + dj;
                if (jj < 0 || jj >= nchan) continue;
                for (di = -rad_samp; di <= rad_samp; di++) {
                    int ii = i + di;
                    if (ii < 0 || ii >= nsamp) continue;
                    count += (mask[jj * nsamp + ii] != 0);
                    total++;
                }
            }
            if (total > 0) {
                float ratio = (float)count / total;
                if (ratio >= thr_up) mask[j * nsamp + i] = 1;
                else if (ratio < thr_down) mask[j * nsamp + i] = 0;
            }
        }
    }
    
    // Count pixels after filtering and report statistics
    int pixelsAfter = 0;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (mask[idx] != 0) pixelsAfter++;
    }
    
    printf("binarySIR filtering statistics:\n");
    printf("  - Window size: %dx%d (samples x channels)\n", win_samp, win_chan);
    printf("  - Thresholds: up=%.3f, down=%.3f\n", thr_up, thr_down);
    printf("  - Pixels before: %d/%d (%.4f%%)\n", 
           pixelsBefore, nsamp*nchan, (float)pixelsBefore/(nsamp*nchan)*100);
    printf("  - Pixels after: %d/%d (%.4f%%)\n", 
           pixelsAfter, nsamp*nchan, (float)pixelsAfter/(nsamp*nchan)*100);
    printf("  - Filtered out: %d pixels (%.4f%%)\n", 
           pixelsBefore - pixelsAfter, (float)(pixelsBefore - pixelsAfter)/(nsamp*nchan)*100);
    printf("  - Reduction ratio: %.2fx\n", 
           pixelsBefore > 0 ? (float)pixelsBefore/pixelsAfter : 0.0f);
}

void flagChannelsByMeanOutliers(float *data, int nsamp, int nchan, int *horizontalMask,
                               float *channel_means, float *channel_means_temp)
{
    int i, j;
    
    // Calculate mean for each channel
    for (i = 0; i < nchan; i++)
    {
        float channel_mean, channel_std;
        findMeanStd(data + i * nsamp, nsamp, &channel_mean, &channel_std);
        channel_means[i] = channel_mean;
    }
    
    // Calculate statistics of channel means
    float mean_mean, mean_std;
    findMeanStd(channel_means, nchan, &mean_mean, &mean_std);
    memcpy(channel_means_temp, channel_means, nchan * sizeof(float));
    float mean_median = median(channel_means_temp, nchan);
    float mean_mad = mad(channel_means, nchan);
    findMeanStd(channel_means, nchan, NULL, &mean_std);
    
    // Define bounds for acceptable channel mean values (20 * MAD threshold)
    // float mean_lower_bound = mean_median - 0.8f * mean_mad;
    // float mean_upper_bound = mean_median + 0.8f * mean_mad;
    float mean_lower_bound = mean_median - 3.0f * mean_mad;
    float mean_upper_bound = mean_median + 3.0f * mean_mad;
    
    // Flag channels whose mean is outside acceptable range
    for (i = 0; i < nchan; i++)
    {
        if (channel_means[i] < mean_lower_bound || channel_means[i] > mean_upper_bound)
        {
            // printf("Flagging channel %d with mean %.2f outside bounds [%.2f, %.2f]\n",
            //        i, channel_means[i], mean_lower_bound, mean_upper_bound);
            // Mark entire channel as bad
            for (j = 0; j < nsamp; j++)
            {
                horizontalMask[i * nsamp + j] = 1;
            }
        }
    }
}


/// @brief Flag channels based on their standard deviation statistics using MAD-based outlier detection
/// @param data Input data array (nsamp * nchan)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param horizontalMask Output mask to mark flagged channels
/// @param channel_stds Pre-allocated array to store channel standard deviations
/// @param channel_stds_temp Pre-allocated temporary array for median calculation
void flagChannelsByStdOutliers(float *data, int nsamp, int nchan, int *horizontalMask,
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold)
{
    int i, j;
    
    // Calculate standard deviation for each channel
    for (i = 0; i < nchan; i++)
    {
        float channel_mean, channel_std;
        findMeanStd(data + i * nsamp, nsamp, &channel_mean, &channel_std);
        channel_stds[i] = channel_std;
    }
    
    // Calculate statistics of channel standard deviations
    float std_mean, std_std;
    findMeanStd(channel_stds, nchan, &std_mean, &std_std);
    memcpy(channel_stds_temp, channel_stds, nchan * sizeof(float));
    float std_median = median(channel_stds_temp, nchan);
    float std_mad = mad(channel_stds, nchan);


    
    // Define bounds for acceptable channel std values (20 * MAD threshold)
    // float std_lower_bound = std_median - 2.5f * std_mad;
    // float std_upper_bound = std_median + 2.5f * std_mad;
    float std_lower_bound = std_median - channel_std_threshold * std_mad;
    float std_upper_bound = std_median + channel_std_threshold * std_mad;
    
    // Flag channels whose std is outside acceptable range
    for (i = 0; i < nchan; i++)
    {
        if (channel_stds[i] < std_lower_bound || channel_stds[i] > std_upper_bound)
        {
            // Mark entire channel as bad
            for (j = 0; j < nsamp; j++)
            {
                horizontalMask[i * nsamp + j] = 1;
            }
        }
    }
}



/// @brief Normalize data by subtracting channel median and dividing by channel standard deviation
/// @param data Input data array (nsamp * nchan)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param finalMedian Output array to store channel medians
/// @param finalStd Output array to store channel standard deviations
/// @param median_temp Temporary array for median calculation
void normalizeChannelData(float *data, int nsamp, int nchan, 
                         float *finalMedian, float *finalStd, float *median_temp)
{
    int i, j;
    
    // Copy data for median calculation
    memcpy(median_temp, data, nsamp * nchan * sizeof(float));

    // Calculate median and standard deviation for each channel
    #pragma omp parallel for
    for (i = 0; i < nchan; i++)
    {
        finalMedian[i] = median(median_temp + i * nsamp, nsamp);
        findMeanStd(data + i * nsamp, nsamp, NULL, &finalStd[i]);
    }
    
    // Normalize each channel: (data - median) / std
    #pragma omp parallel for
    for (i = 0; i < nchan; i++)
    {
        for (j = 0; j < nsamp; j++)
        {
            int idx = j + i * nsamp;
            data[idx] = (data[idx] - finalMedian[i]) / finalStd[i];
        }
    }
}

/// @brief Print threshold statistics for channel values
/// @param channel_values Array of channel values to analyze
/// @param nchan Number of channels
/// @param thresh_values Array of threshold values
/// @param threshold_names Array of threshold names
/// @param num_thresholds Number of thresholds
/// @param metric_name Name of the metric (e.g., "MAD", "STD")
void printThresholdStatistics(const float *channel_values, int nchan, 
                             const float *thresh_values, const char **threshold_names, 
                             int num_thresholds, const char *metric_name)
{
    int *threshold_counts = (int *)calloc(num_thresholds, sizeof(int));
    
    // Count channels that would be flagged at different thresholds
    int i, idx;
    for (i = 0; i < nchan; i++) {
        for (idx = 0; idx < num_thresholds; idx++) {
            // For negative thresholds (like -1*MAD), count values less than threshold
            // For positive thresholds, count values greater than threshold
            if (strstr(threshold_names[idx], "-") == threshold_names[idx]) {
                // Negative threshold: count values less than threshold
                if (channel_values[i] < thresh_values[idx]) threshold_counts[idx]++;
            } else {
                // Positive threshold: count values greater than threshold
                if (channel_values[i] > thresh_values[idx]) threshold_counts[idx]++;
            }
        }
    }
    
    printf("\n=== Channels flagged at different %s thresholds ===\n", metric_name);
    for (idx = 0; idx < num_thresholds; idx++) {
        printf("%s: %d channels (%.2f%%)\n", 
               threshold_names[idx], threshold_counts[idx], 
               (float)threshold_counts[idx]/nchan*100);
    }
    
    free(threshold_counts);
}



void drawChanStatHist(float *data, int nsamp, int nchan, int plot, int use_mad)
{
    int i, j;
    
    // Select appropriate statistical method
    const char* stat_name = use_mad ? "MAD" : "STD";
    const char* stat_symbol = use_mad ? "M" : "\\gs";
    const char* stat_label = use_mad ? "Channel MAD M\\dj\\u" : "Channel STD \\gs\\dj\\u";
    
    printf("\n=== %s Histogram Analysis ===\n", stat_name);
    
    // Allocate memory for channel statistics
    float *channel_stat = (float *)malloc(nchan * sizeof(float));
    float *channel_median = (float *)malloc(nchan * sizeof(float));
    float *temp_data = (float *)malloc(nsamp * sizeof(float));
    
    // Calculate statistics for each channel
    for (i = 0; i < nchan; i++)
    {
        // Copy channel data for processing
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        
        // Calculate median of the channel
        channel_median[i] = median(temp_data, nsamp);
        
        if (nsamp > 1) {
            if (use_mad) {
                // Calculate MAD (Mean Absolute Deviation from median)
                float sum_abs_dev = 0.0f;
                for (j = 0; j < nsamp; j++)
                {
                    float abs_dev = fabsf(data[i * nsamp + j] - channel_median[i]);
                    temp_data[j] = abs_dev;
                    sum_abs_dev += abs_dev;
                }
                channel_stat[i] = sum_abs_dev / nsamp;
            } else {
                // Calculate STD (Standard Deviation from median)
                float sum_squared_dev = 0.0f;
                for (j = 0; j < nsamp; j++)
                {
                    float deviation = data[i * nsamp + j] - channel_median[i];
                    float squared_dev = deviation * deviation;
                    temp_data[j] = squared_dev;
                    sum_squared_dev += squared_dev;
                }
                float mean_squared_dev = sum_squared_dev / nsamp;
                channel_stat[i] = sqrtf(mean_squared_dev);
            }
        } else {
            channel_stat[i] = 0.0f;
        }
    }
    
    // Calculate global statistics
    memcpy(temp_data, channel_stat, nchan * sizeof(float));
    float stat_median = median(temp_data, nchan);
    float stat_mad, stat_std;
    findMeanStd(channel_stat, nchan, &stat_mad, &stat_std); // Note: stat_mad here is actually mean
    
    // Calculate dispersion based on method (to match original functions exactly)
    float dispersion_value;
    if (use_mad) {
        // For MAD: use MAD function exactly like original visualizeChannelMAD
        // This calculates median of absolute deviations * 1.4826
        dispersion_value = mad(channel_stat, nchan);
        stat_mad = dispersion_value; // Keep stat_mad for display
    } else {
        // For STD: use median of absolute deviations (like original visualizeChannelStd)
        for (i = 0; i < nchan; i++) {
            temp_data[i] = fabsf(channel_stat[i] - stat_median);
        }
        dispersion_value = median(temp_data, nchan);
        stat_mad = dispersion_value; // Update stat_mad to show the correct dispersion
    }
    
    // Print statistics in original format
    if (use_mad) {
        printf("=== Channel MAD M_j Statistics ===\n");
        printf("Total channels: %d\n", nchan);
        printf("MAD Mean: %.6f\n", stat_mad);   // stat_mad is actually the mean from findMeanStd
        printf("MAD Std:  %.6f\n", stat_std);   // stat_std is the standard deviation from findMeanStd
        printf("MAD Median: %.6f\n", stat_median);
        printf("MAD MAD: %.6f\n", dispersion_value);
    } else {
        printf("\n=== Channel STD σ_j Statistics ===\n");
        printf("Total channels: %d\n", nchan);
        printf("STD Median: %.6f\n", stat_median);
        printf("STD STD: %.6f\n", dispersion_value);
    }
    
    if (!plot) {
        printf("%s histogram plotting disabled, skipping visualization.\n", stat_name);
        free(channel_stat);
        free(channel_median);
        free(temp_data);
        return;
    }
    
    // Find min and max for plotting
    float stat_min = channel_stat[0], stat_max = channel_stat[0];
    for (i = 1; i < nchan; i++) {
        if (channel_stat[i] < stat_min) stat_min = channel_stat[i];
        if (channel_stat[i] > stat_max) stat_max = channel_stat[i];
    }
    
    // Print statistics in original format
    if (use_mad) {
        printf("=== Channel MAD M_j Statistics ===\n");
        printf("Total channels: %d\n", nchan);
        printf("MAD Mean: %.6f\n", stat_mad);   // stat_mad is actually the mean from findMeanStd
        printf("MAD Std:  %.6f\n", stat_std);   // stat_std is the standard deviation from findMeanStd
        printf("MAD Median: %.6f\n", stat_median);
        printf("MAD MAD: %.6f\n", dispersion_value);
        printf("MAD Min: %.6f\n", stat_min);
        printf("MAD Max: %.6f\n", stat_max);
    } else {
        printf("\n=== Channel STD σ_j Statistics ===\n");
        printf("Total channels: %d\n", nchan);
        printf("STD Median: %.6f\n", stat_median);
        printf("STD STD: %.6f\n", dispersion_value);
        printf("STD Min: %.6f\n", stat_min);
        printf("STD Max: %.6f\n", stat_max);
    }
    
    // Unified 11-threshold line system
    const int NUM_ALL_THRESHOLDS = 11;
    float all_thresh_values[11];
    const char* all_threshold_labels[11];
    
    // Set labels based on statistical method
    if (use_mad) {
        const char* mad_labels[11] = {
            "-5*MAD", "-4*MAD", "-3*MAD", "-2*MAD", "-1*MAD", 
            "Median", 
            "+1*MAD", "+2*MAD", "+3*MAD", "+4*MAD", "+5*MAD"
        };
        for (i = 0; i < 11; i++) all_threshold_labels[i] = mad_labels[i];
    } else {
        const char* std_labels[11] = {
            "-5*STD", "-4*STD", "-3*STD", "-2*STD", "-1*STD",
            "Median",
            "1*STD", "2*STD", "3*STD", "4*STD", "5*STD"
        };
        for (i = 0; i < 11; i++) all_threshold_labels[i] = std_labels[i];
    }
    
    const int all_threshold_colors[11] = {4, 7, 3, 8, 6, 1, 6, 8, 3, 7, 4};
    const float all_y_positions[11] = {0.15f, 0.25f, 0.35f, 0.45f, 0.75f, 0.65f, 0.85f, 0.90f, 1.05f, 1.00f, 0.95f};
    
    // Paired threshold control switches: 5 switches control ±1, ±2, ±3, ±4, ±5 and median line
    const int show_median = 1;              
    const int threshold_pair_enabled[5] = {0, 1, 0, 0, 0}; // Only ±2sigma enabled (as requested)
    
    // Generate enabled array for 11 positions based on paired switches
    int threshold_enabled[11];
    for (i = 0; i < 5; i++) {
        // Negative thresholds
        threshold_enabled[4-i] = threshold_pair_enabled[i];
        // Positive thresholds
        threshold_enabled[6+i] = threshold_pair_enabled[i];
    }
    threshold_enabled[5] = show_median; // Median line
    
    // Calculate all 11 threshold line values uniformly using correct dispersion
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (i < 5) {
            // Negative thresholds
            float multiplier = (float)(5 - i);
            all_thresh_values[i] = stat_median - multiplier * dispersion_value;
        } else if (i == 5) {
            // Median line
            all_thresh_values[i] = stat_median;
        } else {
            // Positive thresholds
            float multiplier = (float)(i - 5);
            all_thresh_values[i] = stat_median + multiplier * dispersion_value;
        }
    }
    
    // Output enabled threshold line statistics
    printf("\n=== Enabled Threshold Statistics ===\n");
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) {
            printf("  %s: %.6f\n", all_threshold_labels[i], all_thresh_values[i]);
        }
    }
    
    // Create histogram
    int nbins = 100;
    float plot_min = stat_min;
    float plot_max = stat_max;
    float bin_width = (plot_max - plot_min) / nbins;
    float *hist = (float *)calloc(nbins, sizeof(float));
    
    // Fill histogram
    for (i = 0; i < nchan; i++) {
        int bin = (int)((channel_stat[i] - plot_min) / bin_width);
        if (bin < 0) bin = 0;
        if (bin >= nbins) bin = nbins - 1;
        hist[bin]++;
    }
    
    // Find maximum count for scaling
    float max_count = 0;
    for (i = 0; i < nbins; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    // Draw histogram
    printf("Creating %s histogram plot...\n", stat_name);
    cpgpage();
    cpgvstd();
    cpgsch(1.2);
    cpgswin(plot_min, plot_max, 0, max_count * 1.1f);
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpglab(stat_label, "Number of Channels", 
           use_mad ? "Channel MAD M\\dj\\u Distribution" : "Channel STD \\gs\\dj\\u Distribution");
    printf("%s histogram axes set up complete\n", stat_name);

    // Draw solid histogram bars
    cpgsci(2); // Red color
    for (i = 0; i < nbins; i++)
    {
        float x1 = plot_min + i * bin_width;
        float x2 = plot_min + (i + 1) * bin_width;
        cpgrect(x1, x2, 0, hist[i]);
    }

    // Draw threshold lines
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) {
            cpgsci(all_threshold_colors[i]);
            cpgmove(all_thresh_values[i], 0);
            cpgdraw(all_thresh_values[i], max_count * 1.1f);
            cpgptxt(all_thresh_values[i], max_count * all_y_positions[i], 0.0, 0.0, all_threshold_labels[i]);
        }
    }

    cpgsci(1); // Restore white color

    // Add Gaussian fitting curve with fixed amplitude
    printf("Fitting Gaussian curve to %s histogram (fixed amplitude method)...\n", stat_name);
    
    const float gaussian_fit_sigma_threshold = 5.0f;
    float fit_range_min = stat_median - gaussian_fit_sigma_threshold * dispersion_value;
    float fit_range_max = stat_median + gaussian_fit_sigma_threshold * dispersion_value;
    printf("Gaussian fitting range: [%.6f, %.6f] (median ± %.1fσ)\n", fit_range_min, fit_range_max, gaussian_fit_sigma_threshold);
    
    // Count non-zero bins for debugging
    int non_zero_bins = 0;
    for (i = 0; i < nbins; i++) {
        if (hist[i] > 0) non_zero_bins++;
    }
    
    // Prepare fitting data
    float *x_data = (float *)malloc(nbins * sizeof(float));
    float *y_data = (float *)malloc(nbins * sizeof(float));
    int fit_points = 0;
    
    for (i = 0; i < nbins; i++) {
        if (hist[i] > 0) {
            float bin_center = plot_min + (i + 0.5f) * bin_width;
            if (bin_center >= fit_range_min && bin_center <= fit_range_max) {
                x_data[fit_points] = bin_center;
                y_data[fit_points] = hist[i];
                fit_points++;
            }
        }
    }
    printf("Using %d out of %d bins for Gaussian fitting (within %.1fσ range)\n", fit_points, non_zero_bins, gaussian_fit_sigma_threshold);
    
    // Simple zero-bin removal
    if (fit_points > 0) {
        printf("Removing potential instrument artifact: y_data[0] = %.1f -> 0.0\n", y_data[0]);
        y_data[0] = 0.0f;
    }
    
    // Calculate fixed amplitude from main histogram max bin value
    float main_hist_amplitude = max_count;
    printf("Main histogram amplitude (max bin value): %.2f\n", main_hist_amplitude);
    
    // Use GSL two-parameter Gaussian fitting with fixed amplitude
    float fitted_mu, fitted_sigma;
    int gsl_success = gsl_gaussian_fit(x_data, y_data, fit_points, main_hist_amplitude, &fitted_mu, &fitted_sigma);
    
    if (gsl_success) {
        printf("Gaussian fit (fixed amp=%.2f): center=%.6f, sigma=%.6f\n", 
               main_hist_amplitude, fitted_mu, fitted_sigma);
    } else {
        printf("Gaussian fit failed, falling back to simple fit\n");
        fitted_sigma = simple_curve_fit(x_data, y_data, fit_points, stat_median);
        fitted_mu = stat_median;
        printf("Fallback Gaussian fit: center=%.6f, sigma=%.6f\n", fitted_mu, fitted_sigma);
    }
    
    // Store fitted parameters for use in zoomed histogram
    float global_fitted_mu = fitted_mu;
    float global_fitted_sigma = fitted_sigma;
    
    // Draw fitted Gaussian curve with fixed amplitude
    cpgsci(1); // Cyan color
    cpgsls(1); // Dashed line
    
    int curve_points = 200;
    float curve_step = (plot_max - plot_min) / curve_points;
    
    for (i = 0; i < curve_points; i++) {
        float x = plot_min + i * curve_step;
        float y = gaus_with_amplitude(x, main_hist_amplitude, global_fitted_mu, global_fitted_sigma);
        
        if (i == 0) {
            cpgmove(x, y);
        } else {
            cpgdraw(x, y);
        }
    }
    
    // Add detailed Gaussian fit parameters as text annotations
    cpgsci(1); // White color for text
    
    char fit_text1[100], fit_text2[100], fit_text3[100];
    sprintf(fit_text1, "Gaussian Fit Parameters:");
    sprintf(fit_text2, "\\gm = %.6f", global_fitted_mu);
    sprintf(fit_text3, "\\gs = %.6f", global_fitted_sigma);
    
    float text_x = plot_min + (plot_max - plot_min) * 0.65f;
    float text_y_base = max_count * 0.85f;
    float line_spacing = max_count * 0.05f;
    
    cpgptxt(text_x, text_y_base, 0.0, 0.0, fit_text1);
    cpgptxt(text_x, text_y_base - line_spacing, 0.0, 0.0, fit_text2);
    cpgptxt(text_x, text_y_base - 2 * line_spacing, 0.0, 0.0, fit_text3);
    
    cpgsls(1);
    free(x_data);
    free(y_data);
    cpgsci(1);

    // =====================================================================
    // Second part: Draw zoomed histogram for 0-0.25 range (detailed view)
    // =====================================================================
    printf("Creating zoomed %s histogram for range 0-0.25...\n", stat_name);
    
    // Check if there is data in the 0-0.25 range
    int has_data_in_range = 0;
    for (i = 0; i < nchan; i++) {
        if (channel_stat[i] >= 0.0f && channel_stat[i] <= 0.25f) {
            has_data_in_range = 1;
            break;
        }
    }
    
    if (has_data_in_range) {
        // Create new page for zoomed histogram
        cpgpage();
        cpgvstd();
        cpgsch(1.2);
        
        // Create subdivided histogram for 0-0.25 range
        int zoom_nbins = 50;
        float zoom_min = 0.0f;
        float zoom_max = 0.25f;
        float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
        float *zoom_hist = (float *)calloc(zoom_nbins, sizeof(float));
        
        // Fill zoomed histogram
        int zoom_count = 0;
        for (i = 0; i < nchan; i++) {
            if (channel_stat[i] >= zoom_min && channel_stat[i] <= zoom_max) {
                int bin = (int)((channel_stat[i] - zoom_min) / zoom_bin_width);
                if (bin < 0) bin = 0;
                if (bin >= zoom_nbins) bin = zoom_nbins - 1;
                zoom_hist[bin]++;
                zoom_count++;
            }
        }
        
        // Calculate maximum count in zoomed range
        float zoom_max_count = 0;
        for (i = 0; i < zoom_nbins; i++) {
            if (zoom_hist[i] > zoom_max_count) zoom_max_count = zoom_hist[i];
        }
        
        if (zoom_max_count > 0) {
            // Set up coordinate system
            cpgswin(zoom_min, zoom_max, 0, zoom_max_count * 1.1f);
            cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
            char zoom_title[200];
            sprintf(zoom_title, "Channel %s %s Distribution (Zoomed: 0-0.25)", stat_name, stat_symbol);
            cpglab(stat_label, "Number of Channels", zoom_title);
            
            // Draw zoomed histogram bars
            cpgsci(2); // Red color
            for (i = 0; i < zoom_nbins; i++) {
                if (zoom_hist[i] > 0) {
                    float x1 = zoom_min + i * zoom_bin_width;
                    float x2 = zoom_min + (i + 1) * zoom_bin_width;
                    cpgrect(x1, x2, 0, zoom_hist[i]);
                }
            }
            
            // Use unified system to draw threshold lines in zoomed plot
            for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
                if (threshold_enabled[i] && all_thresh_values[i] >= zoom_min && all_thresh_values[i] <= zoom_max) {
                    cpgsci(all_threshold_colors[i]);
                    cpgmove(all_thresh_values[i], 0);
                    cpgdraw(all_thresh_values[i], zoom_max_count * 1.1f);
                    float zoom_y_pos = 0.1f + (i * 0.08f);
                    if (zoom_y_pos > 0.9f) zoom_y_pos = 0.9f;
                    cpgptxt(all_thresh_values[i], zoom_max_count * zoom_y_pos, 
                           0.0, 0.0, all_threshold_labels[i]);
                }
            }
            
            // Draw the Gaussian curve with zoomed histogram specific amplitude
            float zoom_hist_amplitude = zoom_max_count;
            printf("Zoomed histogram amplitude (max bin value): %.2f\n", zoom_hist_amplitude);
            
            cpgsci(1);
            cpgsls(1);
            
            float zoom_curve_points = 200;
            for (i = 0; i < zoom_curve_points; i++) {
                float x = zoom_min + i * (zoom_max - zoom_min) / (zoom_curve_points - 1);
                float y = gaus_with_amplitude(x, zoom_hist_amplitude, global_fitted_mu, global_fitted_sigma);
                
                if (i == 0) {
                    cpgmove(x, y);
                } else {
                    cpgdraw(x, y);
                }
            }
            
            cpgsls(1);
            cpgsci(1);
            printf("Zoomed %s histogram completed! (%d channels in 0-0.25 range)\n", stat_name, zoom_count);
        } else {
            printf("No data found in 0-0.25 range for %s histogram\n", stat_name);
        }
        
        free(zoom_hist);
    } else {
        printf("No %s data in 0-0.25 range, skipping zoomed histogram\n", stat_name);
    }

    printf("%s histogram plot completed!\n", stat_name);
    free(hist);
    free(channel_stat);
    free(channel_median);
    free(temp_data);
    
    printf("=== %s Histogram Complete ===\n", stat_name);
}


void logicalOR(int *mask1, int *mask2, int *outmask, int nsamp, int nchan)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < nsamp * nchan; i++)
    {
        outmask[i] = mask1[i] | mask2[i];
    }
}

void flagChannelsByDualSumThreshold(
    float *data, int nsamp, int nchan, int *horizontalMask,
    float *channel_means, float *channel_stds, 
    float *temp_data_mean, int *local_mask_mean,
    float *temp_data_std, int *local_mask_std,
    float *M, float chi_1_mean, float chi_1_std, int M_len
) {
    int i, j;
    
    // Calculate mean and std for each channel
    for (i = 0; i < nchan; i++) {
        findMeanStd(data + i * nsamp, nsamp, &channel_means[i], &channel_stds[i]);
    }

    // Apply sumthreshold to mean sequence
    memset(local_mask_mean, 0, nchan * sizeof(int));
    sumthreshold_1d(channel_means, nchan, local_mask_mean, chi_1_mean, M_len, 
                    temp_data_mean, local_mask_mean, M, temp_data_mean + nchan);

    // Apply sumthreshold to std sequence
    memset(local_mask_std, 0, nchan * sizeof(int));
    sumthreshold_1d(channel_stds, nchan, local_mask_std, chi_1_std, M_len, 
                    temp_data_std, local_mask_std, M, temp_data_std + nchan);

    // Merge results (logical OR)
    for (i = 0; i < nchan; i++) {
        if (local_mask_mean[i] || local_mask_std[i]) {
            for (j = 0; j < nsamp; j++) {
                horizontalMask[i * nsamp + j] = 1;
            }
        }
    }
}

/**
 * @brief Perform iterative outlier detection and flagging for a single frequency channel
 * @param data Channel data array (nsamp length)
 * @param nsamp Number of samples in the channel
 * @param Nsigma Sigma threshold for outlier detection
 * @param mask Output mask for this channel (nsamp length)
 * @param median_temp Temporary array for median calculation (nsamp length)
 * @param good_samples Temporary array for good sample indices (nsamp length)
 * @param random_indices Temporary array for random indices (nsamp length)
 * @return Total number of outliers detected
 */
int iterativeChannelOutlierDetection(float *data, int nsamp, float Nsigma, 
                                   int *mask, float *median_temp,
                                   int *good_samples, int *random_indices)
{
    float lastMean = 0.0f, lastStd = 0.0f;
    float mean = 0.0f, std = 0.0f, med = 0.0f;
    float meanDiff = 0.0f, stdDiff = 0.0f;
    float upperBound, lowerBound;
    int totReplaceCnt = 0;
    int iter = 0;
    const int maxIterations = 3;
    
    // Copy data for median calculation
    memcpy(median_temp, data, nsamp * sizeof(float));
    
    while (iter < maxIterations)
    {
        lastMean = mean;
        lastStd = std;
        
        // Calculate statistics
        findMeanStd(data, nsamp, &mean, &std);
        med = median(median_temp, nsamp);
        
        // Calculate convergence metrics
        if (iter > 0) {
            meanDiff = (lastMean != 0.0f) ? fabsf(mean - lastMean) / fabsf(lastMean) : 0.0f;
            stdDiff = (lastStd != 0.0f) ? fabsf(std - lastStd) / fabsf(lastStd) : 0.0f;
            // medianDiff removed (not used in convergence condition)
        }
        
        // Calculate bounds
        float scale_row = 1.0f; // sqrtf(nsamp/nsamp)
        upperBound = med + Nsigma * scale_row * std;
        lowerBound = med - Nsigma * scale_row * std;
        
        // Flag outliers
    int newOutliers = 0;
    int j;
    for (j = 0; j < nsamp; j++)
        {
            if (data[j] > upperBound || data[j] < lowerBound)
            {
                if (mask[j] == 0) {
                    newOutliers++;
                    totReplaceCnt++;
                }
                mask[j] = 1;
            }
        }
        
        // Check for convergence
        if (newOutliers == 0 || (iter > 0 && meanDiff < 0.001f && stdDiff < 0.001f)) {
            break;
        }
        
        iter++;
    }
    
    return totReplaceCnt;
}

/**
 * @brief Perform iterative outlier detection and flagging for a single time sample
 * @param data Time sample data array (nchan length) 
 * @param nchan Number of channels in the time sample
 * @param Nsigma Sigma threshold for outlier detection
 * @param mask Output mask for this time sample (nchan length)
 * @param median_temp Temporary array for median calculation (nchan length)
 * @param good_samples Temporary array for good sample indices (nchan length)
 * @param random_indices Temporary array for random indices (nchan length)
 * @return Total number of outliers detected
 */
int iterativeTimeSampleOutlierDetection(float *data, int nchan, float Nsigma,
                                       int *mask, float *median_temp,
                                       int *good_samples, int *random_indices)
{
    float lastMean = 0.0f, lastStd = 0.0f;
    float mean = 0.0f, std = 0.0f, med = 0.0f;
    float meanDiff = 0.0f, stdDiff = 0.0f;
    float upperBound, lowerBound;
    
    int totReplaceCnt = 0;
    int iter = 0;
    const int maxIterations = 3;
    
    // Copy data for median calculation
    memcpy(median_temp, data, nchan * sizeof(float));
    
    while (iter < maxIterations)
    {
        lastMean = mean;
        lastStd = std;
        
        // Calculate statistics
        findMeanStd(data, nchan, &mean, &std);
        med = median(median_temp, nchan);
        
        // Calculate convergence metrics
        if (iter > 0) {
            meanDiff = (lastMean != 0.0f) ? fabsf(mean - lastMean) / fabsf(lastMean) : 0.0f;
            stdDiff = (lastStd != 0.0f) ? fabsf(std - lastStd) / fabsf(lastStd) : 0.0f;
            // medianDiff removed (not used in convergence condition)
        }
        
        // Calculate bounds (using scale factor of 1.0 for time samples)
        float scale_col = 1.0f;
        upperBound = med + Nsigma * scale_col * std;
        lowerBound = med - Nsigma * scale_col * std;
        
        // Flag outliers
    int newOutliers = 0;
    int j;
    for (j = 0; j < nchan; j++)
        {
            if (data[j] > upperBound || data[j] < lowerBound)
            {
                if (mask[j] == 0) {
                    newOutliers++;
                    totReplaceCnt++;
                }
                mask[j] = 1;
            }
        }
        
        // Check for convergence
        if (newOutliers == 0 || (iter > 0 && meanDiff < 0.001f && stdDiff < 0.001f)) {
            break;
        }
        
        iter++;
    }
    
    return totReplaceCnt;
}

/**
 * @brief Perform channel-level outlier detection using extracted iterative functions
 * @param data Input data array (nsamp * nchan)
 * @param nsamp Number of samples per channel
 * @param nchan Number of channels
 * @param Nsigma Sigma threshold for outlier detection
 * @param horizontalMask Output horizontal mask array
 * @param channel_fully_flagged Array indicating which channels are fully flagged
 * @return Total number of outliers detected across all channels
 */
int performChannelLevelDetection(float *data, int nsamp, int nchan, float Nsigma,
                               int *horizontalMask, int *channel_fully_flagged)
{
    int totalOutliers = 0;
    
    // Allocate temporary arrays for each thread
    #pragma omp parallel reduction(+:totalOutliers)
    {
        float *median_temp = (float *)malloc(nsamp * sizeof(float));
        int *good_samples = (int *)malloc(nsamp * sizeof(int));
        int *random_indices = (int *)malloc(nsamp * sizeof(int));
        int i;
        
        #pragma omp for
        for (i = 0; i < nchan; i++)
        {
            // Skip this channel if it's already fully flagged
            if (channel_fully_flagged[i]) {
                continue;
            }
            
            // Perform iterative outlier detection for this channel
            int channelOutliers = iterativeChannelOutlierDetection(
                data + i * nsamp, nsamp, Nsigma,
                horizontalMask + i * nsamp, median_temp,
                good_samples, random_indices
            );
            
            totalOutliers += channelOutliers;
        }
        
        free(median_temp);
        free(good_samples);
        free(random_indices);
    }
    
    return totalOutliers;
}

/**
 * @brief Perform time-sample-level outlier detection using extracted iterative functions
 * @param data Input transposed data array (nsamp * nchan, but accessed as nchan * nsamp)
 * @param nsamp Number of time samples
 * @param nchan Number of channels per time sample
 * @param Nsigma Sigma threshold for outlier detection
 * @param verticalMask Output vertical mask array (transposed layout)
 * @return Total number of outliers detected across all time samples
 */
int performTimeSampleLevelDetection(float *data, int nsamp, int nchan, float Nsigma,
                                  int *verticalMask)
{
    int totalOutliers = 0;
    
    // Allocate temporary arrays for each thread
    #pragma omp parallel reduction(+:totalOutliers)
    {
        float *median_temp = (float *)malloc(nchan * sizeof(float));
        int *good_samples = (int *)malloc(nchan * sizeof(int));
        int *random_indices = (int *)malloc(nchan * sizeof(int));
        int i;
        
        #pragma omp for
        for (i = 0; i < nsamp; i++)
        {
            // Perform iterative outlier detection for this time sample
            int sampleOutliers = iterativeTimeSampleOutlierDetection(
                data + i * nchan, nchan, Nsigma,
                verticalMask + i * nchan, median_temp,
                good_samples, random_indices
            );
            
            totalOutliers += sampleOutliers;
        }
        
        free(median_temp);
        free(good_samples);
        free(random_indices);
    }
    
    return totalOutliers;
}

/**
 * @brief Apply killThresh analysis to flag heavily contaminated channels
 * @param globalMask Input/output global mask array
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param killThresh Threshold for flagging entire channels
 * @param flaggedAfterSIR Initial flagged pixel count (after binarySIR)
 * @param killedChannels Output: number of channels killed
 * @param localRFISkipped Output: number of channels skipped due to localized RFI
 * @param totalFlaggedAfter Output: total flagged pixels after killThresh
 */
void applyKillThresh(int *globalMask, int nsamp, int nchan, float killThresh, 
                    int flaggedAfterSIR, int *killedChannels, int *localRFISkippedPtr, 
                    int *totalFlaggedAfter)
{
    *killedChannels = 0;
    *localRFISkippedPtr = 0;
    *totalFlaggedAfter = 0;
    float rangeThreshold = 0.5f;  // If flagged pixels span <50% of channel, don't kill entire channel
    
    printf("\n=== killThresh Analysis (threshold=%.3f) ===\n", killThresh);
    
    // Use local variables for OpenMP reduction, then assign to output parameters
    int localKilledChannels = 0;
    int localRFISkipped = 0;
    
    int chan;
    #pragma omp parallel for reduction(+:localKilledChannels,localRFISkipped)
    for (chan = 0; chan < nchan; chan++) {
        int maskedCount = 0;
        int firstFlagged = -1, lastFlagged = -1;
        int samp;
        
        // First pass: count flagged pixels and find range
        for (samp = 0; samp < nsamp; samp++) {
            int idx = samp + chan * nsamp;
            if (globalMask[idx]) {
                maskedCount++;
                if (firstFlagged == -1) firstFlagged = samp;
                lastFlagged = samp;
            }
        }
        
        float maskedRatio = (float)maskedCount / nsamp;
        int shouldKillChannel = 0;
        
        if (maskedRatio > killThresh) {
            if (firstFlagged != -1 && lastFlagged != -1) {
                int flaggedRange = lastFlagged - firstFlagged + 1;
                float rangeRatio = (float)flaggedRange / nsamp;
                
                if (rangeRatio >= rangeThreshold) {
                    shouldKillChannel = 1;
                } else {
                    localRFISkipped++;
                }
                
                if (maskedRatio > 0.001f) {
                    #pragma omp critical
                    {
                        // Optional detailed channel logging (currently commented out)
                        // printf("Channel %d: %d/%d flagged (%.3f%%), range [%d-%d] (%.1f%% span)", 
                        //        chan, maskedCount, nsamp, maskedRatio*100, 
                        //        firstFlagged, lastFlagged, rangeRatio*100);
                        if (maskedRatio > killThresh) {
                            if (shouldKillChannel) {
                                // printf(" -> KILLING ENTIRE CHANNEL");
                            } else {
                                // printf(" -> SKIPPED (localized RFI)");
                            }
                        }
                        // printf("\n");
                    }
                }
            } else {
                shouldKillChannel = 1;
            }
        }
        
        if (shouldKillChannel) {
            localKilledChannels++;
            for (samp = 0; samp < nsamp; samp++) {
                int idx = samp + chan * nsamp;
                globalMask[idx] = 1;
            }
        }
    }
    
    // Assign local variables to output parameters
    *killedChannels = localKilledChannels;
    *localRFISkippedPtr = localRFISkipped;
    
    // Count total flagged pixels after killThresh
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) (*totalFlaggedAfter)++;
    }
}

/**
 * @brief Print killThresh analysis summary
 * @param killedChannels Number of channels killed
 * @param localRFISkipped Number of channels skipped due to localized RFI
 * @param totalFlaggedBefore Flagged pixels before killThresh
 * @param totalFlaggedAfter Flagged pixels after killThresh
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 */
void printKillThreshSummary(int killedChannels, int localRFISkipped, int totalFlaggedBefore, 
                           int totalFlaggedAfter, int nsamp, int nchan)
{
    float rangeThreshold = 0.5f;  // Keep consistent with applyKillThresh
    
    printf("killThresh Summary:\n");
    printf("  - Killed channels: %d/%d (%.2f%%)\n", killedChannels, nchan, 
           (float)killedChannels/nchan*100);
    printf("  - Localized RFI skipped: %d/%d (%.2f%%)\n", localRFISkipped, nchan, 
           (float)localRFISkipped/nchan*100);
    printf("  - Range threshold: %.1f%% (flagged pixels must span >%.1f%% of channel to kill)\n", 
           rangeThreshold*100, rangeThreshold*100);
    printf("  - Flagged pixels before: %d/%d (%.2f%%)\n", totalFlaggedBefore, nsamp*nchan, 
           (float)totalFlaggedBefore/(nsamp*nchan)*100);
    printf("  - Flagged pixels after: %d/%d (%.2f%%)\n", totalFlaggedAfter, nsamp*nchan, 
           (float)totalFlaggedAfter/(nsamp*nchan)*100);
    printf("  - Additional pixels flagged: %d\n", totalFlaggedAfter - totalFlaggedBefore);
    printf("=== End killThresh Analysis ===\n\n");
}

/**
 * @brief Apply combined killThresh analysis and pixel substitution
 * For channels exceeding killThresh: mark entire channel if range criteria met
 * For channels below killThresh: perform pixel-level substitution
 * @param data Data array (will be modified for pixel substitution)
 * @param globalMask RFI mask array (will be modified for killed channels)
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param killThresh Threshold ratio for considering channel killing
 * @param flaggedBefore Number of flagged pixels before processing
 * @param killedChannels Output: number of channels killed
 * @param localRFISkipped Output: number of channels skipped due to localized RFI
 * @param totalFlaggedAfter Output: total flagged pixels after processing
 * @param pixelsSubstituted Output: number of pixels substituted (not including killed channels)
 */
void applyKillThreshAndSubstitution(float *data, int *globalMask, int nsamp, int nchan, 
                                   float killThresh, int flaggedBefore,
                                   int *killedChannels, int *localRFISkippedPtr, 
                                   int *totalFlaggedAfter, int *pixelsSubstituted)
{
    *killedChannels = 0;
    *localRFISkippedPtr = 0;
    *totalFlaggedAfter = 0;
    *pixelsSubstituted = 0;
    float rangeThreshold = 0.5f;  // If flagged pixels span <50% of channel, don't kill entire channel
    
    printf("\n=== Combined killThresh Analysis and Pixel Substitution (threshold=%.3f) ===\n", killThresh);
    
    // Allocate arrays for per-channel processing
    int *channelActions = (int *)calloc(nchan, sizeof(int)); // 0=substitute, 1=kill, 2=skip
    int *channelMaskedCounts = (int *)malloc(nchan * sizeof(int));
    
    // Use local variables for OpenMP reduction
    int localKilledChannels = 0;
    int localRFISkipped = 0;
    
    // First pass: analyze each channel and decide action
    int chan;
    #pragma omp parallel for reduction(+:localKilledChannels,localRFISkipped)
    for (chan = 0; chan < nchan; chan++) {
        int maskedCount = 0;
        int firstFlagged = -1, lastFlagged = -1;
        int samp;
        
        // Count flagged pixels and find range
        for (samp = 0; samp < nsamp; samp++) {
            int idx = samp + chan * nsamp;
            if (globalMask[idx]) {
                maskedCount++;
                if (firstFlagged == -1) firstFlagged = samp;
                lastFlagged = samp;
            }
        }
        
        channelMaskedCounts[chan] = maskedCount;
        float maskedRatio = (float)maskedCount / nsamp;
        
        if (maskedRatio > killThresh) {
            if (firstFlagged != -1 && lastFlagged != -1) {
                int flaggedRange = lastFlagged - firstFlagged + 1;
                float rangeRatio = (float)flaggedRange / nsamp;
                
                if (rangeRatio >= rangeThreshold) {
                    channelActions[chan] = 1; // Kill entire channel
                    localKilledChannels++;
                } else {
                    channelActions[chan] = 2; // Skip (localized RFI)
                    localRFISkipped++;
                }
            } else {
                channelActions[chan] = 1; // Kill entire channel
                localKilledChannels++;
            }
        } else {
            channelActions[chan] = 0; // Substitute pixels
        }
    }
    
    // Second pass: apply actions
    int localPixelsSubstituted = 0;
    
    for (chan = 0; chan < nchan; chan++) {
        if (channelActions[chan] == 1) {
            // Kill entire channel
            for (int samp = 0; samp < nsamp; samp++) {
                int idx = samp + chan * nsamp;
                globalMask[idx] = 1;
            }
            // printf("Channel %d: killed entire channel (%d/%d pixels, %.2f%%)\n", 
            //        chan, channelMaskedCounts[chan], nsamp, 
            //        (float)channelMaskedCounts[chan]/nsamp*100);
        } else if (channelActions[chan] == 0 && channelMaskedCounts[chan] > 0) {
            // Substitute pixels in this channel only
            float *channelData = data + chan * nsamp;
            int *channelMask = globalMask + chan * nsamp;
            
            // Allocate temporary arrays for this channel
            int *goodSamples = (int *)malloc(nsamp * sizeof(int));
            int *randomIndices = (int *)malloc(nsamp * sizeof(int));
            
            if (goodSamples && randomIndices) {
                substPixels(channelData, nsamp, channelMask, goodSamples, randomIndices);
                localPixelsSubstituted += channelMaskedCounts[chan];
                // printf("Channel %d: substituted %d pixels (%.2f%%)\n", 
                //        chan, channelMaskedCounts[chan], 
                //        (float)channelMaskedCounts[chan]/nsamp*100);
            } else {
                printf("Warning: Memory allocation failed for channel %d substitution\n", chan);
            }
            
            free(goodSamples);
            free(randomIndices);
        } else if (channelActions[chan] == 2) {
            // Skipped due to localized RFI
            // printf("Channel %d: skipped substitution (localized RFI, %d/%d pixels, %.2f%%)\n", 
            //        chan, channelMaskedCounts[chan], nsamp, 
            //        (float)channelMaskedCounts[chan]/nsamp*100);
        }
    }
    
    // Assign local variables to output parameters
    *killedChannels = localKilledChannels;
    *localRFISkippedPtr = localRFISkipped;
    *pixelsSubstituted = localPixelsSubstituted;
    
    // Count total flagged pixels after processing
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) (*totalFlaggedAfter)++;
    }
    
    // Print summary
    printf("\nCombined Processing Summary:\n");
    printf("  - Channels killed: %d/%d (%.2f%%)\n", *killedChannels, nchan, 
           (float)(*killedChannels)/nchan*100);
    printf("  - Channels with pixel substitution: %d/%d\n", 
           nchan - *killedChannels - *localRFISkippedPtr, nchan);
    printf("  - Channels skipped (localized RFI): %d/%d (%.2f%%)\n", 
           *localRFISkippedPtr, nchan, (float)(*localRFISkippedPtr)/nchan*100);
    printf("  - Pixels substituted: %d\n", *pixelsSubstituted);
    printf("  - Flagged pixels before: %d/%d (%.2f%%)\n", flaggedBefore, nsamp*nchan, 
           (float)flaggedBefore/(nsamp*nchan)*100);
    printf("  - Flagged pixels after: %d/%d (%.2f%%)\n", *totalFlaggedAfter, nsamp*nchan, 
           (float)(*totalFlaggedAfter)/(nsamp*nchan)*100);
    printf("  - Additional pixels flagged: %d\n", *totalFlaggedAfter - flaggedBefore);
    printf("=== End Combined Processing ===\n\n");
    
    // Clean up
    free(channelActions);
    free(channelMaskedCounts);
}

void identSubstNSigma(
    float *data, int nsamp, int nchan, float Nsigma, float channel_std_threshold, int iterationIndex, int plot,
    int *horizontalMask, int *verticalMask, int *globalMask,
    float *finalMedian, float *finalStd, int cudaReady)
{    
    memset(horizontalMask, 0, nsamp * nchan * sizeof(int));
    memset(verticalMask, 0, nsamp * nchan * sizeof(int));
    memset(globalMask, 0, nsamp * nchan * sizeof(int));

    int *good_samples = (int *)malloc(nsamp * sizeof(int));
    int *random_indices = (int *)malloc(nsamp * sizeof(int));
    float *median_temp = (float *)malloc(nsamp * nchan * sizeof(float));
    memcpy(median_temp, data, nsamp * nchan * sizeof(float));
    
    float killThresh = 0.2f;
    int i, j;
    
    
    // Visualize channel MAD statistics for threshold determination in first 20 iterations
    // Use iterationIndex as a proxy for iteration counter (passed from ReadFASTData.c)
    if (plot)
    {
        printf("=== Generating Channel MAD M_j Histogram (Iteration %d) ===\n", iterationIndex);
        // visualizeChannelMAD(data, nsamp, nchan, 1);
        drawChanStatHist(data, nsamp, nchan, 1, 1);
        printf("=== MAD M_j Histogram Complete ===\n");
        
        printf("=== Generating Channel STD σ_j Histogram (Iteration %d) ===\n", iterationIndex);
        // visualizeChannelStd(data, nsamp, nchan, 1);
        drawChanStatHist(data, nsamp, nchan, 1, 0);
        printf("=== STD σ_j Histogram Complete ===\n");

        
    }
    
    // === 1. Point interference detection first ===
    printf("=== Performing point-level (pixel) outlier detection ===\n");
    int pixelOutliers = 0;
    int *channel_fully_flagged_temp = (int *)calloc(nchan, sizeof(int)); // Temporary array, all zeros initially
    pixelOutliers = performChannelLevelDetection(data, nsamp, nchan, Nsigma, 
                                                horizontalMask, channel_fully_flagged_temp);
    free(channel_fully_flagged_temp);
    printf("Point-level detection: flagged %d outlier pixels\n", pixelOutliers);
    
    // === 2. Channel level flagging second ===
    float *channel_stds = (float *)malloc(nchan * sizeof(float));
    float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
    flagChannelsByStdOutliers(data, nsamp, nchan, horizontalMask, channel_stds, channel_stds_temp, channel_std_threshold);
    free(channel_stds);
    free(channel_stds_temp);
    
    // Check which channels are fully flagged after both detections
    int fully_flagged_channels = 0;
    int *channel_fully_flagged = (int *)calloc(nchan, sizeof(int));
    for (i = 0; i < nchan; i++) {
        int flagged_count = 0;
        for (j = 0; j < nsamp; j++) {
            if (horizontalMask[i * nsamp + j] == 1) {
                flagged_count++;
            }
        }
        if (flagged_count == nsamp) {
            channel_fully_flagged[i] = 1;
            fully_flagged_channels++;
        }
    }
    printf("Combined detection: %d/%d channels fully flagged (%.2f%%)\n", 
           fully_flagged_channels, nchan, (float)fully_flagged_channels/nchan*100);
    

    // subtractChannelMedians(data, nsamp, nchan);

    // === 3. Copy horizontal mask to global mask (vertical detection removed for efficiency) ===
    int horizontalFlagged = 0;
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        globalMask[idx] = horizontalMask[idx];  // Direct copy instead of logicalOR
        if (horizontalMask[idx] == 1) horizontalFlagged++;
    }
    
    printf("\n=== RFI Detection Statistics ===\n");
    printf("Combined (point+channel) mask flagged: %d/%d pixels (%.4f%%)\n", 
           horizontalFlagged, nsamp*nchan, (float)horizontalFlagged/(nsamp*nchan)*100);
    printf("Global mask flagged: %d/%d pixels (%.4f%%)\n", 
           horizontalFlagged, nsamp*nchan, (float)horizontalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");

    // Apply binarySIR before killThresh analysis to filter isolated pixels for better range calculation
    printf("\n=== Applying binarySIR filtering before killThresh analysis ===\n");
    int flaggedBeforeSIR = horizontalFlagged;
    
    // Use CUDA-accelerated binarySIR if available
    if (cudaReady) {
        printf("Using CUDA-accelerated binarySIR filtering for killThresh analysis...\n");
        cuda_binarySIR(globalMask, nsamp, nchan, 3, 3, 1.0f, 0.2f);
    } else {
        binarySIR(globalMask, nsamp, nchan, 3, 3, 1.0f, 0.2f); // Filter out isolated pixels
    }
    
    // Recount flagged pixels after binarySIR
    int flaggedAfterSIR = 0;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) flaggedAfterSIR++;
    }
    printf("binarySIR filtering: %d -> %d flagged pixels (removed %d isolated pixels)\n", 
           flaggedBeforeSIR, flaggedAfterSIR, flaggedBeforeSIR - flaggedAfterSIR);

    // === 4. Combined killThresh Analysis and Pixel Substitution ===
    int killedChannels, localRFISkipped, totalFlaggedAfter, pixelsSubstituted;
    applyKillThreshAndSubstitution(data, globalMask, nsamp, nchan, killThresh, flaggedAfterSIR,
                                  &killedChannels, &localRFISkipped, &totalFlaggedAfter, 
                                  &pixelsSubstituted);

    // Calculate final statistics for experimental function output
    float finalMedian_temp, finalStd_temp;
    findMeanStd(median_temp, nsamp * nchan, &finalMedian_temp, &finalStd_temp);
    float finalMedian_value = median(median_temp, nsamp * nchan);

    float outlierRatio = (float)pixelOutliers / (nsamp * nchan);
    if (outlierRatio > killThresh)
    {
        printf("WARNING: High outlier ratio %.4f > %.2f detected - data may be corrupted\n",
               outlierRatio, killThresh);
    }

    *finalMedian = finalMedian_value;
    *finalStd = finalStd_temp;

    // Clean up allocated memory
    free(good_samples);
    free(random_indices);
    free(channel_fully_flagged);
    free(median_temp);

    printf("### DEBUG: identSubstNSigma exiting with finalMedian=%.6f, finalStd=%.6f ###\n", *finalMedian, *finalStd);
    fflush(stdout);  // Ensure immediate output
}