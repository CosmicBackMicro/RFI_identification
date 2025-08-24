#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

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
        
        if (i < 5) {  // Debug output for first 5 channels
            printf("Channel %d median: %.6f\n", i, channel_medians[i]);
        }
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

/// @brief Substitute masked pixels in each channel with random samples from good pixels in the same channel.
/// @param data Data array to be processed, time samples from same channel are stored contiguously.
/// @param nsamp Number of time samples in each channel.
/// @param nchan Number of frequency channels.
/// @param mask Mask array indicating which pixels are masked (1 for masked, 0 for good).
void substitute_pixels(float *data, int nsamp, int nchan, int *mask)
{
    int i, j;
    int *good_samples = calloc(nsamp, sizeof(int));
    srand(time(NULL)); // Seed RNG

    // Process each channel separately
    // #pragma omp parallel for
    for (i = 0; i < nchan; i++)
    {
        int chan_offset = i * nsamp;
        int good_count = 0;
        // Single pass: collect good samples and count
        for (j = 0; j < nsamp; j++)
        {
            int idx = chan_offset + j;
            if (!mask[idx])
            {
                good_samples[good_count++] = j;
            }
        }
        // Replace masked samples if good samples exist
        if (good_count > 0)
        {
            for (j = 0; j < nsamp; j++)
            {
                int idx = chan_offset + j;
                if (mask[idx] == 1)
                {
                    int random_sample = good_samples[rand() % good_count];
                    data[idx] = data[chan_offset + random_sample];
                }
            }
        }
    }
    free(good_samples);
}

/// @brief Substitute masked elements in a 1D array with random samples from unmasked elements or the median if all are masked.
/// @param data Pointer to the data array.
/// @param size Number of elements in the array.
/// @param mask Mask array indicating which elements are masked (1 for masked, 0 for good).
/// @param goodSamps Pre-allocated empty array of size `size` to hold indices of unmasked elements.
/// @param randIdx Pre-allocated empty array of size `size` to hold random indices for replacement.
void substitute_pixels_1d(float *data, int size, int *mask, int *goodSamps, int *randIdx) {
    int i, goodCnt = 0;
    // Collect indices of good samples
    for (i = 0; i < size; i++) {
        if (!mask[i]) {
            goodSamps[goodCnt] = i;
            goodCnt++;
        }
    }
    
    if (goodCnt == 0) {
        // printf("Error: No good samples found! Check input mask!\n");
        return;
    }
    

    // Prepare random indices for replacement
    unsigned int seed = (unsigned int)(time(NULL)); // Seed RNG
    for (i = 0; i < size; i++) {
        randIdx[i] = rand_r(&seed) % goodCnt; // Random index in range [0, goodCnt), values are indices in goodSamps
    }

    for (i = 0; i < size; i++) {
        if (mask[i]) {
            data[i] = data[goodSamps[randIdx[i]]];
        }
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
                              float *channel_stds, float *channel_stds_temp)
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
    float std_lower_bound = std_median - 2.0f * std_mad;
    float std_upper_bound = std_median + 2.0f * std_mad;
    
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

/// @brief Calculate and visualize channel Median Absolute Deviation (MAD) statistics for threshold determination
/// MAD is calculated as the mean of absolute deviations from median for each channel
/// @param data Input data array (nsamp * nchan)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param plot Whether to plot the histogram (1 for yes, 0 for no)
void visualizeChannelMAD(float *data, int nsamp, int nchan, int plot)
{
    int i, j;
    
    // Allocate memory for channel statistics
    float *channel_mad = (float *)malloc(nchan * sizeof(float));
    float *channel_median = (float *)malloc(nchan * sizeof(float));
    float *temp_data = (float *)malloc(nsamp * sizeof(float));
    
    // Calculate MAD for each channel (mean of absolute deviations from median)
    for (i = 0; i < nchan; i++)
    {
        // Copy channel data for processing
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        
        // Calculate median of the channel
        channel_median[i] = median(temp_data, nsamp);
        
        // Calculate absolute deviations from median
        if (nsamp > 1) {
            float sum_abs_dev = 0.0f;
            for (j = 0; j < nsamp; j++)
            {
                float abs_dev = fabsf(data[i * nsamp + j] - channel_median[i]);
                temp_data[j] = abs_dev;
                sum_abs_dev += abs_dev;
            }
            
            // Calculate MAD as mean of absolute deviations from median
            channel_mad[i] = sum_abs_dev / nsamp;
            
        } else {
            // If only one sample, set MAD to 0
            channel_mad[i] = 0.0f;
        }
    }
    
    // Calculate statistics of channel MADs
    float mad_mean, mad_std;
    findMeanStd(channel_mad, nchan, &mad_mean, &mad_std);
    
    memcpy(temp_data, channel_mad, nchan * sizeof(float));
    float mad_median = median(temp_data, nchan);
    float mad_mad = mad(channel_mad, nchan);
    
    // Find min and max for histogram range
    float mad_min = channel_mad[0];
    float mad_max = channel_mad[0];
    for (i = 1; i < nchan; i++)
    {
        if (channel_mad[i] < mad_min) mad_min = channel_mad[i];
        if (channel_mad[i] > mad_max) mad_max = channel_mad[i];
    }
    
    // Print statistics
    printf("=== Channel MAD M_j Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("MAD Mean: %.6f\n", mad_mean);
    printf("MAD Std:  %.6f\n", mad_std);
    printf("MAD Median: %.6f\n", mad_median);
    printf("MAD MAD: %.6f\n", mad_mad);
    printf("MAD Min: %.6f\n", mad_min);
    printf("MAD Max: %.6f\n", mad_max);
    
    // Unified 11-threshold line system
    const int NUM_ALL_THRESHOLDS = 11;
    float all_thresh_values[11];
    const char* all_threshold_labels[11] = {
        "-5*MAD", "-4*MAD", "-3*MAD", "-2*MAD", "-1*MAD", 
        "Median", 
        "+1*MAD", "+2*MAD", "+3*MAD", "+4*MAD", "+5*MAD"
    };
    const int all_threshold_colors[11] = {4, 7, 3, 8, 6, 1, 6, 8, 3, 7, 4}; // Color scheme: blue, yellow, green, orange, magenta, white, etc.
    const float all_y_positions[11] = {0.15f, 0.25f, 0.35f, 0.45f, 0.75f, 0.65f, 0.85f, 0.90f, 1.05f, 1.00f, 0.95f};
    
    // Paired threshold control switches: 5 switches control ±1, ±2, ±3, ±4, ±5 and median line
    const int show_median = 1;              
    const int threshold_pair_enabled[5] = {1, 1, 1, 0, 1}; // Enable/disable ±1, ±2, ±3, ±4, ±5 pairs
    
    // Generate enabled array for 11 positions based on paired switches
    int threshold_enabled[11];
    for (i = 0; i < 5; i++) {
        // Negative thresholds (-5*MAD to -1*MAD, indices 0-4)
        threshold_enabled[4-i] = threshold_pair_enabled[i]; // i=0→index4(±1), i=1→index3(±2), etc.
        // Positive thresholds (+1*MAD to +5*MAD, indices 6-10)  
        threshold_enabled[6+i] = threshold_pair_enabled[i]; // i=0→index6(±1), i=1→index7(±2), etc.
    }
    threshold_enabled[5] = show_median; // Median line (index 5)
    
    // Calculate all 11 threshold line values uniformly
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (i < 5) {
            // Negative thresholds: -5*MAD to -1*MAD (indices 0-4)
            float multiplier = (float)(5 - i); // i=0→5, i=1→4, i=2→3, i=3→2, i=4→1
            all_thresh_values[i] = mad_median - multiplier * mad_mad;
        } else if (i == 5) {
            // Median line (index 5)
            all_thresh_values[i] = mad_median;
        } else {
            // Positive thresholds: +1*MAD to +5*MAD (indices 6-10)
            float multiplier = (float)(i - 5); // i=6→1, i=7→2, i=8→3, i=9→4, i=10→5
            all_thresh_values[i] = mad_median + multiplier * mad_mad;
        }
    }
    
    // Output enabled threshold line statistics
    printf("\n=== Enabled Threshold Statistics ===\n");
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) {
            printf("%s threshold: %.6f\n", all_threshold_labels[i], all_thresh_values[i]);
        }
    }
    
    // Statistical analysis only for enabled thresholds
    int enabled_count = 0;
    for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) enabled_count++;
    }
    
    if (enabled_count > 0) {
        float *enabled_values = malloc(enabled_count * sizeof(float));
        const char **enabled_names = malloc(enabled_count * sizeof(const char*));
        
        int idx = 0;
        for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
            if (threshold_enabled[i]) {
                enabled_values[idx] = all_thresh_values[i];
                enabled_names[idx] = all_threshold_labels[i];
                idx++;
            }
        }
        
        printThresholdStatistics(channel_mad, nchan, enabled_values, enabled_names, enabled_count, "MAD");
        
        free(enabled_values);
        free(enabled_names);
    }
    
    if (plot)
    {
        //=============================================================================
        // Part 1: Full MAD Histogram
        //=============================================================================
        
        // Use standard histogram bin count
        int nbins = (int)(sqrt(nchan) * 2);
        if (nbins < 30) nbins = 30;
        if (nbins > 100) nbins = 100;
        printf("Using %d bins for MAD histogram\n", nbins);
        
        float *hist = (float *)calloc(nbins, sizeof(float));
        float plot_min = mad_min;
        float plot_max = mad_max;
        
        // Slightly expand range to avoid boundary issues
        float range = plot_max - plot_min;
        if (range > 0) {
            plot_min -= 0.01f * range;
            plot_max += 0.01f * range;
        } else {
            // If all values are the same, create small range
            plot_min -= 0.001f;
            plot_max += 0.001f;
        }
        float bin_width = (plot_max - plot_min) / nbins;

        printf("MAD value range: [%.6f, %.6f], using %d bins\n", mad_min, mad_max, nbins);

        // Fill histogram
        for (i = 0; i < nchan; i++)
        {
            int bin = (int)((channel_mad[i] - plot_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;
            hist[bin]++;
            if (i < 10) {
                printf("MAD[%d]=%.6f -> bin %d\n", i, channel_mad[i], bin);
            }
        }

        // Count non-zero bins
        int non_zero_bins = 0;
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) {
                non_zero_bins++;
                if (non_zero_bins <= 10) { // Only print first 10 non-zero bins
                    printf("Bin %d: %.0f channels (x-range: %.6f to %.6f)\n", 
                           i, hist[i], plot_min + i * bin_width, plot_min + (i + 1) * bin_width);
                }
            }
        }
        printf("Total non-zero bins: %d out of %d\n", non_zero_bins, nbins);

        // Calculate maximum count
        float max_count = 0;
        for (i = 0; i < nbins; i++)
        {
            if (hist[i] > max_count) max_count = hist[i];
        }

        // Plot 1: Create full MAD histogram
        printf("Creating MAD histogram plot...\n");
        cpgpage();                          // Create new page
        cpgvstd();                         // Set standard viewport
        cpgsch(1.2);                       // Set character size
        cpgswin(plot_min, plot_max, 0, max_count * 1.1f);  // Set world coordinates
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);          // Draw coordinate axes
        cpglab("Channel MAD M\\dj\\u", "Number of Channels", "Channel MAD M\\dj\\u Distribution");  // Add labels
        printf("MAD histogram axes set up complete\n");

        // Plot 1: Draw histogram bars
        cpgsci(2); // Red color
        for (i = 0; i < nbins; i++)
        {
            float x1 = plot_min + i * bin_width;
            float x2 = plot_min + (i + 1) * bin_width;
            cpgrect(x1, x2, 0, hist[i]);   // Draw each histogram bar
        }

        // Plot 1: Draw unified 11-threshold line system
        for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
            if (threshold_enabled[i] && all_thresh_values[i] >= plot_min && all_thresh_values[i] <= plot_max) {
                cpgsci(all_threshold_colors[i]);                    // Set threshold line color
                cpgmove(all_thresh_values[i], 0);                   // Move to starting point
                cpgdraw(all_thresh_values[i], max_count * 1.1f);    // Draw vertical line
                cpgptxt(all_thresh_values[i], max_count * all_y_positions[i], 0.0, 0.0, all_threshold_labels[i]);  // Add label
            }
        }

        cpgsci(1); // Restore white color

        // Add Gaussian fitting curve with fixed amplitude
        printf("Fitting Gaussian curve to MAD histogram (fixed amplitude method)...\n");
        
        // Unified Gaussian fitting threshold control
        const float gaussian_fit_sigma_threshold = 5.0f;  // Sigma threshold for Gaussian fitting range
        
        // Define sigma range around median for fitting
        float fit_range_min = mad_median - gaussian_fit_sigma_threshold * mad_mad;
        float fit_range_max = mad_median + gaussian_fit_sigma_threshold * mad_mad;
        printf("Gaussian fitting range: [%.6f, %.6f] (median ± %.1fσ)\n", fit_range_min, fit_range_max, gaussian_fit_sigma_threshold);
        
        // Prepare fitting data: convert histogram to x-y data points within 5-sigma range
        float *x_data = (float *)malloc(nbins * sizeof(float));
        float *y_data = (float *)malloc(nbins * sizeof(float));
        int fit_points = 0;
        
        // Plot 1: Gaussian curve fitting (only use data within median ± configurable σ)
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) {  // Only use non-zero bins for fitting
                float bin_center = plot_min + (i + 0.5f) * bin_width;
                // Only include bins within sigma range around median
                if (bin_center >= fit_range_min && bin_center <= fit_range_max) {
                    x_data[fit_points] = bin_center;
                    y_data[fit_points] = hist[i];
                    fit_points++;
                }
            }
        }
        printf("Using %d out of %d bins for Gaussian fitting (within %.1fσ range)\n", fit_points, non_zero_bins, gaussian_fit_sigma_threshold);
        
        // Simple zero-bin removal: set the first y value to zero to eliminate instrument artifacts
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
            printf("GSL 2-param Gaussian fit (fixed amp=%.2f): center=%.6f, sigma=%.6f\n", 
                   main_hist_amplitude, fitted_mu, fitted_sigma);
        } else {
            printf("GSL 2-param Gaussian fit failed, falling back to simple fit\n");
            fitted_sigma = simple_curve_fit(x_data, y_data, fit_points, mad_median);
            fitted_mu = mad_median;
            printf("Fallback Gaussian fit: center=%.6f, sigma=%.6f\n", fitted_mu, fitted_sigma);
        }
        
        // Store fitted parameters for use in zoomed histogram
        float global_fitted_mu = fitted_mu;
        float global_fitted_sigma = fitted_sigma;
        
        // Plot 1: Draw fitted Gaussian curve with fixed amplitude
        cpgsci(5); // Cyan color
        cpgsls(2); // Dashed line
        
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
        
        cpgptxt(mad_median + global_fitted_sigma, max_count * 0.4f, 0.0, 0.0, "Gaussian Fit");
        cpgsls(1);

        free(x_data);
        free(y_data);

        cpgsci(1); // Restore white color
        // *** Plot 1 complete ***

        //=============================================================================
        // Part 2: Zoomed MAD Histogram (Range 0-0.25) 
        //=============================================================================
        
        // Add zoomed histogram for 0-0.25 range
        printf("Creating zoomed MAD histogram for range 0-0.25...\n");
        
        // Check if there is data in 0-0.25 range
        int has_data_in_range = 0;
        for (i = 0; i < nchan; i++) {
            if (channel_mad[i] >= 0.0f && channel_mad[i] <= 0.25f) {
                has_data_in_range = 1;
                break;
            }
        }
        
        if (has_data_in_range) {
            // Plot 2: Create new page and setup
            cpgpage();                          // Create new page for zoomed histogram
            cpgvstd();                         // Set standard viewport
            cpgsch(1.2);                       // Set character size
            
            // Plot 2: Create fine-grained histogram data for 0-0.25 range
            int zoom_nbins = 50;               // Use more bins for higher resolution
            float zoom_min = 0.0f;
            float zoom_max = 0.25f;
            float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
            float *zoom_hist = (float *)calloc(zoom_nbins, sizeof(float));
            
            // Plot 2: Fill zoomed histogram data
            int zoom_count = 0;
            for (i = 0; i < nchan; i++) {
                if (channel_mad[i] >= zoom_min && channel_mad[i] <= zoom_max) {
                    int bin = (int)((channel_mad[i] - zoom_min) / zoom_bin_width);
                    if (bin < 0) bin = 0;
                    if (bin >= zoom_nbins) bin = zoom_nbins - 1;
                    zoom_hist[bin]++;
                    zoom_count++;
                }
            }
            
            // Plot 2: Calculate maximum count for zoomed range
            float zoom_max_count = 0;
            for (i = 0; i < zoom_nbins; i++) {
                if (zoom_hist[i] > zoom_max_count) zoom_max_count = zoom_hist[i];
            }
            
            if (zoom_max_count > 0) {
                // Plot 2: Set coordinate system and labels
                cpgswin(zoom_min, zoom_max, 0, zoom_max_count * 1.1f);     // Set world coordinates
                cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);                  // Draw coordinate axes
                cpglab("Channel MAD M\\dj\\u", "Number of Channels", "Channel MAD M\\dj\\u Distribution (Zoomed: 0-0.25)");  // Add labels
                
                // Plot 2: Draw zoomed histogram bars
                cpgsci(2); // Red color
                for (i = 0; i < zoom_nbins; i++) {
                    if (zoom_hist[i] > 0) {
                        float x1 = zoom_min + i * zoom_bin_width;
                        float x2 = zoom_min + (i + 1) * zoom_bin_width;
                        cpgrect(x1, x2, 0, zoom_hist[i]);       // Draw each zoomed histogram bar
                    }
                }
                
                // Plot 2: Draw unified 11-threshold line system (zoomed version)
                for (i = 0; i < NUM_ALL_THRESHOLDS; i++) {
                    if (threshold_enabled[i] && all_thresh_values[i] >= zoom_min && all_thresh_values[i] <= zoom_max) {
                        cpgsci(all_threshold_colors[i]);                    // Set threshold line color
                        cpgmove(all_thresh_values[i], 0);                   // Move to starting point
                        cpgdraw(all_thresh_values[i], zoom_max_count * 1.1f);  // Draw vertical line
                        cpgptxt(all_thresh_values[i], zoom_max_count * all_y_positions[i], 0.0, 0.0, all_threshold_labels[i]);  // Add label
                    }
                }
                
                // Plot 2: Draw the Gaussian curve with zoomed histogram specific amplitude
                // Calculate fixed amplitude from zoomed histogram max bin value
                float zoom_hist_amplitude = zoom_max_count;
                printf("Zoomed histogram amplitude (max bin value): %.2f\n", zoom_hist_amplitude);
                
                cpgsci(5); // Cyan color for fit
                cpgsls(2); // Dashed line style
                
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
                
                cpgsls(1); // Back to solid line
                cpgptxt(zoom_min + (zoom_max - zoom_min) * 0.7f, zoom_max_count * 0.85f, 
                        0.0, 0.0, "Gaussian Fit");
                
                cpgsci(1); // Restore white color
                printf("Zoomed MAD histogram completed! (%d channels in 0-0.25 range)\n", zoom_count);
                // *** Plot 2 complete ***
            } else {
                printf("No data found in 0-0.25 range for MAD histogram\n");
            }
            
            free(zoom_hist);
        } else {
            printf("No MAD data in 0-0.25 range, skipping zoomed histogram\n");
        }

        //=============================================================================
        // Both plots completed
        //=============================================================================

        printf("MAD histogram plot completed!\n");
        free(hist);
    }
    
    // Clean up
    free(channel_mad);
    free(channel_median);
    free(temp_data);
}

void visualizeChannelStd(float *data, int nsamp, int nchan, int plot)
{
    int i, j;
    
    // Allocate memory for channel statistics
    float *channel_std = (float *)malloc(nchan * sizeof(float));
    float *channel_median = (float *)malloc(nchan * sizeof(float));
    float *temp_data = (float *)malloc(nsamp * sizeof(float));
    
    // Calculate standard deviation for each channel (using median instead of mean)
    for (i = 0; i < nchan; i++)
    {
        // Copy channel data for processing
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        
        // Calculate median of the channel
        channel_median[i] = median(temp_data, nsamp);
        
        // Calculate squared deviations from median
        if (nsamp > 1) {
            float sum_squared_dev = 0.0f;
            for (j = 0; j < nsamp; j++)
            {
                float deviation = data[i * nsamp + j] - channel_median[i];
                float squared_dev = deviation * deviation;
                temp_data[j] = squared_dev;
                sum_squared_dev += squared_dev;
            }
            
            // Calculate standard deviation as sqrt(mean of squared deviations from median)
            float mean_squared_dev = sum_squared_dev / nsamp;
            channel_std[i] = sqrtf(mean_squared_dev);
            
        } else {
            // If only one sample, set STD to 0
            channel_std[i] = 0.0f;
        }
    }
    
    // Calculate statistics of channel standard deviations (using median-based approach)
    float std_min = channel_std[0], std_max = channel_std[0];
    
    // Find min and max values
    for (i = 0; i < nchan; i++)
    {
        if (channel_std[i] < std_min) std_min = channel_std[i];
        if (channel_std[i] > std_max) std_max = channel_std[i];
    }
    
    // Calculate median of the STD values
    memcpy(temp_data, channel_std, nchan * sizeof(float));
    float std_median = median(temp_data, nchan);
    
    // Calculate STD of STD values (using median absolute deviation approach)
    for (i = 0; i < nchan; i++)
    {
        temp_data[i] = fabsf(channel_std[i] - std_median);
    }
    float std_std = median(temp_data, nchan);
    
    printf("\n=== Channel STD σ_j Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("STD Median: %.6f\n", std_median);
    printf("STD STD: %.6f\n", std_std);
    printf("STD Min: %.6f\n", std_min);
    printf("STD Max: %.6f\n", std_max);
    
    // Unified threshold calculation system
    float multipliers[6] = {-1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float threshold_values[6];
    const char* threshold_names[6] = {"-1*STD", "1*STD", "2*STD", "3*STD", "4*STD", "5*STD"};
    
    // Calculate all thresholds directly
    for (i = 0; i < 6; i++) {
        threshold_values[i] = std_median + multipliers[i] * std_std;
    }
    
    // Print thresholds uniformly
    printf("\n=== Suggested Thresholds ===\n");
    for (i = 0; i < 6; i++) {
        printf("%s threshold: %.6f\n", threshold_names[i], threshold_values[i]);
    }
    
    printThresholdStatistics(channel_std, nchan, threshold_values, threshold_names, 6, "STD");
    
    // =====================================================================
    // Part 1: Draw full-range STD histogram (including all unified threshold lines)
    // =====================================================================
    if (plot)
    {
        // Variable to store Gaussian fitting result for use in both full and zoomed histograms
        float fitted_sigma = 0.0f;
        
        // Use standard histogram bin count
        int nbins = (int)(sqrt(nchan) * 2);
        if (nbins < 30) nbins = 30;
        if (nbins > 100) nbins = 100;
        printf("Using %d bins for STD histogram\n", nbins);
        
        float *hist = (float *)calloc(nbins, sizeof(float));
        float plot_min = std_min;
        float plot_max = std_max;
        
        // Slightly expand range to avoid boundary issues
        float range = plot_max - plot_min;
        if (range > 0) {
            plot_min -= 0.01f * range;
            plot_max += 0.01f * range;
        } else {
            // If all values are the same, create a small range
            plot_min -= 0.001f;
            plot_max += 0.001f;
        }
        float bin_width = (plot_max - plot_min) / nbins;

        printf("STD value range: [%.6f, %.6f], using %d bins\n", std_min, std_max, nbins);

        // Fill histogram
        for (i = 0; i < nchan; i++)
        {
            int bin = (int)((channel_std[i] - plot_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;
            hist[bin]++;
        }

        // Calculate maximum count
        float max_count = 0;
        for (i = 0; i < nbins; i++)
        {
            if (hist[i] > max_count) max_count = hist[i];
        }

        // Draw histogram
        printf("Creating STD histogram plot...\n");
        cpgpage();
        cpgvstd();
        cpgsch(1.2);
        cpgswin(plot_min, plot_max, 0, max_count * 1.1f);
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpglab("Channel STD \\gs\\dj\\u", "Number of Channels", "Channel STD \\gs\\dj\\u Distribution");
        printf("STD histogram axes set up complete\n");

        // Draw solid histogram bars
        cpgsci(2); // Red color
        for (i = 0; i < nbins; i++)
        {
            float x1 = plot_min + i * bin_width;
            float x2 = plot_min + (i + 1) * bin_width;
            cpgrect(x1, x2, 0, hist[i]);
        }

        // === Unified threshold configuration system (similar to MAD function) ===
        // Threshold pair control switches: [±1, ±2, ±3, ±4, ±5]
        int threshold_pair_enabled[5] = {1, 1, 1, 0, 1}; // ±1,±2,±3 enabled, ±4 disabled, ±5 enabled
        int show_median = 1; // Show median line
        
        // Calculate negative thresholds (using existing positive thresholds for symmetric calculation)
        float thresh_neg1std = threshold_values[0];  // Use previously calculated result directly
        float thresh_neg2std = std_median - (threshold_values[2] - std_median);  // Symmetric relative to median
        float thresh_neg3std = std_median - (threshold_values[3] - std_median);
        float thresh_neg4std = std_median - (threshold_values[4] - std_median);
        float thresh_neg5std = std_median - (threshold_values[5] - std_median);
        
        // Unified threshold array configuration
        float all_thresh_values[11] = {
            thresh_neg5std, thresh_neg4std, thresh_neg3std, thresh_neg2std, thresh_neg1std,
            std_median,
            threshold_values[1], threshold_values[2], threshold_values[3], threshold_values[4], threshold_values[5]
        };
        
        char *all_threshold_labels[11] = {
            "-5*STD", "-4*STD", "-3*STD", "-2*STD", "-1*STD",
            "Median",
            "1*STD", "2*STD", "3*STD", "4*STD", "5*STD"
        };
        
        int all_threshold_colors[11] = {4, 7, 3, 8, 6, 1, 6, 8, 3, 7, 4}; // Symmetric coloring
        float all_y_positions[11] = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 0.65f, 0.75f, 0.9f, 1.05f, 1.0f, 0.85f};

        // Draw threshold lines
        int thresh_idx;
        for (thresh_idx = 0; thresh_idx < 11; thresh_idx++) {
            int should_draw = 0;
            
            if (thresh_idx == 5) { // Median line
                should_draw = show_median;
            } else {
                // Threshold pairs: indices 0,1,2,3,4 for negative values, indices 6,7,8,9,10 for positive values
                int pair_idx = (thresh_idx < 5) ? (4 - thresh_idx) : (thresh_idx - 6);
                should_draw = threshold_pair_enabled[pair_idx];
            }
            
            if (should_draw) {
                cpgsci(all_threshold_colors[thresh_idx]);
                cpgmove(all_thresh_values[thresh_idx], 0);
                cpgdraw(all_thresh_values[thresh_idx], max_count * 1.1f);
                cpgptxt(all_thresh_values[thresh_idx], max_count * all_y_positions[thresh_idx], 
                       0.0, 0.0, all_threshold_labels[thresh_idx]);
            }
        }

        cpgsci(1); // Restore white color

        // Add Gaussian fitting curve with fixed amplitude
        printf("Fitting Gaussian curve to STD histogram (fixed amplitude method)...\n");
        
        // Unified Gaussian fitting threshold control (same as MAD function)
        const float gaussian_fit_sigma_threshold = 5.0f;  // Sigma threshold for Gaussian fitting range
        
        // Define sigma range around median for fitting
        float fit_range_min = std_median - gaussian_fit_sigma_threshold * std_std;
        float fit_range_max = std_median + gaussian_fit_sigma_threshold * std_std;
        printf("Gaussian fitting range: [%.6f, %.6f] (median ± %.1fσ)\n", fit_range_min, fit_range_max, gaussian_fit_sigma_threshold);
        
        // Count non-zero bins for debugging
        int non_zero_bins = 0;
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) non_zero_bins++;
        }
        
        // Prepare fitting data: convert histogram to x-y data points within 5-sigma range
        float *x_data = (float *)malloc(nbins * sizeof(float));
        float *y_data = (float *)malloc(nbins * sizeof(float));
        int fit_points = 0;
        
        // Gaussian curve fitting (only use data within median ± configurable σ)
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) {  // Only use non-zero bins for fitting
                float bin_center = plot_min + (i + 0.5f) * bin_width;
                // Only include bins within sigma range around median
                if (bin_center >= fit_range_min && bin_center <= fit_range_max) {
                    x_data[fit_points] = bin_center;
                    y_data[fit_points] = hist[i];
                    fit_points++;
                }
            }
        }
        printf("Using %d out of %d bins for Gaussian fitting (within %.1fσ range)\n", fit_points, non_zero_bins, gaussian_fit_sigma_threshold);
        
        // Simple zero-bin removal: set the first y value to zero to eliminate instrument artifacts
        if (fit_points > 0) {
            printf("Removing potential instrument artifact: y_data[0] = %.1f -> 0.0\n", y_data[0]);
            y_data[0] = 0.0f;
        }
        
        // Calculate fixed amplitude from main histogram max bin value
        float main_hist_amplitude = max_count;
        printf("Main histogram amplitude (max bin value): %.2f\n", main_hist_amplitude);
        
        // Use GSL two-parameter Gaussian fitting with fixed amplitude
        float fitted_mu;
        int gsl_success = gsl_gaussian_fit(x_data, y_data, fit_points, main_hist_amplitude, &fitted_mu, &fitted_sigma);
        
        if (gsl_success) {
            printf("Gaussian fit (fixed amp=%.2f): center=%.6f, sigma=%.6f\n", 
                   main_hist_amplitude, fitted_mu, fitted_sigma);
        } else {
            printf("Gaussian fit failed, falling back to simple fit\n");
            fitted_sigma = simple_curve_fit(x_data, y_data, fit_points, std_median);
            fitted_mu = std_median;
            printf("Fallback Gaussian fit: center=%.6f, sigma=%.6f\n", fitted_mu, fitted_sigma);
        }
        
        // Store fitted parameters for use in zoomed histogram
        float global_fitted_mu = fitted_mu;
        float global_fitted_sigma = fitted_sigma;
        
        // Draw fitted Gaussian curve with fixed amplitude
        cpgsci(5); // Cyan color
        cpgsls(2); // Dashed line
        
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
        
        cpgptxt(global_fitted_mu + global_fitted_sigma, max_count * 0.4f, 0.0, 0.0, "Gaussian Fit");
        cpgsls(1);

        free(x_data);
        free(y_data);

        cpgsci(1); // Restore white color

        // =====================================================================
        // Second part: Draw zoomed STD histogram for 0-0.25 range (detailed view)
        // =====================================================================
        printf("Creating zoomed STD histogram for range 0-0.25...\n");
        
        // Check if there is data in the 0-0.25 range
        int has_data_in_range = 0;
        for (i = 0; i < nchan; i++) {
            if (channel_std[i] >= 0.0f && channel_std[i] <= 0.25f) {
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
            int zoom_nbins = 50; // Use more bins for higher resolution
            float zoom_min = 0.0f;
            float zoom_max = 0.25f;
            float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
            float *zoom_hist = (float *)calloc(zoom_nbins, sizeof(float));
            
            // Fill zoomed histogram
            int zoom_count = 0;
            for (i = 0; i < nchan; i++) {
                if (channel_std[i] >= zoom_min && channel_std[i] <= zoom_max) {
                    int bin = (int)((channel_std[i] - zoom_min) / zoom_bin_width);
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
                cpglab("Channel STD \\gs\\dj\\u", "Number of Channels", "Channel STD \\gs\\dj\\u Distribution (Zoomed: 0-0.25)");
                
                // Draw zoomed histogram bars
                cpgsci(2); // Red color
                for (i = 0; i < zoom_nbins; i++) {
                    if (zoom_hist[i] > 0) {
                        float x1 = zoom_min + i * zoom_bin_width;
                        float x2 = zoom_min + (i + 1) * zoom_bin_width;
                        cpgrect(x1, x2, 0, zoom_hist[i]);
                    }
                }
                
                // === Use unified system to draw threshold lines in zoomed plot ===
                for (thresh_idx = 0; thresh_idx < 11; thresh_idx++) {
                    int should_draw = 0;
                    
                    if (thresh_idx == 5) { // Median line
                        should_draw = show_median;
                    } else {
                        // Threshold pairs: indices 0,1,2,3,4 for negative values, indices 6,7,8,9,10 for positive values
                        int pair_idx = (thresh_idx < 5) ? (4 - thresh_idx) : (thresh_idx - 6);
                        should_draw = threshold_pair_enabled[pair_idx];
                    }
                    
                    if (should_draw && all_thresh_values[thresh_idx] >= zoom_min && all_thresh_values[thresh_idx] <= zoom_max) {
                        cpgsci(all_threshold_colors[thresh_idx]);
                        cpgmove(all_thresh_values[thresh_idx], 0);
                        cpgdraw(all_thresh_values[thresh_idx], zoom_max_count * 1.1f);
                        // Adjust y position to fit zoomed plot
                        float zoom_y_pos = 0.1f + (thresh_idx * 0.08f);
                        if (zoom_y_pos > 0.9f) zoom_y_pos = 0.9f;
                        cpgptxt(all_thresh_values[thresh_idx], zoom_max_count * zoom_y_pos, 
                               0.0, 0.0, all_threshold_labels[thresh_idx]);
                    }
                }
                
                // Draw the Gaussian curve with zoomed histogram specific amplitude
                // Calculate fixed amplitude from zoomed histogram max bin value
                float zoom_hist_amplitude = zoom_max_count;
                printf("Zoomed histogram amplitude (max bin value): %.2f\n", zoom_hist_amplitude);
                
                cpgsci(5); // Cyan color for fit
                cpgsls(2); // Dashed line style
                
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
                
                cpgsls(1); // Back to solid line
                cpgptxt(zoom_min + (zoom_max - zoom_min) * 0.7f, zoom_max_count * 0.85f, 
                        0.0, 0.0, "Gaussian Fit");
                
                cpgsci(1); // Restore white color
                printf("Zoomed STD histogram completed! (%d channels in 0-0.25 range)\n", zoom_count);
            } else {
                printf("No data found in 0-0.25 range for STD histogram\n");
            }
            
            free(zoom_hist);
        } else {
            printf("No STD data in 0-0.25 range, skipping zoomed histogram\n");
        }

        printf("STD histogram plot completed!\n");
        free(hist);
    }
    
    // Clean up
    free(channel_std);
    free(channel_median);
    free(temp_data);
    
    printf("=== STD Histogram Complete ===\n");
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

void identSubstNSigma(
    float *data, int nsamp, int nchan, float Nsigma, int iterationIndex, int plot,
    int *horizontalMask, int *verticalMask, int *globalMask,
    float *finalMedian, float *finalStd, int cudaReady)
{
    // Debug output at function entry
    printf("### DEBUG: identSubstNSigma called with iterationIndex=%d, plot=%d ###\n", iterationIndex, plot);
    printf("### Parameters: nsamp=%d, nchan=%d, Nsigma=%.2f ###\n", nsamp, nchan, Nsigma);
    fflush(stdout);  // Ensure immediate output
    
    memset(horizontalMask, 0, nsamp * nchan * sizeof(int));
    memset(verticalMask, 0, nsamp * nchan * sizeof(int));
    memset(globalMask, 0, nsamp * nchan * sizeof(int));

    int *good_samples = (int *)malloc(nsamp * sizeof(int));
    int *random_indices = (int *)malloc(nsamp * sizeof(int));
    float *median_temp = (float *)malloc(nsamp * nchan * sizeof(float));
    memcpy(median_temp, data, nsamp * nchan * sizeof(float));
    
    float lastMean = 0.0f, lastStd = 0.0f;
    float mean = 0.0f, std = 0.0f, med = 0.0f;
    float meanDiff = 0.0f, stdDiff = 0.0f;
    float upperBound, lowerBound;
    
    int totReplaceCnt = 0;
    float killThresh = 0.2f;  // 10% pixel ratio threshold
    int i, j;

    // subtractChannelMedians(data, nsamp, nchan);
    
    // Visualize channel MAD statistics for threshold determination in first 20 iterations
    // Use iterationIndex as a proxy for iteration counter (passed from ReadFASTData.c)
    if (plot)
    {
        printf("=== Generating Channel MAD M_j Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelMAD(data, nsamp, nchan, 1);
        printf("=== MAD M_j Histogram Complete ===\n");
        
        printf("=== Generating Channel STD σ_j Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelStd(data, nsamp, nchan, 1);
        printf("=== STD σ_j Histogram Complete ===\n");
    }
    
    // === 1. Channel level flagging first ===
    float *channel_stds = (float *)malloc(nchan * sizeof(float));
    float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
    flagChannelsByStdOutliers(data, nsamp, nchan, horizontalMask, channel_stds, channel_stds_temp);
    free(channel_stds);
    free(channel_stds_temp);
    
    // Check which channels are fully flagged after channel-level detection
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
    printf("Channel-level flagging: %d/%d channels fully flagged (%.2f%%), skipping pixel-level detection for these\n", 
           fully_flagged_channels, nchan, (float)fully_flagged_channels/nchan*100);
    
    // === 2. Channel-internal pixel flagging ===
    // Skip pixel-level detection for channels that are already fully flagged
    #pragma omp parallel for reduction(+:totReplaceCnt)
    for (i = 0; i < nchan; i++)
    {
        // Skip this channel if it's already fully flagged
        if (channel_fully_flagged[i]) {
            continue;
        }
        
        int iter = 0;
        while (1)
        {
            lastMean = mean;
            lastStd = std;
            
            findMeanStd(data + i * nsamp, nsamp, &mean, &std);
            med = median(median_temp + i * nsamp, nsamp);
            meanDiff = fabsf(mean - lastMean) / lastMean;
            stdDiff = fabsf(std - lastStd) / lastStd;
            // medianDiff removed (not used)

            float scale_row = 1.0f;
            upperBound = med + Nsigma * scale_row * std;
            lowerBound = med - Nsigma * scale_row * std;

            for (j = 0; j < nsamp; j++)
            {
                int idx = j + i * nsamp;
                if (data[idx] > upperBound || data[idx] < lowerBound)
                {
                    horizontalMask[idx] = 1;
                    totReplaceCnt++;
                }
            }
            
            substitute_pixels_1d(data + i * nsamp, nsamp, horizontalMask + i * nsamp,
                                 good_samples, random_indices);

            // if (plot && (i == 0))
                // calc8bitHist(data + i * nsamp, nsamp);
            iter++;
            if (iter > 3)
                break;
        }
    }
    free(good_samples);
    free(random_indices);
    free(channel_fully_flagged); // Clean up the channel tracking array

    totReplaceCnt = 0;
    float *transposedData = (float *)malloc(nsamp * nchan * sizeof(float));
    int *transposedMask = (int *)calloc(nsamp * nchan, sizeof(int));
    int *good_samples_v = (int *)malloc(nsamp * sizeof(int));
    int *random_indices_v = (int *)malloc(nsamp * sizeof(int));

    transpose(data, nsamp, nchan, transposedData);
    memcpy(median_temp, transposedData, nsamp * nchan * sizeof(float));
    transpose_int(horizontalMask, nsamp, nchan, transposedMask);

    // Process each time sample
    #pragma omp parallel for reduction(+:totReplaceCnt)
    for (i = 0; i < nsamp; i++)
    {
        int iter = 0;
        while (1)
        {
            lastMean = mean;
            lastStd = std;
            
            findMeanStd(transposedData + i * nchan, nchan, &mean, &std);
            med = median(median_temp + i * nchan, nchan);
            meanDiff = fabsf(mean - lastMean) / lastMean;
            stdDiff = fabsf(std - lastStd) / lastStd;
            // medianDiff removed (not used)

            // float scale_col = sqrtf(n_ref / nchan);
            float scale_col = 1.0f;
            upperBound = med + Nsigma * scale_col * std;
            lowerBound = med - Nsigma * scale_col * std;
            
            for (j = 0; j < nchan; j++)
            {
                int idx = i * nchan + j;
                if (transposedData[idx] > upperBound || transposedData[idx] < lowerBound)
                {
                    transposedMask[idx] = 1;
                    totReplaceCnt++;
                }
            }
            
            substitute_pixels_1d(transposedData + i * nchan, nchan, transposedMask + i * nchan,
                                    good_samples_v, random_indices_v);

            // if (plot && (i == 0))
            //     calc8bitHist(transposedData + i * nchan, nchan);
            iter++;
            if (iter > 3)
                break;
        }
    }

    transpose(transposedData, nchan, nsamp, data);
    transpose_int(transposedMask, nchan, nsamp, verticalMask);

    free(transposedData);
    free(transposedMask);
    free(good_samples_v);
    free(random_indices_v);
    
    // Debug: Check horizontal and vertical mask statistics before combining
    int horizontalFlagged = 0, verticalFlagged = 0;
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (horizontalMask[idx] == 1) horizontalFlagged++;
        if (verticalMask[idx] == 1) verticalFlagged++;
    }
    printf("\n=== RFI Detection Statistics ===\n");
    printf("Horizontal mask flagged: %d/%d pixels (%.4f%%)\n", 
           horizontalFlagged, nsamp*nchan, (float)horizontalFlagged/(nsamp*nchan)*100);
    printf("Vertical mask flagged: %d/%d pixels (%.4f%%)\n", 
           verticalFlagged, nsamp*nchan, (float)verticalFlagged/(nsamp*nchan)*100);
    
    logicalOR(horizontalMask, verticalMask, globalMask, nsamp, nchan);
    
    // Debug: Check globalMask immediately after logicalOR
    int globalFlagged = 0;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) globalFlagged++;
    }
    printf("Global mask flagged after logicalOR: %d/%d pixels (%.4f%%)\n", 
           globalFlagged, nsamp*nchan, (float)globalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");

    // Apply binarySIR before killThresh analysis to filter isolated pixels for better range calculation
    printf("\n=== Applying binarySIR filtering before killThresh analysis ===\n");
    int flaggedBeforeSIR = globalFlagged;
    
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

    // === 3. Heavy RFI channel marking (killThresh - flag heavily contaminated channels) ===
    // Apply killThresh: if a channel has more than killThresh fraction of flagged pixels, flag the entire channel
    // Add position range check: avoid killing channels with concentrated local RFI
    printf("\n=== killThresh Analysis (threshold=%.3f) ===\n", killThresh);
    int killedChannels = 0;
    int totalFlaggedBefore = flaggedAfterSIR; // Use count after binarySIR filtering
    int totalFlaggedAfter = 0;
    int localRFISkipped = 0;  // Count channels skipped due to local RFI pattern
    float rangeThreshold = 0.5f;  // If flagged pixels span <30% of channel, don't kill entire channel
    int chan;

    #pragma omp parallel for reduction(+:killedChannels,localRFISkipped)
    for (chan = 0; chan < nchan; chan++) {
        int maskedCount = 0;
        int firstFlagged = -1, lastFlagged = -1;
        int samp;
        
        // First pass: count flagged pixels and find range
        for (samp = 0; samp < nsamp; samp++) {
            int idx = samp + chan * nsamp; // Corrected indexing to match original
            if (globalMask[idx]) {
                maskedCount++;
                if (firstFlagged == -1) firstFlagged = samp;  // First flagged position
                lastFlagged = samp;  // Update last flagged position
            }
        }
        
        float maskedRatio = (float)maskedCount / nsamp;
        int shouldKillChannel = 0;  // Use int instead of bool for compatibility
        
        if (maskedRatio > killThresh) {
            // Calculate the range of flagged pixels
            if (firstFlagged != -1 && lastFlagged != -1) {
                int flaggedRange = lastFlagged - firstFlagged + 1;
                float rangeRatio = (float)flaggedRange / nsamp;
                
                // Only kill channel if flagged pixels span a significant portion of the channel
                if (rangeRatio >= rangeThreshold) {
                    shouldKillChannel = 1;
                } else {
                    localRFISkipped++;
                }
                
                // Print detailed info for channels with significant flagging
                if (maskedRatio > 0.001f) { // Print if >0.1% flagged
                    #pragma omp critical
                    {
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
                        printf("\n");
                    }
                }
            } else {
                // This shouldn't happen if maskedCount > 0, but handle it
                shouldKillChannel = 1;
            }
        }
        
        if (shouldKillChannel) {
            killedChannels++;
            for (samp = 0; samp < nsamp; samp++) {
                int idx = samp + chan * nsamp;
                globalMask[idx] = 1;
            }
        }
    }
    
    // Count total flagged pixels after killThresh
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) totalFlaggedAfter++;
    }
    
    printf("killThresh Summary:\n");
    printf("  - Killed channels: %d/%d (%.2f%%)\n", killedChannels, nchan, (float)killedChannels/nchan*100);
    printf("  - Localized RFI skipped: %d/%d (%.2f%%)\n", localRFISkipped, nchan, (float)localRFISkipped/nchan*100);
    printf("  - Range threshold: %.1f%% (flagged pixels must span >%.1f%% of channel to kill)\n", rangeThreshold*100, rangeThreshold*100);
    printf("  - Flagged pixels before: %d/%d (%.2f%%)\n", totalFlaggedBefore, nsamp*nchan, (float)totalFlaggedBefore/(nsamp*nchan)*100);
    printf("  - Flagged pixels after: %d/%d (%.2f%%)\n", totalFlaggedAfter, nsamp*nchan, (float)totalFlaggedAfter/(nsamp*nchan)*100);
    printf("  - Additional pixels flagged: %d\n", totalFlaggedAfter - totalFlaggedBefore);
    printf("=== End killThresh Analysis ===\n\n");

    free(median_temp);
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

void identSubstNSigma_Experiment(
    float *data, int nsamp, int nchan, float Nsigma, int iterationIndex, int plot,
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
    
    subtractChannelMedians(data, nsamp, nchan);
    
    // Visualize channel MAD statistics for threshold determination in first 20 iterations
    // Use iterationIndex as a proxy for iteration counter (passed from ReadFASTData.c)
    if (plot)
    {
        printf("=== Generating Channel MAD M_j Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelMAD(data, nsamp, nchan, 1);
        printf("=== MAD M_j Histogram Complete ===\n");
        
        printf("=== Generating Channel STD σ_j Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelStd(data, nsamp, nchan, 1);
        printf("=== STD σ_j Histogram Complete ===\n");
    }
    
    // === 1. Channel level flagging first ===
    float *channel_stds = (float *)malloc(nchan * sizeof(float));
    float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
    flagChannelsByStdOutliers(data, nsamp, nchan, horizontalMask, channel_stds, channel_stds_temp);
    free(channel_stds);
    free(channel_stds_temp);
    
    // Check which channels are fully flagged after channel-level detection
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
    printf("Channel-level flagging: %d/%d channels fully flagged (%.2f%%), skipping pixel-level detection for these\n", 
           fully_flagged_channels, nchan, (float)fully_flagged_channels/nchan*100);
    
    // === 2. Channel-internal pixel flagging ===
    printf("=== Performing channel-level outlier detection ===\n");
    int channelOutliers = performChannelLevelDetection(data, nsamp, nchan, Nsigma, 
                                                     horizontalMask, channel_fully_flagged);
    printf("Channel-level detection: flagged %d outlier pixels\n", channelOutliers);
    
    free(good_samples);
    free(random_indices);
    free(channel_fully_flagged);

    // === 3. Time-sample level flagging ===
    float *transposedData = (float *)malloc(nsamp * nchan * sizeof(float));
    int *transposedMask = (int *)calloc(nsamp * nchan, sizeof(int));

    // Transpose data and mask for time-sample processing
    transpose(data, nsamp, nchan, transposedData);
    transpose_int(horizontalMask, nsamp, nchan, transposedMask);

    printf("=== Performing time-sample-level outlier detection ===\n");
    int timeSampleOutliers = performTimeSampleLevelDetection(transposedData, nsamp, nchan, Nsigma, transposedMask);
    printf("Time-sample-level detection: flagged %d outlier pixels\n", timeSampleOutliers);

    // Transpose back to original layout
    transpose(transposedData, nchan, nsamp, data);
    transpose_int(transposedMask, nchan, nsamp, verticalMask);
    free(transposedData);
    free(transposedMask);
    
    int horizontalFlagged = 0, verticalFlagged = 0;
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (horizontalMask[idx] == 1) horizontalFlagged++;
        if (verticalMask[idx] == 1) verticalFlagged++;
    }
    printf("\n=== RFI Detection Statistics ===\n");
    printf("Horizontal mask flagged: %d/%d pixels (%.4f%%)\n", 
           horizontalFlagged, nsamp*nchan, (float)horizontalFlagged/(nsamp*nchan)*100);
    printf("Vertical mask flagged: %d/%d pixels (%.4f%%)\n", 
           verticalFlagged, nsamp*nchan, (float)verticalFlagged/(nsamp*nchan)*100);
    
    logicalOR(horizontalMask, verticalMask, globalMask, nsamp, nchan);
    
    int globalFlagged = 0;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) globalFlagged++;
    }
    printf("Global mask flagged after logicalOR: %d/%d pixels (%.4f%%)\n", 
           globalFlagged, nsamp*nchan, (float)globalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");

    // Apply binarySIR before killThresh analysis to filter isolated pixels for better range calculation
    printf("\n=== Applying binarySIR filtering before killThresh analysis ===\n");
    int flaggedBeforeSIR = globalFlagged;
    
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

    // === 3. Heavy RFI channel marking (killThresh - flag heavily contaminated channels) ===
    int killedChannels, localRFISkipped, totalFlaggedAfter;
    applyKillThresh(globalMask, nsamp, nchan, killThresh, flaggedAfterSIR, 
                   &killedChannels, &localRFISkipped, &totalFlaggedAfter);
    
    // Print killThresh analysis summary
    printKillThreshSummary(killedChannels, localRFISkipped, flaggedAfterSIR, 
                          totalFlaggedAfter, nsamp, nchan);

    // Calculate final statistics for experimental function output
    float finalMedian_temp, finalStd_temp;
    findMeanStd(median_temp, nsamp * nchan, &finalMedian_temp, &finalStd_temp);
    float finalMedian_value = median(median_temp, nsamp * nchan);

    float outlierRatio = (float)(channelOutliers + timeSampleOutliers) / (nsamp * nchan);
    if (outlierRatio > killThresh)
    {
        printf("WARNING: High outlier ratio %.4f > %.2f detected - data may be corrupted\n",
               outlierRatio, killThresh);
    }

    *finalMedian = finalMedian_value;
    *finalStd = finalStd_temp;

    free(median_temp);

    printf("### DEBUG: identSubstNSigma_Experiment exiting with finalMedian=%.6f, finalStd=%.6f ###\n", *finalMedian, *finalStd);
    fflush(stdout);  // Ensure immediate output
}