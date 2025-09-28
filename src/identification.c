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

    // Find histogram amplitude (max value)
    float max_amplitude = hist[0];
    for (i = 1; i < bins; i++) {
        if (hist[i] > max_amplitude) {
            max_amplitude = hist[i];
        }
    }

    // Use GSL to jointly fit mean and sigma
    float fitted_mu, fitted_sigma;
    int gsl_success = gsl_gaussian_fit(x_val, hist, bins, max_amplitude, &fitted_mu, &fitted_sigma);
    
    if (gsl_success) {
        return fitted_sigma;
    } else {
        // If GSL fails, return a default sigma estimate
        printf("Warning: GSL fitting failed in ksigma_1d, returning default sigma\n");
        return (max_val - min_val) / 6.0f; // rough estimate: range/6
    }
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

    // Pre-compute M and chi_i values (corrected formula)
    int i, j, e, m;
    for (i = 0; i < M_len; i++) {
        int m_val = i + 1;  // m starts from 1
        M[i] = powf(2.0f, (float)(m_val - 1));  // M = 2^(m-1): 1, 2, 4, 8, 16, 32
        chi_i[i] = chi_1 / powf(p, log2f((float)m_val));  // Use m_val instead of M[i]
    }

    memcpy(temp_data, data, length * sizeof(float));
    memset(local_mask, 0, length * sizeof(int));

    // Main thresholding logic
    for (e = 0; e < eta_len; e++) {
        float current_eta = eta_i[e];
        for (m = 0; m < M_len; m++) {
            int window = (int)M[m];
            float threshold = chi_i[m] / current_eta;

            if (window == 1) {
                // Special case for m=1: direct threshold comparison
                for (i = 0; i < length; i++) {
                    if (!mask[i] && fabsf(temp_data[i]) > threshold) {
                        local_mask[i] = 1;
                    }
                }
            } else {
                // Window processing for m > 1
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

    // Global normalization and background removal
    float global_mean, global_std;
    findMeanStd(temp_dataT, nsamp * nchan, &global_mean, &global_std);
    
    // Background removal using Gaussian smoothing (parameters from Python reference)
    float *smoothed_data = (float *)malloc(nsamp * nchan * sizeof(float));
    
    // Gaussian smoothing parameters (matching sum_threshold.py)
    float sigma_m = 7.5f;  
    int kernel_m = 40;    
    
    // Pre-compute Gaussian kernel for time axis (sigma_m)
    int half_kernel_m = kernel_m / 2;
    float *gaussian_kernel_m = (float *)malloc(kernel_m * sizeof(float));
    float kernel_sum_m = 0.0f;
    for (int k = 0; k < kernel_m; k++) {
        int offset = k - half_kernel_m;
        gaussian_kernel_m[k] = expf(-(offset * offset) / (2.0f * sigma_m * sigma_m));
        kernel_sum_m += gaussian_kernel_m[k];
    }
    // Normalize kernel
    for (int k = 0; k < kernel_m; k++) {
        gaussian_kernel_m[k] /= kernel_sum_m;
    }
    
    // Apply Gaussian smoothing along time axis (for each channel)
    #pragma omp parallel for
    for (j = 0; j < nchan; j++) {
        for (i = 0; i < nsamp; i++) {
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = 0; k < kernel_m; k++) {
                int sample_idx = i + k - half_kernel_m;
                if (sample_idx >= 0 && sample_idx < nsamp && !mask_chanRFI[j * nsamp + sample_idx]) {
                    weighted_sum += temp_dataT[j * nsamp + sample_idx] * gaussian_kernel_m[k];
                    weight_sum += gaussian_kernel_m[k];
                }
            }
            smoothed_data[j * nsamp + i] = (weight_sum > 0) ? weighted_sum / weight_sum : temp_dataT[j * nsamp + i];
        }
    }
    
    free(gaussian_kernel_m);
    
    // Use residual data for thresholding
    #pragma omp parallel for collapse(2)
    for (j = 0; j < nchan; j++) {
        for (i = 0; i < nsamp; i++) {
            temp_dataT[j * nsamp + i] = (temp_dataT[j * nsamp + i] - smoothed_data[j * nsamp + i]) / (global_std + 1e-6f);
        }
    }
    
    free(smoothed_data);
    
    // Calculate chi_1 after background removal
    float chi_1 = timesOfSigma * ksigma_2d(temp_dataT, mask_chanRFI, nsamp, nchan);

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

// Function to randomly replace flagged pixels with unflagged pixels from the same time sample
void randomReplaceRFIPixels(float *data, const int *mask, int nsamp, int nchan)
{
    int i, j;
    
    // Initialize random seed once
    srand((unsigned int)time(NULL));
    
    #pragma omp parallel for private(j)
    for (i = 0; i < nsamp; i++) {
        // For each time sample (column), collect unflagged pixel values
        float *unflagged_values = (float*)malloc(nchan * sizeof(float));
        int unflagged_count = 0;
        
        // Collect unflagged values in this time sample
        for (j = 0; j < nchan; j++) {
            if (!mask[j * nsamp + i]) {
                unflagged_values[unflagged_count++] = data[j * nsamp + i];
            }
        }
        
        // If we have unflagged values, use them to replace flagged pixels
        if (unflagged_count > 0) {
            // Create thread-local random state for thread safety
            unsigned int seed = (unsigned int)(i + time(NULL) + omp_get_thread_num());
            
            for (j = 0; j < nchan; j++) {
                if (mask[j * nsamp + i]) {
                    // Randomly select from unflagged values in this time sample
                    int random_idx = rand_r(&seed) % unflagged_count;
                    data[j * nsamp + i] = unflagged_values[random_idx];
                }
            }
        } else {
            // If no unflagged values in this time sample, set flagged values to 0
            for (j = 0; j < nchan; j++) {
                if (mask[j * nsamp + i]) {
                    data[j * nsamp + i] = 0.0f;
                }
            }
        }
        
        free(unflagged_values);
    }
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



/// @brief Flag channels based on their standard deviation statistics using iterative 3-sigma outlier detection
/// @param data Input data array (nsamp * nchan)
/// @param nsamp Number of time samples
/// @param nchan Number of frequency channels
/// @param horizontalMask Output mask to mark flagged channels
/// @param channel_stds Pre-allocated array to store channel standard deviations
/// @param channel_stds_temp Pre-allocated temporary array for median calculation
/// @param channel_std_threshold Sigma threshold for outlier detection (T_chan)
/// @param nsigma_in Sigma threshold for inChannel detection (T_point)
void outChanDetection(float *data, int nsamp, int nchan, int *channelFlagged,
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold, float nsigma_in)
{
    const int MAX_ITERATIONS = 100;   // Maximum iterations for convergence
    const float STD_CHANGE_THRESHOLD = 0.0001f;  // Standard deviation change rate threshold (0.01%)
    const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;  // Median change threshold
    
    int i;
    
    // Save initial channel statistics for comparison
    float *initial_channel_stds = (float *)malloc(nchan * sizeof(float));
    int initial_flagged_count = 0;
    
    // Count initially flagged channels
    for (i = 0; i < nchan; i++) {
        channelFlagged[i] = 0;  // Initialize to 0
    }
    
    // Calculate standard deviation for each channel
    printf("    [DEBUG] outChannel: Calculating channel standard deviations\n");
    for (i = 0; i < nchan; i++)
    {
        float channel_mean, channel_std;
        findMeanStd(data + i * nsamp, nsamp, &channel_mean, &channel_std);
        channel_stds[i] = channel_std;
        initial_channel_stds[i] = channel_std; // Save initial value for comparison
    }
    
    // Create a mask for channels (0 = valid, 1 = flagged)
    int *channel_mask = (int *)calloc(nchan, sizeof(int));
    float *valid_stds = (float *)malloc(nchan * sizeof(float));
    
    float last_median = 0.0f, last_std = 0.0f;
    float current_median, current_std;
    float median_change, std_change_rate;
    int total_flagged = 0;
    int iter = 0;
    
    // Extract valid channels for initial statistics
    int valid_count = 0;
    for (i = 0; i < nchan; i++) {
        if (channel_mask[i] == 0) {  // Valid channel
            valid_stds[valid_count] = channel_stds[i];
            valid_count++;
        }
    }
    
    if (valid_count < 3) {
        printf("    [DEBUG] outChannel: Too few valid channels (%d < 3), skipping\n", valid_count);
        free(channel_mask);
        free(valid_stds);
        return;
    }
    
    printf("    [DEBUG] outChannel: Starting iterative outlier detection: initial_channels=%d/%d (%.1f%%)\n", 
           valid_count, nchan, (float)valid_count/nchan*100);
    
    while (iter < MAX_ITERATIONS && valid_count >= 3)
    {
        // Calculate current median of channel standard deviations
        current_median = median(valid_stds, valid_count);
        
        // Calculate standard deviation of channel standard deviations using stdFromMedian
        current_std = stdFromMedian(valid_stds, valid_count);
        
        // Check convergence conditions after first iteration
        int converged = 0;
        if (iter > 0) {
            median_change = fabsf(current_median - last_median);
            std_change_rate = (last_std > 0) ? fabsf(current_std - last_std) / last_std : 0.0f;
            
            if (median_change < MEDIAN_CHANGE_THRESHOLD && std_change_rate < STD_CHANGE_THRESHOLD) {
                converged = 1;
            }
        }
        
        // Calculate bounds
        float upper_bound = current_median + channel_std_threshold * current_std;
        float lower_bound = current_median - channel_std_threshold * current_std;
        
        // Flag new outlier channels and rebuild valid data array
        int new_outliers = 0;
        valid_count = 0;  // Reset and rebuild
        
        #pragma omp parallel for reduction(+:new_outliers)
        for (i = 0; i < nchan; i++) {
            if (channel_mask[i] == 0) {  // Currently valid
                if (channel_stds[i] > upper_bound || channel_stds[i] < lower_bound) {
                    channel_mask[i] = 1;  // Flag as outlier
                    channelFlagged[i] = 1;  // Set channel flagged
                    new_outliers++;
                }
            }
        }
        
        // Rebuild valid_stds array sequentially
        for (i = 0; i < nchan; i++) {
            if (channel_mask[i] == 0) {
                valid_stds[valid_count] = channel_stds[i];
                valid_count++;
            }
        }
        
        total_flagged += new_outliers;
        
        // === Centralized Debug Output ===
        printf("    [DEBUG] outChannel Iter %d: valid=%d, median=%.6f, std=%.6f", 
               iter + 1, valid_count + new_outliers, current_median, current_std);
        if (iter > 0) {
            printf(", med_change=%.8f, std_change_rate=%.6f", median_change, std_change_rate);
        }
        printf("\n    [DEBUG]      bounds=[%.6f, %.6f], flagged=%d, remaining=%d/%d (%.1f%%)", 
               lower_bound, upper_bound, new_outliers, valid_count, nchan, (float)valid_count/nchan*100);
        
        if (converged) {
            printf(" -> CONVERGED (median stable & std rate < %.3f)\n", STD_CHANGE_THRESHOLD);
            break;
        } else if (new_outliers == 0) {
            printf(" -> CONVERGED (no new outliers)\n");
            break;
        } else {
            printf("\n");
        }
        
        // Update for next iteration
        last_median = current_median;
        last_std = current_std;
        iter++;
    }
    
    // Final summary
    if (iter >= MAX_ITERATIONS) {
        printf("    [DEBUG] outChannel -> STOPPED (max iterations %d reached)\n", MAX_ITERATIONS);
    } else if (valid_count < 3) {
        printf("    [DEBUG] outChannel -> STOPPED (too few valid channels: %d < 3)\n", valid_count);
    }
    
    printf("    [DEBUG] outChannel Final result: flagged %d channels in %d iterations, remaining_valid=%d/%d (%.1f%%)\n",
           total_flagged, iter, valid_count, nchan, (float)valid_count/nchan*100);
    
    // Count final flagged channels
    int final_flagged_count = 0;
    for (i = 0; i < nchan; i++) {
        if (channelFlagged[i]) final_flagged_count++;
    }
    
    // Create final statistics array (mark flagged channels with negative values)
    float *final_channel_stds = (float *)malloc(nchan * sizeof(float));
    #pragma omp parallel for
    for (i = 0; i < nchan; i++) {
        if (channelFlagged[i]) {
            final_channel_stds[i] = -1.0f; // Mark as flagged
        } else {
            final_channel_stds[i] = channel_stds[i]; // Keep current value
        }
    }
    
    // Display comparison histogram if significant changes occurred
    if (total_flagged > 0) {
        printf("\n=== Displaying outChannel Detection Comparison ===\n");
        drawOutChannelComparisonHist(initial_channel_stds, final_channel_stds, nchan, 
                                   0, initial_flagged_count, final_flagged_count, iter, channel_std_threshold, nsigma_in); // Using STD (0)
        printf("=== OutChannel Comparison Complete ===\n");
    } else {
        printf("    [DEBUG] outChannel: No channels flagged, skipping comparison histogram\n");
    }
    
    // Cleanup
    free(initial_channel_stds);
    free(final_channel_stds);
    free(channel_mask);
    free(valid_stds);
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
                // Calculate STD (Standard Deviation from median) using stdFromMedian function
                channel_stat[i] = stdFromMedian(data + i * nsamp, nsamp);
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
        printf("Error: GSL Gaussian fit failed\n");
        // Set default values to prevent crashes
        fitted_mu = stat_median;
        fitted_sigma = stat_mad > 0 ? stat_mad : 1.0f;
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
 * Uses configurable N-sigma threshold with median-based robust statistics
 * @param data Channel data array (nsamp length)
 * @param nsamp Number of samples in the channel
 * @param Nsigma Sigma threshold for outlier detection
 * @param mask Output mask for this channel (nsamp length)
 * @param median_temp Temporary array for median calculation (nsamp length)
 * @param good_samples Temporary array for good sample indices (nsamp length)
 * @param random_indices Temporary array for random indices (nsamp length)
 * @return Total number of outliers detected
 */
int inChanOutlierIter(float *data, int nsamp, float Nsigma, 
                                   int *mask, float *median_temp,
                                   int *good_samples, int *random_indices)
{
    const int MAX_ITERATIONS = 15;   // Increased maximum iterations
    const float STD_CHANGE_THRESHOLD = 0.0001f;  // Standard deviation change rate threshold (0.01%)
    const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;  // Median change threshold
    
    float last_median = 0.0f, last_std = 0.0f;
    float current_median, current_std;
    float median_change, std_change_rate;
    int total_outliers = 0;
    int iter = 0;
    
    // Extract valid (unmasked) data for initial statistics
    int valid_count = 0;
    for (int i = 0; i < nsamp; i++) {
        if (mask[i] == 0) {  // Unmasked data
            median_temp[valid_count] = data[i];
            valid_count++;
        }
    }
    
    if (valid_count < 3) {
        // Too few samples for meaningful statistics
        printf("    [DEBUG] Channel has too few valid samples (%d < 3), skipping\n", valid_count);
        return 0;
    }
    
    printf("    [DEBUG] Starting iterative outlier detection: initial_samples=%d/%d (%.1f%%)\n", 
           valid_count, nsamp, (float)valid_count/nsamp*100);
    
    while (iter < MAX_ITERATIONS && valid_count >= 3)
    {
        // Calculate current median
        current_median = median(median_temp, valid_count);
        
        // Calculate standard deviation from median (robust approach) using stdFromMedian
        current_std = stdFromMedian(median_temp, valid_count);
        
        // Check convergence conditions after first iteration
        int converged = 0;
        if (iter > 0) {
            median_change = fabsf(current_median - last_median);
            std_change_rate = (last_std > 0) ? fabsf(current_std - last_std) / last_std : 0.0f;
            
            if (median_change < MEDIAN_CHANGE_THRESHOLD && std_change_rate < STD_CHANGE_THRESHOLD) {
                converged = 1;
            }
        }
        
        // Calculate N-sigma bounds
        float upper_bound = current_median + Nsigma * current_std;
        float lower_bound = current_median - Nsigma * current_std;
        
        // Flag new outliers and rebuild valid data array
        int new_outliers = 0;
        valid_count = 0;  // Reset and rebuild
        
        for (int i = 0; i < nsamp; i++) {
            if (mask[i] == 0) {  // Currently unmasked
                if (data[i] > upper_bound || data[i] < lower_bound) {
                    mask[i] = 1;  // Flag as outlier
                    new_outliers++;
                    total_outliers++;
                } else {
                    median_temp[valid_count] = data[i];  // Keep in valid data
                    valid_count++;
                }
            }
        }
        
        // === Centralized Debug Output ===
        printf("    [DEBUG] Iter %d: valid=%d, median=%.6f, std=%.6f", 
               iter + 1, valid_count + new_outliers, current_median, current_std);
        if (iter > 0) {
            printf(", med_change=%.8f, std_change_rate=%.6f", median_change, std_change_rate);
        }
        printf("\n    [DEBUG]      bounds=[%.6f, %.6f], flagged=%d, remaining=%d/%d (%.1f%%)", 
               lower_bound, upper_bound, new_outliers, valid_count, nsamp, (float)valid_count/nsamp*100);
        
        if (converged) {
            printf(" -> CONVERGED (median stable & std rate < %.3f)\n", STD_CHANGE_THRESHOLD);
            break;
        } else if (new_outliers == 0) {
            printf(" -> CONVERGED (no new outliers)\n");
            break;
        } else {
            printf("\n");
        }
        
        // Update for next iteration
        last_median = current_median;
        last_std = current_std;
        iter++;
    }
    
    // Final summary
    if (iter >= MAX_ITERATIONS) {
        printf("    [DEBUG] -> STOPPED (max iterations %d reached)\n", MAX_ITERATIONS);
    } else if (valid_count < 3) {
        printf("    [DEBUG] -> STOPPED (too few valid samples: %d < 3)\n", valid_count);
    }
    
    printf("    [DEBUG] Final result: removed %d outliers in %d iterations, final_valid=%d/%d (%.1f%%)\n",
           total_outliers, iter, valid_count, nsamp, (float)valid_count/nsamp*100);
    
    return total_outliers;
}

/**
 * @brief Perform channel-level outlier detection using extracted iterative functions
 * @param data Input data array (nsamp * nchan)
 * @param nsamp Number of samples per channel
 * @param nchan Number of channels
 * @param Nsigma Sigma threshold for outlier detection
 * @param horizontalMask Output horizontal mask array
 * @param flaggedChans Array indicating which channels are fully flagged
 * @return Total number of outliers detected across all channels
 */
int inChanDetection(float *data, int nsamp, int nchan, float Nsigma,
                               int *horizontalMask, int *flaggedChans)
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
            // Perform iterative outlier detection for this channel
            int channelOutliers = inChanOutlierIter(
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
 * @brief Apply killThresh analysis to flag heavily contaminated channels
 * @param pointMask Input point-wise mask (pixels flagged in-channel)
 * @param channelFlagged Output 1D array indicating which channels are fully flagged
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param killThresh Threshold for flagging entire channels
 * @param flaggedAfterSIR Initial flagged pixel count (after binarySIR)
 * @param killedChannels Output: number of channels killed
 * @param localRFISkipped Output: number of channels skipped due to localized RFI
 * @param totalFlaggedAfter Output: total flagged pixels after killThresh
 */
void applyKillThresh(const int *pointMask, int *channelFlagged, int nsamp, int nchan, float killThresh, 
                    int flaggedAfterSIR, int *killedChannels, int *localRFISkippedPtr, 
                    int *totalFlaggedAfter)
{
    *killedChannels = 0;
    *localRFISkippedPtr = 0;
    *totalFlaggedAfter = 0;
    float rangeThreshold = 0.5f;  // If flagged pixels span <50% of channel, don't kill entire channel
    
    printf("\n=== killThresh Analysis (threshold=%.3f) ===\n", killThresh);
    
    int i, j;
    int localKilledChannels = 0;
    int localRFISkipped = 0;
    
    #pragma omp parallel for reduction(+:localKilledChannels,localRFISkipped)
    for (i = 0; i < nchan; i++) {
        int maskedCount = 0;
        int firstFlagged = -1, lastFlagged = -1;
        
        // First pass: count flagged pixels and find range
        for (j = 0; j < nsamp; j++) {
            int idx = j + i * nsamp;
            if (pointMask[idx]) {
                maskedCount++;
                if (firstFlagged == -1) firstFlagged = j;
                lastFlagged = j;
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
            } else {
                shouldKillChannel = 1;
            }
        }
        
        if (shouldKillChannel) {
            localKilledChannels++;
            channelFlagged[i] = 1;  // Mark channel as flagged
        }
    }
    
    // Assign local variables to output parameters
    *killedChannels = localKilledChannels;
    *localRFISkippedPtr = localRFISkipped;
    
    // Count total flagged pixels after killThresh
    int localTotalFlaggedAfter = 0;
    int idx;
    for (idx = 0; idx < nsamp * nchan; idx++) {
        // Union of pointMask and flagged channels
        int chan = idx / nsamp;
        if (pointMask[idx] || channelFlagged[chan]) localTotalFlaggedAfter++;
    }
    
    *totalFlaggedAfter = localTotalFlaggedAfter;
    
    // Print killThresh summary
    printf("killThresh Results:\n");
    printf("  - Killed channels: %d/%d (%.2f%%)\n", *killedChannels, nchan, 
           (float)(*killedChannels)/nchan*100);
    printf("  - Localized RFI skipped: %d/%d (%.2f%%)\n", *localRFISkippedPtr, nchan, 
           (float)(*localRFISkippedPtr)/nchan*100);
    printf("  - Flagged pixels before: %d/%d (%.2f%%)\n", flaggedAfterSIR, nsamp*nchan, 
           (float)flaggedAfterSIR/(nsamp*nchan)*100);
    printf("  - Flagged pixels after: %d/%d (%.2f%%)\n", *totalFlaggedAfter, nsamp*nchan, 
           (float)(*totalFlaggedAfter)/(nsamp*nchan)*100);
    printf("  - Additional pixels flagged: %d\n", *totalFlaggedAfter - flaggedAfterSIR);
    printf("=== End killThresh Analysis ===\n\n");
}

/**
 * @brief Apply pixel substitution for flagged pixels in channels not killed by killThresh
 * @param data Data array (will be modified for pixel substitution)
 * @param globalMask RFI mask array (input only, not modified)
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param pixelsSubstituted Output: number of pixels substituted
 */
void applySubstitution(float *data, int *globalMask, int nsamp, int nchan, int *pixelsSubstituted)
{
    *pixelsSubstituted = 0;
    printf("\n=== Pixel Substitution ===\n");
    
    int i;
    int localPixelsSubstituted = 0;
    
    #pragma omp parallel for reduction(+:localPixelsSubstituted)
    for (i = 0; i < nchan; i++) {
        // Count masked pixels in this channel
        int channelMaskedCount = 0;
        for (int samp = 0; samp < nsamp; samp++) {
            int idx = samp + i * nsamp;
            if (globalMask[idx] == 1) {
                channelMaskedCount++;
            }
        }
        
        // Skip channels with no flagged pixels
        if (channelMaskedCount == 0) continue;
        
        // Skip channels that are fully flagged (killed by killThresh)
        if (channelMaskedCount == nsamp) continue;
        
        // Substitute pixels in this channel
        float *channelData = data + i * nsamp;
        int *channelMask = globalMask + i * nsamp;
        
        // Allocate temporary arrays for this channel
        int *goodSamples = (int *)malloc(nsamp * sizeof(int));
        int *randomIndices = (int *)malloc(nsamp * sizeof(int));
        
        if (goodSamples && randomIndices) {
            substPixels(channelData, nsamp, channelMask, goodSamples, randomIndices);
            localPixelsSubstituted += channelMaskedCount;
        } else {
            printf("Warning: Memory allocation failed for channel %d substitution\n", i);
        }
        
        free(goodSamples);
        free(randomIndices);
    }
    
    *pixelsSubstituted = localPixelsSubstituted;
    
    printf("Substitution Summary:\n");
    printf("  - Pixels substituted: %d\n", *pixelsSubstituted);
    printf("=== End Pixel Substitution ===\n\n");
}

// Helper: Expand 1D channel mask to 2D mask (set entire channel to 1 if flagged)
static inline void expandChannelMask(const int *channelFlagged, int *mask2D, int nsamp, int nchan)
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

// Helper: OR one mask into the global mask (element-wise logical OR)
static inline void logicalOR(int *globalMask, const int *mask, int nsamp, int nchan)
{
    if (!globalMask || !mask) return;
    int total = nsamp * nchan;
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        if (mask[idx]) globalMask[idx] = 1;
    }
}

void identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan,
    int iterationIndex, int plot,
    IdentNSigmaMasks *masks,
    float *finalMedian, float *finalStd, int cudaReady, int *flaggedChans)
{    
    int *horizontalMask = masks->horizontalMask;
    int *verticalMask = masks->verticalMask;
    int *globalMask = masks->globalMask;
    int *pointMask = masks->pointMask;
    int *brightMask = masks->chanBrightMask;
    int *darkMask = masks->chanDarkMask;

    memset(horizontalMask, 0, nsamp * nchan * sizeof(int));
    memset(verticalMask, 0, nsamp * nchan * sizeof(int));
    memset(globalMask, 0, nsamp * nchan * sizeof(int));
    memset(pointMask, 0, nsamp * nchan * sizeof(int));
    memset(brightMask, 0, nsamp * nchan * sizeof(int));
    memset(darkMask, 0, nsamp * nchan * sizeof(int));

    int *good_samples = (int *)malloc(nsamp * sizeof(int));
    int *random_indices = (int *)malloc(nsamp * sizeof(int));
    float *median_temp = (float *)malloc(nsamp * nchan * sizeof(float));
    memcpy(median_temp, data, nsamp * nchan * sizeof(float));
    
    float killThresh = 0.2f;
    int i, j;
    

    if (plot)
    {
        printf("=== Generating Channel STD sigma_j Histogram (Iteration %d) ===\n", iterationIndex);
        drawChanStatHist(data, nsamp, nchan, 1, 0);
        printf("=== STD sigma_j Histogram Complete ===\n");
    }
    
    // === 1. inChannel Detection ===
    printf("=== Performing point-level (pixel) outlier detection ===\n");
    int pixelOutliers = 0;
    pixelOutliers = inChanDetection(data, nsamp, nchan, NSigmaInChan, 
                                                pointMask, flaggedChans);
    printf("Point-level detection: flagged %d outlier pixels\n", pixelOutliers);
    // Accumulate point-wise mask into global
    logicalOR(globalMask, pointMask, nsamp, nchan);
    
    // === 2. inChannel Pixel Substitution ===
    int inChanPixelsSubstituted;
    applySubstitution(data, pointMask, nsamp, nchan, &inChanPixelsSubstituted);
    printf("inChannel substitution: replaced %d pixels\n", inChanPixelsSubstituted);
    
    // === 3. killThresh Analysis ===
    // Count flagged pixels before killThresh
    int flaggedBeforeKillThresh = 0;
    for (i = 0; i < nsamp * nchan; i++) {
        if (pointMask[i] == 1) flaggedBeforeKillThresh++;
    }
    
    // === 4. In-Chan Subst ===
    printf("=== Applying random replacement to flagged pixels ===\n");
    randomReplaceRFIPixels(data, pointMask, nsamp, nchan);
    printf("Random replacement completed\n");

    int killedChannels, localRFISkipped, totalFlaggedAfter;
    applyKillThresh(pointMask, flaggedChans, nsamp, nchan, killThresh, flaggedBeforeKillThresh,
                   &killedChannels, &localRFISkipped, &totalFlaggedAfter);
    
    // === 4. outChannel Detection ===
    printf("=== Performing channel-level outlier detection ===\n");
    float *channel_stds = (float *)malloc(nchan * sizeof(float));
    float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
    outChanDetection(data, nsamp, nchan, flaggedChans, channel_stds, channel_stds_temp, NSigmaOutChan, NSigmaInChan);
    
    // Expand 1D flaggedChans to 2D horizontalMask
    expandChannelMask(flaggedChans, horizontalMask, nsamp, nchan);
    free(channel_stds);
    free(channel_stds_temp);
    // Accumulate channel-wise mask into global
    logicalOR(globalMask, horizontalMask, nsamp, nchan);
    
    // === 5. Out-Chan Subst ===
    int outChanPixelsSubstituted;
    applySubstitution(data, horizontalMask, nsamp, nchan, &outChanPixelsSubstituted);
    printf("outChannel substitution: replaced %d pixels\n", outChanPixelsSubstituted);
    
    int nFlaggedChans = 0;
    for (i = 0; i < nchan; i++) {
        if (flaggedChans[i]) nFlaggedChans++;
    }
    
    float *channelMeans = NULL;
    float overallChannelMean = 0.0f;
    if ((brightMask || darkMask) && nchan > 0) {
        channelMeans = (float *)malloc(nchan * sizeof(float));
        for (i = 0; i < nchan; i++) {
            double sum = 0.0;
            int base = i * nsamp;
            for (j = 0; j < nsamp; j++) {
                sum += median_temp[base + j];
            }
            float mean = (nsamp > 0) ? (float)(sum / nsamp) : 0.0f;
            channelMeans[i] = mean;
            overallChannelMean += mean;
        }
        overallChannelMean /= (float)nchan;

        for (i = 0; i < nchan; i++) {
            if (!flaggedChans[i]) {
                continue;
            }

            int *targetMask = NULL;
            if (channelMeans[i] >= overallChannelMean) {
                targetMask = brightMask;
            } else {
                targetMask = darkMask;
            }

            if (targetMask) {
                int base = i * nsamp;
                for (j = 0; j < nsamp; j++) {
                    targetMask[base + j] = 1;
                }
            }
        }
    }

    printf("Final status: %d/%d channels fully flagged (%.2f%%)\n", 
           nFlaggedChans, nchan, (float)nFlaggedChans/nchan*100);
    printf("\n=== RFI Detection Statistics ===\n");
    // Accumulate other masks into global
    logicalOR(globalMask, verticalMask, nsamp, nchan);
    logicalOR(globalMask, brightMask, nsamp, nchan);
    logicalOR(globalMask, darkMask, nsamp, nchan);

    // Recompute totals for reporting
    int globalFlagged = 0;
    for (i = 0; i < nsamp * nchan; i++) if (globalMask[i]) globalFlagged++;
    printf("Combined mask (all sources) flagged: %d/%d pixels (%.4f%%)\n", 
        globalFlagged, nsamp*nchan, (float)globalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");
    
    // globalMask already equals union of all masks via logicalOR
    float finalMedian_temp, finalStd_temp;
    findMeanStd(median_temp, nsamp * nchan, &finalMedian_temp, &finalStd_temp);
    float finalMedian_value = median(median_temp, nsamp * nchan);

    // Calculate total pixels substituted
    int totalPixelsSubstituted = inChanPixelsSubstituted + outChanPixelsSubstituted;
    printf("\n=== Final Processing Summary ===\n");
    printf("  - inChannel pixels substituted: %d\n", inChanPixelsSubstituted);
    printf("  - outChannel pixels substituted: %d\n", outChanPixelsSubstituted);
    printf("  - Total pixels substituted: %d\n", totalPixelsSubstituted);
    printf("  - Channels killed by killThresh: %d\n", killedChannels);

    float outlierRatio = (float)pixelOutliers / (nsamp * nchan);
    if (outlierRatio > killThresh)
    {
        printf("WARNING: High outlier ratio %.4f > %.2f detected - data may be corrupted\n",
               outlierRatio, killThresh);
    }

    *finalMedian = finalMedian_value;
    *finalStd = finalStd_temp;

    free(channelMeans);

    // Clean up allocated memory
    free(good_samples);
    free(random_indices);
    free(median_temp);

    printf("### DEBUG: identSubstNSigma exiting with finalMedian=%.6f, finalStd=%.6f ###\n", *finalMedian, *finalStd);
    fflush(stdout);  // Ensure immediate output
}

/**
 * @brief Draw comparison histograms showing before and after channel statistics
 * @param initial_stats Initial channel statistics before outChannel detection
 * @param final_stats Final channel statistics after outChannel detection
 * @param nchan Number of channels
 * @param use_mad Whether to use MAD (1) or STD (0) statistics
 * @param initial_flagged_count Number of initially flagged channels
 * @param final_flagged_count Number of finally flagged channels
 * @param iterations Number of iterations performed
 * @param nsigma_out NSigma threshold value used for outChannel detection (T_chan)
 * @param nsigma_in NSigma threshold value used for inChannel detection (T_point)
 */
void drawOutChannelComparisonHist(float *initial_stats, float *final_stats, int nchan, 
                                  int use_mad, int initial_flagged_count, int final_flagged_count,
                                  int iterations, float nsigma_out, float nsigma_in)
{
    const char* stat_name = use_mad ? "MAD" : "STD";
    const char* stat_label = use_mad ? "Channel MAD M\\dj\\u" : "Channel STD \\gs\\dj\\u";
    
    printf("\n=== OutChannel Comparison: %s Histogram Analysis ===\n", stat_name);
    printf("Initial valid channels: %d/%d (%.1f%%)\n", 
           nchan - initial_flagged_count, nchan, (float)(nchan - initial_flagged_count)/nchan*100);
    printf("Final valid channels: %d/%d (%.1f%%)\n", 
           nchan - final_flagged_count, nchan, (float)(nchan - final_flagged_count)/nchan*100);
    printf("Flagged by outChannel: %d channels\n", final_flagged_count - initial_flagged_count);
    
    // Calculate statistics for both datasets
    float *temp_data = (float *)malloc(nchan * sizeof(float));
    
    // Initial statistics
    memcpy(temp_data, initial_stats, nchan * sizeof(float));
    float initial_median = median(temp_data, nchan);
    float initial_dispersion = use_mad ? mad(initial_stats, nchan) : stdFromMedian(initial_stats, nchan);
    
    // Final statistics (only for non-flagged channels)
    int valid_count = 0;
    for (int i = 0; i < nchan; i++) {
        if (final_stats[i] >= 0) { // Assuming negative values indicate flagged channels
            temp_data[valid_count] = final_stats[i];
            valid_count++;
        }
    }
    
    float final_median = 0.0f, final_dispersion = 0.0f;
    if (valid_count > 0) {
        final_median = median(temp_data, valid_count);
        final_dispersion = use_mad ? mad(temp_data, valid_count) : stdFromMedian(temp_data, valid_count);
    }
    
    printf("Initial - Median: %.6f, %s: %.6f\n", initial_median, stat_name, initial_dispersion);
    printf("Final   - Median: %.6f, %s: %.6f (from %d valid channels)\n", 
           final_median, stat_name, final_dispersion, valid_count);
    
    // Find overall min and max for consistent scaling
    float overall_min = initial_stats[0], overall_max = initial_stats[0];
    for (int i = 0; i < nchan; i++) {
        if (initial_stats[i] < overall_min) overall_min = initial_stats[i];
        if (initial_stats[i] > overall_max) overall_max = initial_stats[i];
        if (final_stats[i] >= 0) { // Only consider non-flagged
            if (final_stats[i] < overall_min) overall_min = final_stats[i];
            if (final_stats[i] > overall_max) overall_max = final_stats[i];
        }
    }
    
    // Create histograms
    int nbins = 100;  // Match drawChanStatHist bin count
    float bin_width = (overall_max - overall_min) / nbins;
    float *initial_hist = (float *)calloc(nbins, sizeof(float));
    float *final_hist = (float *)calloc(nbins, sizeof(float));
    
    // Fill initial histogram
    for (int i = 0; i < nchan; i++) {
        int bin = (int)((initial_stats[i] - overall_min) / bin_width);
        if (bin < 0) bin = 0;
        if (bin >= nbins) bin = nbins - 1;
        initial_hist[bin]++;
    }
    
    // Fill final histogram (only non-flagged channels)
    for (int i = 0; i < nchan; i++) {
        if (final_stats[i] >= 0) { // Only non-flagged
            int bin = (int)((final_stats[i] - overall_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;
            final_hist[bin]++;
        }
    }
    
    // Find maximum count for scaling
    float max_count = 0;
    for (int i = 0; i < nbins; i++) {
        if (initial_hist[i] > max_count) max_count = initial_hist[i];
        if (final_hist[i] > max_count) max_count = final_hist[i];
    }
    
    // Create comparison plot
    printf("Creating outChannel comparison %s histogram...\n", stat_name);
    cpgpage();
    cpgvstd();
    cpgsch(1.2);
    cpgswin(overall_min, overall_max, 0, max_count * 1.1f);
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    
    char title[200];
    sprintf(title, "OutChannel Detection: %s Distribution Comparison", stat_name);
    
    // Use cpgmtxt for better positioning control - closer to plot
    cpgmtxt("B", 2.0, 0.5, 0.5, stat_label);           // X-axis label closer
    cpgmtxt("L", 2.0, 0.5, 0.5, "Number of Channels");  // Y-axis label closer  
    cpgmtxt("T", 1.0, 0.5, 0.5, title);                 // Title much closer to plot
    
    // Draw initial histogram (semi-transparent outline)
    cpgsci(3); // Green for initial
    cpgsls(2); // Dashed line
    for (int i = 0; i < nbins; i++) {
        if (initial_hist[i] > 0) {
            float x1 = overall_min + i * bin_width;
            float x2 = overall_min + (i + 1) * bin_width;
            float y = initial_hist[i];
            // Draw outline
            cpgmove(x1, 0);
            cpgdraw(x1, y);
            cpgdraw(x2, y);
            cpgdraw(x2, 0);
        }
    }
    
    // Draw final histogram (solid bars)
    cpgsci(2); // Red for final
    cpgsls(1); // Solid line
    for (int i = 0; i < nbins; i++) {
        if (final_hist[i] > 0) {
            float x1 = overall_min + i * bin_width;
            float x2 = overall_min + (i + 1) * bin_width;
            cpgrect(x1, x2, 0, final_hist[i]);
        }
    }
    
    // Draw threshold lines
    cpgsci(1); // White
    cpgmove(initial_median, 0);
    cpgdraw(initial_median, max_count * 1.1f);
    cpgptxt(initial_median, max_count * 0.9f, 0.0, 0.0, "Initial Median");
    
    if (valid_count > 0) {
        cpgmove(final_median, 0);
        cpgdraw(final_median, max_count * 1.1f);
        cpgptxt(final_median, max_count * 0.8f, 0.0, 0.0, "Final Median");
    }
    
    // Add legend
    float legend_x = overall_min + (overall_max - overall_min) * 0.65f;
    float legend_y_base = max_count * 0.75f;
    float line_spacing = max_count * 0.05f;
    
    cpgsci(3);
    cpgptxt(legend_x, legend_y_base, 0.0, 0.0, "Initial (before outChannel)");
    cpgsci(2);
    cpgptxt(legend_x, legend_y_base - line_spacing, 0.0, 0.0, "Final (after outChannel)");
    
    // Add statistics text
    cpgsci(1);
    char stats_text[200];
    sprintf(stats_text, "Flagged: %d/%d channels (%.1f%%)", 
            final_flagged_count - initial_flagged_count, nchan,
            (float)(final_flagged_count - initial_flagged_count)/nchan*100);
    cpgptxt(legend_x, legend_y_base - 3*line_spacing, 0.0, 0.0, stats_text);
    
    cpgsci(1); // Restore white color
    printf("OutChannel comparison %s histogram completed!\n", stat_name);

    // =====================================================================
    // Second part: Draw zoomed comparison histogram for 0-0.25 range
    // =====================================================================
    printf("Creating zoomed outChannel comparison %s histogram for range 0-0.25...\n", stat_name);
    
    // Check if there is data in the 0-0.25 range
    int has_data_in_range = 0;
    for (int i = 0; i < nchan; i++) {
        if ((initial_stats[i] >= 0.0f && initial_stats[i] <= 0.25f) ||
            (final_stats[i] >= 0.0f && final_stats[i] <= 0.25f && final_stats[i] >= 0)) {
            has_data_in_range = 1;
            break;
        }
    }
    
    if (has_data_in_range) {
        // Create new page for zoomed comparison histogram
        cpgpage();
        cpgvstd();
        cpgsch(1.2);
        
        // Create subdivided histograms for 0-0.25 range
        int zoom_nbins = 50;  // Match drawChanStatHist zoom bin count
        float zoom_min = 0.0f;
        float zoom_max = 0.25f;
        float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
        float *zoom_initial_hist = (float *)calloc(zoom_nbins, sizeof(float));
        float *zoom_final_hist = (float *)calloc(zoom_nbins, sizeof(float));
        
        // Fill zoomed initial histogram
        int zoom_initial_count = 0;
        for (int i = 0; i < nchan; i++) {
            if (initial_stats[i] >= zoom_min && initial_stats[i] <= zoom_max) {
                int bin = (int)((initial_stats[i] - zoom_min) / zoom_bin_width);
                if (bin < 0) bin = 0;
                if (bin >= zoom_nbins) bin = zoom_nbins - 1;
                zoom_initial_hist[bin]++;
                zoom_initial_count++;
            }
        }
        
        // Fill zoomed final histogram (only non-flagged channels)
        int zoom_final_count = 0;
        for (int i = 0; i < nchan; i++) {
            if (final_stats[i] >= 0 && final_stats[i] >= zoom_min && final_stats[i] <= zoom_max) {
                int bin = (int)((final_stats[i] - zoom_min) / zoom_bin_width);
                if (bin < 0) bin = 0;
                if (bin >= zoom_nbins) bin = zoom_nbins - 1;
                zoom_final_hist[bin]++;
                zoom_final_count++;
            }
        }
        
        // Calculate maximum count in zoomed range
        float zoom_max_count = 0;
        for (int i = 0; i < zoom_nbins; i++) {
            if (zoom_initial_hist[i] > zoom_max_count) zoom_max_count = zoom_initial_hist[i];
            if (zoom_final_hist[i] > zoom_max_count) zoom_max_count = zoom_final_hist[i];
        }
        
        if (zoom_max_count > 0) {
            // Set up coordinate system
            cpgswin(zoom_min, zoom_max, 0, zoom_max_count * 1.1f);
            cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
            char zoom_title[200];
            sprintf(zoom_title, "OutChannel Detection: %s Distribution Comparison (Zoomed: 0-0.25)", stat_name);
            
            // Use cpgmtxt for better positioning control - zoomed histogram, closer to plot
            cpgmtxt("B", 2.0, 0.5, 0.5, stat_label);           // X-axis label closer
            cpgmtxt("L", 2.0, 0.5, 0.5, "Number of Channels");  // Y-axis label closer
            cpgmtxt("T", 1.0, 0.5, 0.5, zoom_title);            // Title much closer to plot
            
            // Draw zoomed initial histogram (dashed outline)
            cpgsci(3); // Green for initial
            cpgsls(2); // Dashed line
            for (int i = 0; i < zoom_nbins; i++) {
                if (zoom_initial_hist[i] > 0) {
                    float x1 = zoom_min + i * zoom_bin_width;
                    float x2 = zoom_min + (i + 1) * zoom_bin_width;
                    float y = zoom_initial_hist[i];
                    // Draw outline
                    cpgmove(x1, 0);
                    cpgdraw(x1, y);
                    cpgdraw(x2, y);
                    cpgdraw(x2, 0);
                }
            }
            
            // Draw zoomed final histogram (solid bars)
            cpgsci(2); // Red for final
            cpgsls(1); // Solid line
            for (int i = 0; i < zoom_nbins; i++) {
                if (zoom_final_hist[i] > 0) {
                    float x1 = zoom_min + i * zoom_bin_width;
                    float x2 = zoom_min + (i + 1) * zoom_bin_width;
                    cpgrect(x1, x2, 0, zoom_final_hist[i]);
                }
            }
            
            // Draw threshold lines in zoomed range
            cpgsci(1); // White
            if (initial_median >= zoom_min && initial_median <= zoom_max) {
                cpgmove(initial_median, 0);
                cpgdraw(initial_median, zoom_max_count * 1.1f);
                cpgptxt(initial_median, zoom_max_count * 0.9f, 0.0, 0.0, "Initial Median");
            }
            
            if (valid_count > 0 && final_median >= zoom_min && final_median <= zoom_max) {
                cpgmove(final_median, 0);
                cpgdraw(final_median, zoom_max_count * 1.1f);
                cpgptxt(final_median, zoom_max_count * 0.8f, 0.0, 0.0, "Final Median");
            }
            
            // Add zoomed legend and statistics
            float zoom_legend_x = zoom_min + (zoom_max - zoom_min) * 0.05f;
            float zoom_legend_y_base = zoom_max_count * 0.9f;
            float zoom_line_spacing = zoom_max_count * 0.05f;
            
            cpgsci(3);
            cpgptxt(zoom_legend_x, zoom_legend_y_base, 0.0, 0.0, "Initial (before outChannel)");
            cpgsci(2);
            cpgptxt(zoom_legend_x, zoom_legend_y_base - zoom_line_spacing, 0.0, 0.0, "Final (after outChannel)");
            
            // Add iteration statistics
            cpgsci(1);
            char iteration_info1[200], iteration_info2[200], iteration_info3[200], iteration_info4[200], iteration_info5[200];
            sprintf(iteration_info1, "Iterations: %d", iterations);
            sprintf(iteration_info2, "T\\dpoint\\u: %.1f", nsigma_in);   // T_point (inChannel threshold)
            sprintf(iteration_info3, "T\\dchan\\u: %.1f", nsigma_out);  // T_chan (outChannel threshold)
            sprintf(iteration_info4, "Valid channels: %d", nchan - final_flagged_count);
            sprintf(iteration_info5, "Flagged channels: %d", final_flagged_count - initial_flagged_count);
            
            cpgptxt(zoom_legend_x, zoom_legend_y_base - 3*zoom_line_spacing, 0.0, 0.0, iteration_info1);
            cpgptxt(zoom_legend_x, zoom_legend_y_base - 4*zoom_line_spacing, 0.0, 0.0, iteration_info2);
            cpgptxt(zoom_legend_x, zoom_legend_y_base - 5*zoom_line_spacing, 0.0, 0.0, iteration_info3);
            cpgptxt(zoom_legend_x, zoom_legend_y_base - 6*zoom_line_spacing, 0.0, 0.0, iteration_info4);
            cpgptxt(zoom_legend_x, zoom_legend_y_base - 7*zoom_line_spacing, 0.0, 0.0, iteration_info5);
            
            // Add Gaussian fitting curve for final data
            printf("Fitting Gaussian curve to final data in zoomed histogram...\n");
            
            // Prepare fitting data for final histogram only
            float *x_data = (float *)malloc(zoom_nbins * sizeof(float));
            float *y_data = (float *)malloc(zoom_nbins * sizeof(float));
            int fit_points = 0;
            
            for (int i = 0; i < zoom_nbins; i++) {
                if (zoom_final_hist[i] > 0) {
                    float bin_center = zoom_min + (i + 0.5f) * zoom_bin_width;
                    x_data[fit_points] = bin_center;
                    y_data[fit_points] = zoom_final_hist[i];
                    fit_points++;
                }
            }
            
            if (fit_points >= 3) {
                // Calculate fixed amplitude from final histogram max bin value
                float zoom_final_amplitude = zoom_max_count;
                printf("Zoomed final histogram amplitude (max bin value): %.2f\n", zoom_final_amplitude);
                
                // Use GSL Gaussian fitting with fixed amplitude
                float fitted_mu, fitted_sigma;
                int gsl_success = gsl_gaussian_fit(x_data, y_data, fit_points, zoom_final_amplitude, &fitted_mu, &fitted_sigma);
                
                if (gsl_success) {
                    printf("Zoomed Gaussian fit (fixed amp=%.2f): center=%.6f, sigma=%.6f\n", 
                           zoom_final_amplitude, fitted_mu, fitted_sigma);
                    
                    // Draw fitted Gaussian curve
                    cpgsci(1); // Black color for fitted curve
                    cpgsls(1); // Solid line
                    
                    int curve_points = 100;
                    for (int i = 0; i < curve_points; i++) {
                        float x = zoom_min + i * (zoom_max - zoom_min) / (curve_points - 1);
                        float y = gaus_with_amplitude(x, zoom_final_amplitude, fitted_mu, fitted_sigma);
                        
                        if (i == 0) {
                            cpgmove(x, y);
                        } else {
                            cpgdraw(x, y);
                        }
                    }
                    
                    // Add Gaussian fit parameters as text annotations
                    cpgsci(1); // White color for text
                    char fit_text1[100], fit_text2[100], fit_text3[100];
                    sprintf(fit_text1, "Gaussian Fit (Final Data):");
                    sprintf(fit_text2, "\\gm = %.6f", fitted_mu);
                    sprintf(fit_text3, "\\gs = %.6f", fitted_sigma);
                    
                    float fit_text_x = zoom_min + (zoom_max - zoom_min) * 0.55f;
                    float fit_text_y_base = zoom_max_count * 0.6f;
                    float fit_line_spacing = zoom_max_count * 0.04f;
                    
                    cpgptxt(fit_text_x, fit_text_y_base, 0.0, 0.0, fit_text1);
                    cpgptxt(fit_text_x, fit_text_y_base - fit_line_spacing, 0.0, 0.0, fit_text2);
                    cpgptxt(fit_text_x, fit_text_y_base - 2 * fit_line_spacing, 0.0, 0.0, fit_text3);
                    
                    // Add fitted curve to legend
                    cpgsci(6);
                    cpgptxt(zoom_legend_x, zoom_legend_y_base - 2*zoom_line_spacing, 0.0, 0.0, "Gaussian fit (final data)");
                    
                } else {
                    printf("Error: GSL Gaussian fit failed for zoomed histogram\n");
                }
            } else {
                printf("Insufficient data points (%d < 3) for Gaussian fitting in zoomed range\n", fit_points);
            }
            
            free(x_data);
            free(y_data);
            
            cpgsls(1);
            cpgsci(1);
            printf("Zoomed outChannel comparison %s histogram completed! (Initial: %d, Final: %d channels in 0-0.25 range)\n", 
                   stat_name, zoom_initial_count, zoom_final_count);
        } else {
            printf("No data found in 0-0.25 range for outChannel comparison %s histogram\n", stat_name);
        }
        
        free(zoom_initial_hist);
        free(zoom_final_hist);
    } else {
        printf("No %s data in 0-0.25 range, skipping zoomed comparison histogram\n", stat_name);
    }
    
    // Cleanup
    free(temp_data);
    free(initial_hist);
    free(final_hist);
}