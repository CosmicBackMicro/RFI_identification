#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include <png.h>
#include <omp.h>
#include "cpgplot.h"

#include "ReadFASTData.h"
#include "findStats.h"
#include "identification.h"
#include "transpose.h"
#include "plot.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

float gaus(float x, float med, float sigma)
{
    return expf(-(x - med) * (x - med) / (2 * sigma * sigma)) / (sqrtf(2 * PI) * sigma);
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
    for (int i = 0; i < nchan; i++) {
        // Copy channel data for median calculation
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        channel_medians[i] = median(temp_data, nsamp);
        
        if (i < 5) {  // Debug output for first 5 channels
            printf("Channel %d median: %.6f\n", i, channel_medians[i]);
        }
    }
    
    // Subtract median from each channel
    #pragma omp parallel for
    for (int i = 0; i < nchan; i++) {
        for (int j = 0; j < nsamp; j++) {
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
    for (int i = 0; i < M_len; i++) {
        M[i] = powf(2.0f, (float)i);
        chi_i[i] = chi_1 / powf(p, log2f(M[i]));
    }

    memcpy(temp_data, data, length * sizeof(float));
    memset(local_mask, 0, length * sizeof(int));

    // Main thresholding logic
    for (int e = 0; e < eta_len; e++) {
        float current_eta = eta_i[e];
        for (int m = 0; m < M_len; m++) {
            int window = (int)M[m];
            float threshold = chi_i[m] / current_eta;

            // Window processing
            for (int i = 0; i <= length - window; i++) {
                float sum = 0.0f;
                int count = 0;

                // Calculate sum and count
                for (int j = 0; j < window; j++) {
                    if (!mask[i + j]) {
                        sum += fabsf(temp_data[i + j]);
                        count++;
                    }
                }

                // Apply threshold
                if (count > 0 && (sum / count) > threshold) {
                    for (int j = 0; j < window; j++) {
                        local_mask[i + j] = 1;
                    }
                }
            }
        }
    }

    // Merge masks
    for (int i = 0; i < length; i++) {
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
    for (int i = 0; i < nchan; i++) {
        findMeanStd(&temp_dataT[i * nsamp], nsamp, &means[i], &stds[i]);
    }

    // Global normalization
    float global_mean, global_std;
    findMeanStd(temp_dataT, nsamp * nchan, &global_mean, &global_std);
    float chi_1 = timesOfSigma * ksigma_2d(temp_dataT, mask_chanRFI, nsamp, nchan);

    // Normalize data
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < nchan; j++) {
        for (int i = 0; i < nsamp; i++) {
            temp_dataT[j * nsamp + i] = (temp_dataT[j * nsamp + i] - global_mean) / (global_std + 1e-6f);
        }
    }

    // Time-axis processing with optimized 1D
    #pragma omp parallel for
    for (int j = 0; j < nchan; j++) {
        sumthreshold_1d(&temp_dataT[j * nsamp], nsamp, &mask[j * nsamp], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Transpose for frequency processing
    float *transposed_data = (float *)malloc(nsamp * nchan * sizeof(float));
    transpose(temp_dataT, nchan, nsamp, transposed_data);

    // Frequency-axis processing
    #pragma omp parallel for
    for (int i = 0; i < nsamp; i++) {
        sumthreshold_1d(&transposed_data[i * nchan], nchan, &temp_maskT[i * nchan], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Merge masks
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nsamp; i++) {
        for (int j = 0; j < nchan; j++) {
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
    for (int idx = 0; idx < nsamp * nchan; idx++) {
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
    for (int idx = 0; idx < nsamp * nchan; idx++) {
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
    for (int i = 0; i < nchan; i++) {
        for (int idx = 0; idx < num_thresholds; idx++) {
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
    for (int idx = 0; idx < num_thresholds; idx++) {
        printf("%s: %d channels (%.2f%%)\n", 
               threshold_names[idx], threshold_counts[idx], 
               (float)threshold_counts[idx]/nchan*100);
    }
    
    free(threshold_counts);
}

/// @brief 绘制统一的阈值线系统
/// @param thresh_values 所有阈值的实际值数组
/// @param threshold_labels 阈值标签数组  
/// @param threshold_colors 阈值颜色数组
/// @brief Calculate and visualize channel Median Absolute Difference (MAD) statistics for threshold determination
/// MAD is calculated as the median of absolute differences between adjacent data points in each channel
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
            
            // 调试：分析第一个通道的偏差分布
            if (i == 0 && nsamp > 10) {
                printf("=== Debug: First channel MAD analysis ===\n");
                printf("Channel median: %.6f\n", channel_median[i]);
                printf("First 20 absolute deviations: ");
                for (int k = 0; k < 20 && k < nsamp; k++) {
                    printf("%.6f ", temp_data[k]);
                }
                printf("\n");
                
                // 统计唯一偏差值数量
                int unique_devs = 0;
                int zero_devs = 0;
                for (int k = 0; k < nsamp; k++) {
                    if (temp_data[k] == 0.0f) zero_devs++;
                    
                    int is_unique_dev = 1;
                    for (int l = 0; l < k; l++) {
                        if (fabsf(temp_data[k] - temp_data[l]) < 1e-9f) {
                            is_unique_dev = 0;
                            break;
                        }
                    }
                    if (is_unique_dev) unique_devs++;
                }
                printf("Channel 0: %d total samples, %d unique deviations, %d zero deviations\n", 
                       nsamp, unique_devs, zero_devs);
                printf("Mean absolute deviation: %.9f\n", sum_abs_dev / nsamp);
                printf("Channel 0 MAD: %.9f\n", channel_mad[i]);
            }
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
    printf("=== Channel MAD Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("MAD Mean: %.6f\n", mad_mean);
    printf("MAD Std:  %.6f\n", mad_std);
    printf("MAD Median: %.6f\n", mad_median);
    printf("MAD MAD: %.6f\n", mad_mad);
    printf("MAD Min: %.6f\n", mad_min);
    printf("MAD Max: %.6f\n", mad_max);
    
    // === 统一的11条阈值线系统 ===
    
    // === 统一的11条阈值线系统 ===
    const int NUM_ALL_THRESHOLDS = 11;
    float all_thresh_values[11];
    const char* all_threshold_labels[11] = {
        "-5*MAD", "-4*MAD", "-3*MAD", "-2*MAD", "-1*MAD", 
        "Median", 
        "+1*MAD", "+2*MAD", "+3*MAD", "+4*MAD", "+5*MAD"
    };
    const int all_threshold_colors[11] = {4, 7, 3, 8, 6, 1, 6, 8, 3, 7, 4}; // 蓝 黄 绿 橙 品红 白 品红 橙 绿 黄 蓝
    const float all_y_positions[11] = {0.15f, 0.25f, 0.35f, 0.45f, 0.75f, 0.65f, 0.85f, 0.90f, 1.05f, 1.00f, 0.95f};
    
    // 成对阈值控制开关：6个开关控制±1, ±2, ±3, ±4, ±5和中位线
    const int show_median = 1;              
    const int threshold_pair_enabled[5] = {1, 1, 1, 0, 1}; // ±1, ±2, ±3, ±4, ±5 对的开关
    
    // 根据成对开关生成11个位置的启用数组
    int threshold_enabled[11];
    for (int i = 0; i < 5; i++) {
        // 负阈值 (-5*MAD 到 -1*MAD，索引0-4)
        threshold_enabled[4-i] = threshold_pair_enabled[i]; // i=0→索引4(±1), i=1→索引3(±2), 依此类推
        // 正阈值 (+1*MAD 到 +5*MAD，索引6-10)  
        threshold_enabled[6+i] = threshold_pair_enabled[i]; // i=0→索引6(±1), i=1→索引7(±2), 依此类推
    }
    threshold_enabled[5] = show_median; // 中位线 (索引5)
    
    // 统一计算所有11条阈值线的值
    for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (i < 5) {
            // 负阈值：-5*MAD 到 -1*MAD (索引0-4)
            float multiplier = (float)(5 - i); // i=0→5, i=1→4, i=2→3, i=3→2, i=4→1
            all_thresh_values[i] = mad_median - multiplier * mad_mad;
        } else if (i == 5) {
            // 中位线 (索引5)
            all_thresh_values[i] = mad_median;
        } else {
            // 正阈值：+1*MAD 到 +5*MAD (索引6-10)
            float multiplier = (float)(i - 5); // i=6→1, i=7→2, i=8→3, i=9→4, i=10→5
            all_thresh_values[i] = mad_median + multiplier * mad_mad;
        }
    }
    
    // 输出开启的阈值线统计
    printf("\n=== Enabled Threshold Statistics ===\n");
    for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) {
            printf("%s threshold: %.6f\n", all_threshold_labels[i], all_thresh_values[i]);
        }
    }
    
    // 只对启用的阈值进行统计分析
    int enabled_count = 0;
    for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
        if (threshold_enabled[i]) enabled_count++;
    }
    
    if (enabled_count > 0) {
        float *enabled_values = malloc(enabled_count * sizeof(float));
        const char **enabled_names = malloc(enabled_count * sizeof(const char*));
        
        int idx = 0;
        for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
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
        // 📊 第一个图：完整MAD直方图 (Full MAD Histogram)
        //=============================================================================
        
        // 使用标准的直方图bin数量（MAD离散化问题已在算法层面解决）
        int nbins = (int)(sqrt(nchan) * 2);
        if (nbins < 30) nbins = 30;
        if (nbins > 100) nbins = 100;
        printf("Using %d bins for MAD histogram\n", nbins);
        
        float *hist = (float *)calloc(nbins, sizeof(float));
        float plot_min = mad_min;
        float plot_max = mad_max;
        
        // 为避免边界问题，略微扩展范围
        float range = plot_max - plot_min;
        if (range > 0) {
            plot_min -= 0.01f * range;
            plot_max += 0.01f * range;
        } else {
            // 如果所有值相同，创建小范围
            plot_min -= 0.001f;
            plot_max += 0.001f;
        }
        float bin_width = (plot_max - plot_min) / nbins;

        printf("MAD value range: [%.6f, %.6f], using %d bins\n", mad_min, mad_max, nbins);

        // 填充直方图
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

        // 统计非零bin数量
        int non_zero_bins = 0;
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) {
                non_zero_bins++;
                if (non_zero_bins <= 10) { // 只打印前10个非零bin
                    printf("Bin %d: %.0f channels (x-range: %.6f to %.6f)\n", 
                           i, hist[i], plot_min + i * bin_width, plot_min + (i + 1) * bin_width);
                }
            }
        }
        printf("Total non-zero bins: %d out of %d\n", non_zero_bins, nbins);

        // 计算最大计数
        float max_count = 0;
        for (i = 0; i < nbins; i++)
        {
            if (hist[i] > max_count) max_count = hist[i];
        }

        // --- 第一个图：绘制完整MAD直方图 ---
        printf("Creating MAD histogram plot...\n");
        cpgpage();                          // 创建新页面
        cpgvstd();                         // 设置标准视窗
        cpgsch(1.2);                       // 设置字符大小
        cpgswin(plot_min, plot_max, 0, max_count * 1.1f);  // 设置世界坐标
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);          // 绘制坐标轴
        cpglab("Channel MAD", "Number of Channels", "Channel MAD Distribution");  // 添加标签
        printf("MAD histogram axes set up complete\n");

        // --- 第一个图：绘制直方图条 ---
        cpgsci(2); // 红色
        for (i = 0; i < nbins; i++)
        {
            float x1 = plot_min + i * bin_width;
            float x2 = plot_min + (i + 1) * bin_width;
            cpgrect(x1, x2, 0, hist[i]);   // 绘制每个直方图条
        }

        // --- 第一个图：绘制统一的11条阈值线系统 ---
        for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
            if (threshold_enabled[i] && all_thresh_values[i] >= plot_min && all_thresh_values[i] <= plot_max) {
                cpgsci(all_threshold_colors[i]);                    // 设置阈值线颜色
                cpgmove(all_thresh_values[i], 0);                   // 移动到起点
                cpgdraw(all_thresh_values[i], max_count * 1.1f);    // 绘制垂直线
                cpgptxt(all_thresh_values[i], max_count * all_y_positions[i], 0.0, 0.0, all_threshold_labels[i]);  // 添加标签
            }
        }

        cpgsci(1); // 恢复白色

        // === 添加高斯拟合曲线 ===
        printf("Fitting Gaussian curve to MAD histogram...\n");
        
        // 准备拟合数据：将直方图转换为x-y数据点
        float *x_data = (float *)malloc(nbins * sizeof(float));
        float *y_data = (float *)malloc(nbins * sizeof(float));
        int fit_points = 0;
        
        // --- 第一个图：高斯曲线拟合 ---
        for (i = 0; i < nbins; i++) {
            if (hist[i] > 0) {  // 只使用非零的bin进行拟合
                x_data[fit_points] = plot_min + (i + 0.5f) * bin_width;  // bin center
                y_data[fit_points] = hist[i];
                fit_points++;
            }
        }
        
        // 使用现有的simple_curve_fit函数进行拟合
        float fitted_sigma = simple_curve_fit(x_data, y_data, fit_points, mad_median);
        printf("Gaussian fit: center=%.6f, sigma=%.6f\n", mad_median, fitted_sigma);
        
        // --- 第一个图：绘制拟合的高斯曲线 ---
        cpgsci(5); // 青色
        cpgsls(2); // 虚线
        
        int curve_points = 200;
        float curve_step = (plot_max - plot_min) / curve_points;
        
        for (i = 0; i < curve_points; i++) {
            float x = plot_min + i * curve_step;
            float y = max_count * gaus(x, mad_median, fitted_sigma);
            
            if (i == 0) {
                cpgmove(x, y);
            } else {
                cpgdraw(x, y);
            }
        }
        
        cpgptxt(mad_median + fitted_sigma, max_count * 0.4f, 0.0, 0.0, "Gaussian Fit");
        cpgsls(1);

        free(x_data);
        free(y_data);

        cpgsci(1); // 恢复白色
        // *** 第一个图绘制完成 ***

        //=============================================================================
        // 🔍 第二个图：放大MAD直方图 (Zoomed MAD Histogram - Range 0-0.25) 
        //=============================================================================
        
        // === 添加0到0.25区间的放大直方图 ===
        printf("Creating zoomed MAD histogram for range 0-0.25...\n");
        
        // 检查是否有数据在0-0.25范围内
        int has_data_in_range = 0;
        for (i = 0; i < nchan; i++) {
            if (channel_mad[i] >= 0.0f && channel_mad[i] <= 0.25f) {
                has_data_in_range = 1;
                break;
            }
        }
        
        if (has_data_in_range) {
            // --- 第二个图：创建新页面和设置 ---
            cpgpage();                          // 创建新页面用于放大直方图
            cpgvstd();                         // 设置标准视窗
            cpgsch(1.2);                       // 设置字符大小
            
            // --- 第二个图：为0-0.25区间创建细分直方图数据 ---
            int zoom_nbins = 50;               // 使用更多bin获得更高分辨率
            float zoom_min = 0.0f;
            float zoom_max = 0.25f;
            float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
            float *zoom_hist = (float *)calloc(zoom_nbins, sizeof(float));
            
            // --- 第二个图：填充放大直方图数据 ---
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
            
            // --- 第二个图：计算放大区间的最大计数 ---
            float zoom_max_count = 0;
            for (i = 0; i < zoom_nbins; i++) {
                if (zoom_hist[i] > zoom_max_count) zoom_max_count = zoom_hist[i];
            }
            
            if (zoom_max_count > 0) {
                // --- 第二个图：设置坐标系和标签 ---
                cpgswin(zoom_min, zoom_max, 0, zoom_max_count * 1.1f);     // 设置世界坐标
                cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);                  // 绘制坐标轴
                cpglab("Channel MAD", "Number of Channels", "Channel MAD Distribution (Zoomed: 0-0.25)");  // 添加标签
                
                // --- 第二个图：绘制放大的直方图条 ---
                cpgsci(2); // 红色
                for (i = 0; i < zoom_nbins; i++) {
                    if (zoom_hist[i] > 0) {
                        float x1 = zoom_min + i * zoom_bin_width;
                        float x2 = zoom_min + (i + 1) * zoom_bin_width;
                        cpgrect(x1, x2, 0, zoom_hist[i]);       // 绘制每个放大直方图条
                    }
                }
                
                // --- 第二个图：绘制统一的11条阈值线系统（放大版本） ---
                for (int i = 0; i < NUM_ALL_THRESHOLDS; i++) {
                    if (threshold_enabled[i] && all_thresh_values[i] >= zoom_min && all_thresh_values[i] <= zoom_max) {
                        cpgsci(all_threshold_colors[i]);                    // 设置阈值线颜色
                        cpgmove(all_thresh_values[i], 0);                   // 移动到起点
                        cpgdraw(all_thresh_values[i], zoom_max_count * 1.1f);  // 绘制垂直线
                        cpgptxt(all_thresh_values[i], zoom_max_count * all_y_positions[i], 0.0, 0.0, all_threshold_labels[i]);  // 添加标签
                    }
                }
                
                // --- 第二个图：高斯曲线拟合（针对放大直方图） ---
                if (zoom_count >= 10) {  // 需要足够的数据点进行拟合
                    // 将放大直方图转换为拟合用的数据点
                    int fit_points = 0;
                    float *fit_x = calloc(zoom_nbins, sizeof(float));
                    float *fit_y = calloc(zoom_nbins, sizeof(float));
                    
                    if (fit_x && fit_y) {
                        for (int i = 0; i < zoom_nbins; i++) {
                            if (zoom_hist[i] > 0) {
                                fit_x[fit_points] = zoom_min + (i + 0.5f) * zoom_bin_width;
                                fit_y[fit_points] = (float)zoom_hist[i];
                                fit_points++;
                            }
                        }
                        
                        if (fit_points >= 3) {  // 高斯拟合需要的最少点数
                            // Fit Gaussian curve using existing simple_curve_fit function
                            float fitted_sigma = simple_curve_fit(fit_x, fit_y, fit_points, mad_median);
                            
                            if (fitted_sigma > 1e-6f) {
                                printf("Zoomed histogram Gaussian fit: center=%.6f, sigma=%.6f\n", mad_median, fitted_sigma);
                                
                                // Draw fitted Gaussian curve
                                cpgsci(5); // Cyan color for fit
                                cpgsls(2); // Dashed line style
                                
                                float fit_curve_x[200], fit_curve_y[200];
                                int n_curve = 200;
                                for (int i = 0; i < n_curve; i++) {
                                    fit_curve_x[i] = zoom_min + i * (zoom_max - zoom_min) / (n_curve - 1);
                                    fit_curve_y[i] = zoom_max_count * gaus(fit_curve_x[i], mad_median, fitted_sigma);
                                }
                                cpgline(n_curve, fit_curve_x, fit_curve_y);
                                
                                cpgsls(1); // Back to solid line
                                cpgptxt(zoom_min + (zoom_max - zoom_min) * 0.7f, zoom_max_count * 0.85f, 
                                        0.0, 0.0, "Gaussian Fit");
                            }
                        }
                    }
                    
                    free(fit_x);
                    free(fit_y);
                }
                
                cpgsci(1); // 恢复白色
                printf("Zoomed MAD histogram completed! (%d channels in 0-0.25 range)\n", zoom_count);
                // *** 第二个图绘制完成 ***
            } else {
                printf("No data found in 0-0.25 range for MAD histogram\n");
            }
            
            free(zoom_hist);
        } else {
            printf("No MAD data in 0-0.25 range, skipping zoomed histogram\n");
        }

        //=============================================================================
        // 🏁 两个图的绘制全部完成
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
            
            // 调试：分析第一个通道的偏差分布
            if (i == 0 && nsamp > 10) {
                printf("=== Debug: First channel std analysis ===\n");
                printf("Channel median: %.6f\n", channel_median[i]);
                printf("First 20 squared deviations: ");
                for (int k = 0; k < 20 && k < nsamp; k++) {
                    printf("%.6f ", temp_data[k]);
                }
                printf("\n");
                printf("Mean squared deviation: %.9f\n", mean_squared_dev);
                printf("Channel 0 STD: %.9f\n", channel_std[i]);
            }
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
    
    printf("\n=== Channel STD Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("STD Median: %.6f\n", std_median);
    printf("STD STD: %.6f\n", std_std);
    printf("STD Min: %.6f\n", std_min);
    printf("STD Max: %.6f\n", std_max);
    
    // 统一的阈值计算系统（移除fallback逻辑）
    float multipliers[6] = {-1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float threshold_values[6];
    const char* threshold_names[6] = {"-1*STD", "1*STD", "2*STD", "3*STD", "4*STD", "5*STD"};
    
    // 直接计算所有阈值
    for (int i = 0; i < 6; i++) {
        threshold_values[i] = std_median + multipliers[i] * std_std;
    }
    
    // 统一打印阈值
    printf("\n=== Suggested Thresholds ===\n");
    for (int i = 0; i < 6; i++) {
        printf("%s threshold: %.6f\n", threshold_names[i], threshold_values[i]);
    }
    
    // Print threshold statistics using the new function
    printThresholdStatistics(channel_std, nchan, threshold_values, threshold_names, 6, "STD");
    
    // =====================================================================
    // 第一部分：绘制完整范围的STD直方图（包含所有统一的阈值线）
    // =====================================================================
    if (plot)
    {
        // 使用标准的直方图bin数量（STD离散化问题已在算法层面解决）
        int nbins = (int)(sqrt(nchan) * 2);
        if (nbins < 30) nbins = 30;
        if (nbins > 100) nbins = 100;
        printf("Using %d bins for STD histogram\n", nbins);
        
        float *hist = (float *)calloc(nbins, sizeof(float));
        float plot_min = std_min;
        float plot_max = std_max;
        
        // 为避免边界问题，略微扩展范围
        float range = plot_max - plot_min;
        if (range > 0) {
            plot_min -= 0.01f * range;
            plot_max += 0.01f * range;
        } else {
            // 如果所有值相同，创建小范围
            plot_min -= 0.001f;
            plot_max += 0.001f;
        }
        float bin_width = (plot_max - plot_min) / nbins;

        printf("STD value range: [%.6f, %.6f], using %d bins\n", std_min, std_max, nbins);

        // 填充直方图
        for (i = 0; i < nchan; i++)
        {
            int bin = (int)((channel_std[i] - plot_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;
            hist[bin]++;
        }

        // 计算最大计数
        float max_count = 0;
        for (i = 0; i < nbins; i++)
        {
            if (hist[i] > max_count) max_count = hist[i];
        }

        // 绘制直方图
        printf("Creating STD histogram plot...\n");
        cpgpage();
        cpgvstd();
        cpgsch(1.2);
        cpgswin(plot_min, plot_max, 0, max_count * 1.1f);
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpglab("Channel STD", "Number of Channels", "Channel STD Distribution");
        printf("STD histogram axes set up complete\n");

        // 绘制实心直方图条
        cpgsci(2); // 红色
        for (i = 0; i < nbins; i++)
        {
            float x1 = plot_min + i * bin_width;
            float x2 = plot_min + (i + 1) * bin_width;
            cpgrect(x1, x2, 0, hist[i]);
        }

        // === 统一的阈值配置系统（类似MAD函数） ===
        // 阈值对控制开关：[±1, ±2, ±3, ±4, ±5]
        int threshold_pair_enabled[5] = {1, 1, 1, 0, 1}; // ±1,±2,±3开启，±4关闭，±5开启
        int show_median = 1; // 显示中位线
        
        // 计算负值阈值（使用已有的正值阈值进行对称计算）
        float thresh_neg1std = threshold_values[0];  // 直接使用前面计算的结果
        float thresh_neg2std = std_median - (threshold_values[2] - std_median);  // 相对于中位值对称
        float thresh_neg3std = std_median - (threshold_values[3] - std_median);
        float thresh_neg4std = std_median - (threshold_values[4] - std_median);
        float thresh_neg5std = std_median - (threshold_values[5] - std_median);
        
        // 统一的阈值数组配置
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
        
        int all_threshold_colors[11] = {4, 7, 3, 8, 6, 1, 6, 8, 3, 7, 4}; // 对称配色
        float all_y_positions[11] = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 0.65f, 0.75f, 0.9f, 1.05f, 1.0f, 0.85f};

        // 绘制阈值线
        for (int thresh_idx = 0; thresh_idx < 11; thresh_idx++) {
            int should_draw = 0;
            
            if (thresh_idx == 5) { // 中位线
                should_draw = show_median;
            } else {
                // 阈值对：索引0,1,2,3,4对应负值，索引6,7,8,9,10对应正值
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

        cpgsci(1); // 恢复白色

        // =====================================================================
        // 第二部分：绘制0-0.25区间的放大STD直方图（细节视图）
        // =====================================================================
        printf("Creating zoomed STD histogram for range 0-0.25...\n");
        
        // 检查是否有数据在0-0.25范围内
        int has_data_in_range = 0;
        for (i = 0; i < nchan; i++) {
            if (channel_std[i] >= 0.0f && channel_std[i] <= 0.25f) {
                has_data_in_range = 1;
                break;
            }
        }
        
        if (has_data_in_range) {
            // 创建新页面用于放大直方图
            cpgpage();
            cpgvstd();
            cpgsch(1.2);
            
            // 为0-0.25区间创建细分直方图
            int zoom_nbins = 50; // 使用更多bin获得更高分辨率
            float zoom_min = 0.0f;
            float zoom_max = 0.25f;
            float zoom_bin_width = (zoom_max - zoom_min) / zoom_nbins;
            float *zoom_hist = (float *)calloc(zoom_nbins, sizeof(float));
            
            // 填充放大直方图
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
            
            // 计算放大区间的最大计数
            float zoom_max_count = 0;
            for (i = 0; i < zoom_nbins; i++) {
                if (zoom_hist[i] > zoom_max_count) zoom_max_count = zoom_hist[i];
            }
            
            if (zoom_max_count > 0) {
                // 设置坐标系
                cpgswin(zoom_min, zoom_max, 0, zoom_max_count * 1.1f);
                cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
                cpglab("Channel STD", "Number of Channels", "Channel STD Distribution (Zoomed: 0-0.25)");
                
                // 绘制放大的直方图条
                cpgsci(2); // 红色
                for (i = 0; i < zoom_nbins; i++) {
                    if (zoom_hist[i] > 0) {
                        float x1 = zoom_min + i * zoom_bin_width;
                        float x2 = zoom_min + (i + 1) * zoom_bin_width;
                        cpgrect(x1, x2, 0, zoom_hist[i]);
                    }
                }
                
                // === 使用统一系统绘制放大图中的阈值线 ===
                for (int thresh_idx = 0; thresh_idx < 11; thresh_idx++) {
                    int should_draw = 0;
                    
                    if (thresh_idx == 5) { // 中位线
                        should_draw = show_median;
                    } else {
                        // 阈值对：索引0,1,2,3,4对应负值，索引6,7,8,9,10对应正值
                        int pair_idx = (thresh_idx < 5) ? (4 - thresh_idx) : (thresh_idx - 6);
                        should_draw = threshold_pair_enabled[pair_idx];
                    }
                    
                    if (should_draw && all_thresh_values[thresh_idx] >= zoom_min && all_thresh_values[thresh_idx] <= zoom_max) {
                        cpgsci(all_threshold_colors[thresh_idx]);
                        cpgmove(all_thresh_values[thresh_idx], 0);
                        cpgdraw(all_thresh_values[thresh_idx], zoom_max_count * 1.1f);
                        // 调整y位置以适应放大图
                        float zoom_y_pos = 0.1f + (thresh_idx * 0.08f);
                        if (zoom_y_pos > 0.9f) zoom_y_pos = 0.9f;
                        cpgptxt(all_thresh_values[thresh_idx], zoom_max_count * zoom_y_pos, 
                               0.0, 0.0, all_threshold_labels[thresh_idx]);
                    }
                }
                
                cpgsci(1); // 恢复白色
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
    float *finalMedian, float *finalStd)
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
    
    float lastMean = 0.0f, lastStd = 0.0f, lastMedian = 0.0f;
    float mean = 0.0f, std = 0.0f, med = 0.0f;
    float meanDiff = 0.0f, stdDiff = 0.0f, medianDiff = 0.0f;
    float upperBound, lowerBound;
    float n_ref = nsamp;
    int totReplaceCnt = 0;
    // float killThresh = 0.04f; // Threshold for killing a channel if too many pixels are masked
    float killThresh = 0.2f;  // 10% pixel ratio threshold
    int i, j;

    // float chanMedian[nchan];
    // float chanMean[nchan];
    // float chanStd[nchan];
    // for (i = 0; i < nchan; i++) {
    //     chanMedian[i] = median(median_temp + i * nsamp, nsamp);
    //     findMeanStd(data + i * nsamp, nsamp, &chanMean[i], &chanStd[i]);
    // }
    // normalizeChannelData(data, nsamp, nchan, chanMedian, chanStd, median_temp);
    
    // memcpy(median_temp, data, nsamp * nchan * sizeof(float));
    // Process each frequency channel

    // 减中值 - Use the new function to subtract channel medians
    subtractChannelMedians(data, nsamp, nchan);
    
    // Visualize channel MAD statistics for threshold determination in first 20 iterations
    // Use iterationIndex as a proxy for iteration counter (passed from ReadFASTData.c)
    if (plot)
    {
        printf("=== Generating Channel MAD Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelMAD(data, nsamp, nchan, 1);
        printf("=== MAD Histogram Complete ===\n");
        
        printf("=== Generating Channel STD Histogram (Iteration %d) ===\n", iterationIndex);
        visualizeChannelStd(data, nsamp, nchan, 1);
        printf("=== STD Histogram Complete ===\n");
    }
    
    // === 1. 通道级标记 (Channel level flagging first) ===
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
    
    // === 2. 通道内像素异常值标记 (Channel-internal pixel flagging) ===
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
            lastMedian = med;
            findMeanStd(data + i * nsamp, nsamp, &mean, &std);
            med = median(median_temp + i * nsamp, nsamp);
            meanDiff = fabsf(mean - lastMean) / lastMean;
            stdDiff = fabsf(std - lastStd) / lastStd;
            medianDiff = fabsf(med - lastMedian) / lastMedian;

            float scale_row = sqrtf(n_ref / nsamp);
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
            
            // substitute_pixels_1d(data + i * nsamp, nsamp, horizontalMask + i * nsamp,
            //                      good_samples, random_indices);

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
            lastMedian = med;
            findMeanStd(transposedData + i * nchan, nchan, &mean, &std);
            med = median(median_temp + i * nchan, nchan);
            meanDiff = fabsf(mean - lastMean) / lastMean;
            stdDiff = fabsf(std - lastStd) / lastStd;
            medianDiff = fabsf(med - lastMedian) / lastMedian;

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
            
            // substitute_pixels_1d(transposedData + i * nchan, nchan, transposedMask + i * nchan,
            //                         good_samples_v, random_indices_v);

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
    for (int idx = 0; idx < nsamp * nchan; idx++) {
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
    for (int idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) globalFlagged++;
    }
    printf("Global mask flagged after logicalOR: %d/%d pixels (%.4f%%)\n", 
           globalFlagged, nsamp*nchan, (float)globalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");

    // Apply binarySIR before killThresh analysis to filter isolated pixels for better range calculation
    printf("\n=== Applying binarySIR filtering before killThresh analysis ===\n");
    int flaggedBeforeSIR = globalFlagged;
    binarySIR(globalMask, nsamp, nchan, 3, 3, 1.0f, 0.2f); // Filter out isolated pixels
    
    // Recount flagged pixels after binarySIR
    int flaggedAfterSIR = 0;
    for (int idx = 0; idx < nsamp * nchan; idx++) {
        if (globalMask[idx] == 1) flaggedAfterSIR++;
    }
    printf("binarySIR filtering: %d -> %d flagged pixels (removed %d isolated pixels)\n", 
           flaggedBeforeSIR, flaggedAfterSIR, flaggedBeforeSIR - flaggedAfterSIR);

    // === 3. 点干扰严重通道标记 (killThresh - flag heavily contaminated channels) ===
    // Apply killThresh: if a channel has more than killThresh fraction of flagged pixels, flag the entire channel
    // Add position range check: avoid killing channels with concentrated local RFI
    printf("\n=== killThresh Analysis (threshold=%.3f) ===\n", killThresh);
    int killedChannels = 0;
    int totalFlaggedBefore = flaggedAfterSIR; // Use count after binarySIR filtering
    int totalFlaggedAfter = 0;
    int localRFISkipped = 0;  // Count channels skipped due to local RFI pattern
    float rangeThreshold = 0.5f;  // If flagged pixels span <30% of channel, don't kill entire channel

    #pragma omp parallel for reduction(+:killedChannels,localRFISkipped)
    for (int chan = 0; chan < nchan; chan++) {
        int maskedCount = 0;
        int firstFlagged = -1, lastFlagged = -1;
        
        // First pass: count flagged pixels and find range
        for (int samp = 0; samp < nsamp; samp++) {
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
            for (int samp = 0; samp < nsamp; samp++) {
                int idx = samp + chan * nsamp;
                globalMask[idx] = 1;
            }
        }
    }
    
    // Count total flagged pixels after killThresh
    for (int idx = 0; idx < nsamp * nchan; idx++) {
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




