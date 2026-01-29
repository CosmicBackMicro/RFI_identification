#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include <png.h>
#include <omp.h>
#include <stdint.h>
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
#include "mask.h"
/* Algorithm headers (declare IQRM and CLFD) */
#include "include/alg_IQRM.h"
#include "include/alg_CLFD.h"
#include "include/alg_CLFD.h"
#include "include/alg_IQRM.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

/* -------------------------------------------------------------------------
 * Pulse-mask generation (Python parity with src/experiment_pulse_mask.py)
 *
 * Coordinate convention (matches the latest Python script):
 * - Time axis for mask generation is *window-local*, i.e. t=0 corresponds to
 *   the start of the currently displayed/processed block.
 * - `t0_local` is the arrival time (seconds) of the first pulse at the highest
 *   frequency within this window.
 * - Mask is written in channel-major layout: mask[ch*nsamp + t].
 *
 * Threading:
 * - This function does not allocate memory and touches only the output buffer
 *   provided by caller, so it is safe to call from an OpenMP parallel loop as
 *   long as each thread uses a distinct mask buffer slice.
 * ------------------------------------------------------------------------- */

static inline float dispersion_delay_sec(float f_mhz, float f_ref_mhz, float dm)
{
    /* dt = 4.148808e3 * DM * (f^-2 - f_ref^-2)  (seconds), same constant as Python */
    const float k_dm = 4.148808e3f;
    if (f_mhz < 1e-6f) f_mhz = 1e-6f;
    if (f_ref_mhz < 1e-6f) f_ref_mhz = 1e-6f;
    return k_dm * dm * (1.0f / (f_mhz * f_mhz) - 1.0f / (f_ref_mhz * f_ref_mhz));
}

void identPulse(
    int hasPulse,
    float DM,
    float P0,
    float width,
    float T0_local,
    float lo_freq,
    float hi_freq,
    const float *freqs_mhz,
    int nchan,
    float tbin_s,
    int nsamp,
    bool *pulseMask
)
{
    if (!hasPulse) return;
    if (!pulseMask || !freqs_mhz) return;
    if (nchan <= 0 || nsamp <= 0) return;
    if (tbin_s <= 0.0f) return;
    if (P0 <= 0.0f) return;
    if (width <= 0.0f) width = tbin_s; /* at least 1 sample after rounding */

    /* Reference frequency = max(freqs) (highest frequency arrives first) */
    float f_ref = freqs_mhz[0];
    for (int c = 1; c < nchan; ++c) {
        if (freqs_mhz[c] > f_ref) f_ref = freqs_mhz[c];
    }

    /* Compute per-channel delays and find max delay for k-range bounds */
    float max_delay = 0.0f;
    /* We avoid heap allocations; delay computed on-the-fly in channel loop. */
    for (int c = 0; c < nchan; ++c) {
        float d = dispersion_delay_sec(freqs_mhz[c], f_ref, DM);
        if (d > max_delay) max_delay = d;
    }

    int width_samp = (int)floorf(width / tbin_s);
    if (width_samp < 1) width_samp = 1;

    const float duration_sec = (float)nsamp * tbin_s;

    /* Match Python bounds:
       k_min = floor((-max_delay - t0)/P)
       k_max = ceil((duration - t0)/P)
       iterate k in [k_min, k_max+1]
     */
    const int k_min = (int)floorf(((-max_delay) - T0_local) / P0);
    const int k_max = (int)ceilf(((duration_sec) - T0_local) / P0);

    /* Debug output for first call */
    static int debug_printed = 0;
    if (!debug_printed) {
        printf("[identPulse info] DM=%.2f P0=%.4f Width=%.4f T0=%.4f Clamp=[%.1f, %.1f] MHz\n", 
               DM, P0, width, T0_local, lo_freq, hi_freq);
        printf("[identPulse info] f_ref=%.2f MHz, max_delay=%.4fs, tbin=%.2e s\n", f_ref, max_delay, tbin_s);
        printf("[identPulse info] nsamp=%d width_samp=%d duration=%.4fs\n", nsamp, width_samp, duration_sec);
        printf("[identPulse info] k_min=%d k_max=%d\n", k_min, k_max);
        debug_printed = 1;
    }

    for (int k = k_min; k <= (k_max + 1); ++k) {
        const float pulse_arrival_t0 = T0_local + (float)k * P0;

        for (int c = 0; c < nchan; ++c) {
            /* Clamp frequency range */
            if (freqs_mhz[c] < lo_freq || freqs_mhz[c] > hi_freq) continue;

            const float delay = dispersion_delay_sec(freqs_mhz[c], f_ref, DM);
            const float t_start = (pulse_arrival_t0 + delay) / tbin_s;
            const int center_samp = (int)floorf(t_start);

            int s0 = center_samp - (width_samp / 2);
            int s1 = s0 + width_samp;
            if (s0 < 0) s0 = 0;
            if (s1 > nsamp) s1 = nsamp;

            for (int s = s0; s < s1; ++s) {
                pulseMask[(size_t)c * (size_t)nsamp + (size_t)s] = true;
            }
        }
    }
}

int eraseIsolatedPixels(bool *restrict mask, int width, int height, int N)
{
    if (!mask || width <= 0 || height <= 0) {
        return 0;
    }

    // Normalize N: odd, at least 1, at most min(width,height)
    if (N < 1) N = 1;
    if ((N & 1) == 0) N += 1; // make odd
    const int W = width;
    const int H = height;
    const int min_wh = (W < H) ? W : H;
    if (N > min_wh) {
        N = (min_wh % 2 == 1) ? min_wh : (min_wh - 1);
        if (N < 1) N = 1;
    }

    // Build a padded integral image S of size (H+1) x (W+1), zero-initialized
    const int SW = W + 1;
    const int SH = H + 1;
    uint32_t *S = (uint32_t *)calloc((size_t)SW * (size_t)SH, sizeof(uint32_t));
    if (!S) return 0;

    // 1-based build: S[y+1,x+1] = bin + S[y,x+1] + S[y+1,x] - S[y,x]
    for (int y = 0; y < H; ++y) {
        const int row_off = (y + 1) * SW;
        const int prev_off = y * SW;
        const int base = y * W;
        for (int x = 0; x < W; ++x) {
            const uint32_t v = (mask[base + x] != 0) ? 1u : 0u;
            S[row_off + (x + 1)] = v + S[prev_off + (x + 1)] + S[row_off + x] - S[prev_off + x];
        }
    }

    const int half = N / 2;
    int suppressed = 0;
    // #pragma omp parallel for reduction(+:suppressed) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (!mask[idx]) continue;

            int x0 = x - half; if (x0 < 0) x0 = 0;
            int y0 = y - half; if (y0 < 0) y0 = 0;
            int x1 = x + half; if (x1 >= W) x1 = W - 1;
            int y1 = y + half; if (y1 >= H) y1 = H - 1;

            // Convert to 1-based indices for S
            const int X0 = x0, Y0 = y0;
            const int X1 = x1 + 1, Y1 = y1 + 1;
            const int sum = (int)( S[Y1 * SW + X1]
                                 - S[Y0 * SW + X1]
                                 - S[Y1 * SW + X0]
                                 + S[Y0 * SW + X0] );
            if (sum == 1) {
                mask[idx] = false;
                ++suppressed;
            }
        }
    }

    free(S);
    return suppressed;
}

/**
 * Simple 4-connected (diamond) dilation for integer masks.
 * Any non-zero pixel in the input is treated as 1. The dilation radius
 * controls how far each pixel expands using Manhattan distance
 * (radius=1 => up/down/left/right neighbors only).
 * The operation can be repeated for multiple iterations.
 *
 * Returns the count of newly activated pixels compared to the original mask.
 */
int dilateMask(bool *restrict mask, int width, int height, int radius, int iterations)
{
    if (!mask || width <= 0 || height <= 0) {
        return 0;
    }
    if (radius < 1) radius = 1;
    if (iterations < 1) iterations = 1;

    const size_t total = (size_t)width * (size_t)height;
    bool *src = (bool *)malloc(total * sizeof(bool));
    bool *dst = (bool *)malloc(total * sizeof(bool));
    if (!src || !dst) {
        free(src);
        free(dst);
        return 0;
    }

    // Copy input mask to src
    memcpy(src, mask, total * sizeof(bool));

    for (int iter = 0; iter < iterations; ++iter) {
        memset(dst, 0, total * sizeof(bool));

        for (int y = 0; y < height; ++y) {
            const int row_off = y * width;
            for (int x = 0; x < width; ++x) {
                if (!src[row_off + x]) {
                    continue;
                }
                const int y_min = (y - radius < 0) ? 0 : (y - radius);
                const int y_max = (y + radius >= height) ? (height - 1) : (y + radius);
                const int x_min = (x - radius < 0) ? 0 : (x - radius);
                const int x_max = (x + radius >= width) ? (width - 1) : (x + radius);
                for (int ny = y_min; ny <= y_max; ++ny) {
                    const int base = ny * width;
                    for (int nx = x_min; nx <= x_max; ++nx) {
                        if (abs(nx - x) + abs(ny - y) <= radius) {
                            dst[base + nx] = true;
                        }
                    }
                }
            }
        }

        if (iter + 1 < iterations) {
            bool *tmp = src;
            src = dst;
            dst = tmp;
        }
    }

    int newly_activated = 0;
    bool *final_mask = dst; // dst holds the latest result after the loop
    for (size_t i = 0; i < total; ++i) {
        if (final_mask[i] && !mask[i]) {
            ++newly_activated;
        }
        mask[i] = final_mask[i];
    }

    free(src);
    free(dst);
    return newly_activated;
}


float gaus(float x, float med, float sigma)
{
    return expf(-(x - med) * (x - med) / (2 * sigma * sigma)) / (sqrtf(2 * PI) * sigma);
}

// Helper: judge whether two time columns are nearly identical across channels
static inline int columns_similar_cols(
    const float *data,
    int nsamp,
    int nchan,
    int ta,
    int tb,
    float abs_epsilon,
    float rel_sigma)
{
    double sum_abs_diff = 0.0;
    double sum_abs_base = 0.0;
    int eq_cnt = 0;
    for (int c = 0; c < nchan; ++c) {
        size_t ia = (size_t)c * (size_t)nsamp + (size_t)ta;
        size_t ib = (size_t)c * (size_t)nsamp + (size_t)tb;
        float va = data[ia];
        float vb = data[ib];
        float dv = va - vb;
        float ad = fabsf(dv);
        sum_abs_diff += (double)ad;
        sum_abs_base += (double)(0.5f * (fabsf(va) + fabsf(vb)));
        if (ad <= abs_epsilon) eq_cnt++;
    }
    double mean_diff = sum_abs_diff / (double)nchan;
    double mean_base = sum_abs_base / (double)nchan;
    // 更严格：同时满足绝对和相对两个门槛（取更小者）
    double thr_abs = (double)abs_epsilon;
    double thr_rel = rel_sigma * mean_base;
    double thr = (thr_abs < thr_rel) ? thr_abs : thr_rel;
    // 要求绝大多数通道几乎相等，避免整体偏移引发误判
    const double eq_frac_min = 0.98; // 至少98%的通道在 abs_epsilon 内
    double eq_frac = (double)eq_cnt / (double)nchan;
    return (mean_diff <= thr && eq_frac >= eq_frac_min) ? 1 : 0;
}

// Simple Gaussian fit: use mean and std as mu and sigma
// void fit_gaussian(float *data, int n, float *mu, float *sigma) {
//     float sum = 0.0f, sum_sq = 0.0f;
//     int i;
//     for (i = 0; i < n; i++) {
//         sum += data[i];
//         sum_sq += data[i] * data[i];
//     }
//     *mu = sum / n;
//     *sigma = sqrtf((sum_sq / n) - (*mu * *mu));
// }

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
     /* (Variables channel_stds/channel_stds_temp unused and removed) */
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

void subChanMed(float *data, int nsamp, int nchan, float *channel_medians, float *temp_data)
{
    printf("=== Subtracting channel medians from data ===\n");

    if (nchan <= 0 || nsamp <= 0) {
        printf("Channel median subtraction completed for %d channels\n", nchan);
        return;
    }

    // 1) Compute per-channel median. Prefer parallel execution with per-thread scratch.
    int nthreads = omp_get_max_threads();
    float **thread_bufs = (float**)malloc((size_t)nthreads * sizeof(float*));
    int alloc_ok = (thread_bufs != NULL);
    if (alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            thread_bufs[t] = (float*)malloc((size_t)nsamp * sizeof(float));
            if (!thread_bufs[t]) {
                alloc_ok = 0;
                // cleanup partially allocated
                for (int k = 0; k < t; ++k) free(thread_bufs[k]);
                free(thread_bufs);
                thread_bufs = NULL;
                break;
            }
        }
    }

    if (alloc_ok && thread_bufs) {
    // #pragma omp parallel for schedule(static)
        for (int i = 0; i < nchan; i++) {
            int tid = omp_get_thread_num();
            float *tmp = thread_bufs[tid];
            // Copy channel data for median calculation (median() reorders data)
            memcpy(tmp, data + (size_t)i * nsamp, (size_t)nsamp * sizeof(float));
            channel_medians[i] = median(tmp, nsamp);
        }
        for (int t = 0; t < nthreads; ++t) free(thread_bufs[t]);
        free(thread_bufs);
    } else {
        // Fallback: serial computation using caller-provided temp_data
        for (int i = 0; i < nchan; i++) {
            memcpy(temp_data, data + (size_t)i * nsamp, (size_t)nsamp * sizeof(float));
            channel_medians[i] = median(temp_data, nsamp);
        }
    }

    // 2) Subtract median from each channel (safe to parallelize; each channel slice is independent)
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < nchan; i++) {
        const float med = channel_medians[i];
        float *row = data + (size_t)i * nsamp;
        for (int j = 0; j < nsamp; j++) {
            row[j] -= med;
        }
    }

    printf("Channel median subtraction completed for %d channels\n", nchan);
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

/*
 * ksigma_2d: estimate sigma by flattening unmasked data and calling ksigma_1d.
 * To avoid repeated large heap allocations, caller may provide two pre-allocated
 * buffers: `unmasked_buf` (length at least nsamp*nchan) and `median_temp_buf`
 * (length at least nsamp*nchan). If either is NULL, function will allocate/free
 * locally as before.
 */
float ksigma_2d(const float *dataT, const int *mask_chanRFI, int nsamp, int nchan,
                float *unmasked_buf, float *median_temp_buf)
{
    int i;
    int total_size = nsamp * nchan;

    /* prepare unmasked buffer (either caller-provided or locally allocated) */
    float *unmasked_data = unmasked_buf;
    int allocated_local_unmasked = 0;
    if (!unmasked_data) {
        unmasked_data = (float *)malloc(total_size * sizeof(float));
        if (!unmasked_data) return 0.0f;
        allocated_local_unmasked = 1;
    }

    int unmasked_count = 0;
    for (i = 0; i < total_size; i++) {
        if (mask_chanRFI == NULL || mask_chanRFI[i] == 0) {
            unmasked_data[unmasked_count++] = dataT[i];
        }
    }

    /* Use `ksigma_1d` on the flattened array */
    int bins = 50; /* small; safe on stack */
    float hist[bins];
    float x_val[bins];
    memset(hist, 0, sizeof(hist));

    float *median_temp = median_temp_buf;
    int allocated_local_mtemp = 0;
    if (!median_temp) {
        median_temp = (float *)malloc((unmasked_count > 0 ? unmasked_count : 1) * sizeof(float));
        if (!median_temp) {
            if (allocated_local_unmasked) free(unmasked_data);
            return 0.0f;
        }
        allocated_local_mtemp = 1;
    }

    float sigma = ksigma_1d(unmasked_data, unmasked_count, bins, hist, x_val, median_temp);

    if (allocated_local_unmasked) free(unmasked_data);
    if (allocated_local_mtemp) free(median_temp);
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
     /* Small arrays used as parameters to the 1D routine: allocate on stack (VLA)
         to avoid frequent malloc/free. M_len is expected to be small (few elements).
         These decay to pointers when passed to functions. */
     float M[M_len];
     float chi_i[M_len];
    
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
    /* Gaussian kernel is small (kernel_m typically ~40) — allocate on stack */
    float gaussian_kernel_m[kernel_m];
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
    // #pragma omp parallel for
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
    
    /* gaussian_kernel_m is on the stack now, no free needed */
    
    // Use residual data for thresholding
    // #pragma omp parallel for collapse(2)
    for (j = 0; j < nchan; j++) {
        for (i = 0; i < nsamp; i++) {
            temp_dataT[j * nsamp + i] = (temp_dataT[j * nsamp + i] - smoothed_data[j * nsamp + i]) / (global_std + 1e-6f);
        }
    }
    
    free(smoothed_data);
    
    // Calculate chi_1 after background removal
    /* Allocate buffers once and pass into ksigma_2d to avoid large per-call allocs */
    float *ksigma_unmasked = (float *)malloc(nsamp * nchan * sizeof(float));
    float *ksigma_mtemp = (float *)malloc(nsamp * nchan * sizeof(float));
    float chi_1 = timesOfSigma * ksigma_2d(temp_dataT, mask_chanRFI, nsamp, nchan,
                                           ksigma_unmasked, ksigma_mtemp);
    free(ksigma_unmasked);
    free(ksigma_mtemp);

    // Time-axis processing with optimized 1D
    // #pragma omp parallel for
    for (j = 0; j < nchan; j++) {
        sumthreshold_1d(&temp_dataT[j * nsamp], nsamp, &mask[j * nsamp], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Transpose for frequency processing
    float *transposed_data = (float *)malloc(nsamp * nchan * sizeof(float));
    transpose(temp_dataT, nchan, nsamp, transposed_data);

    // Frequency-axis processing
    // #pragma omp parallel for
    for (i = 0; i < nsamp; i++) {
        sumthreshold_1d(&transposed_data[i * nchan], nchan, &temp_maskT[i * nchan], 
                       chi_1, M_len, temp_data_1d, local_mask_1d, M, chi_i);
    }

    // Merge masks
    // #pragma omp parallel for collapse(2)
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
    /* M, chi_i and gaussian_kernel_m were allocated on the stack (VLA) */
}

// Function to randomly replace pixels of channels flagged in a 1D channel mask
// For each time sample (column), values are sampled from unflagged channels at the same time
// channelMask[j] == 1 -> this channel's pixels will be replaced; 0 -> used as source pool only
// If pointMask is provided (non-NULL), only unmasked pixels (pointMask[k*nsamp+i]==0) are allowed in the source pool
void outChanSubstitution(float *data, const int *channelMask, const int *pointMask, int nsamp, int nchan)
{
    int i, j;

    /* Allocate source_values once and reuse across time samples to avoid
       allocating/freeing in every iteration of the inner loop. */
    float *source_values = (float*)malloc(nchan * sizeof(float));
    if (!source_values) return;

    for (i = 0; i < nsamp; i++) {
        // For each time sample (column), collect values from unflagged channels (channelMask==0)
        int source_count = 0;

        for (j = 0; j < nchan; j++) {
            if (channelMask[j] == 0) {
                if (!pointMask || pointMask[j * nsamp + i] == 0) {
                    source_values[source_count++] = data[j * nsamp + i];
                }
            }
        }

        if (source_count > 0) {
            unsigned int seed = (unsigned int)(i * 1315423911u + 12345u);
            for (j = 0; j < nchan; j++) {
                if (channelMask[j] != 0) {
                    int random_idx = rand_r(&seed) % source_count;
                    data[j * nsamp + i] = source_values[random_idx];
                }
            }
        } else {
            // No unflagged channels (or all masked by pointMask) at this time sample; zero out flagged channels at this time
            for (j = 0; j < nchan; j++) {
                if (channelMask[j] != 0) {
                    data[j * nsamp + i] = 0.0f;
                }
            }
        }
    }

    free(source_values);
}


/// @brief Substitute masked elements in a 1D array with random samples from unmasked elements.
/// Core implementation that handles the actual pixel substitution logic.
/// @param data Pointer to the data array.
/// @param size Number of elements in the array.
/// @param mask Mask array indicating which elements are masked (1 for masked, 0 for good).
/// @param goodSamps Pre-allocated array of size `size` to hold indices of unmasked elements.
/// @param randIdx Pre-allocated array of size `size` to hold random indices for replacement.
void substPixels(float *data, int size, bool *mask, int *goodSamps, int *randIdx) {
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
    // unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 1000 + size);
    unsigned int seed = (unsigned int)(time(NULL) + size);
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

/// @brief 2D wrapper: perform substitution per-channel using substPixels
/// @param data channel-major layout: channel * nsamp + t
/// @param nsamp number of time samples per channel
/// @param nchan number of channels
/// @param mask int mask array (channel-major) where 1 means masked
void substPixels2D(float *data, int nsamp, int nchan, int *mask) {
    if (!data || !mask || nsamp <= 0 || nchan <= 0) return;
    /* Allocation success is assumed by caller (user accepted risk) */
    int *goodSamps = (int *)malloc((size_t)nsamp * sizeof(int));
    int *randIdx = (int *)malloc((size_t)nsamp * sizeof(int));
    bool *tempBoolMask = (bool *)malloc((size_t)nsamp * sizeof(bool)); // Allocate conversion buffer
    
    if (!goodSamps || !randIdx || !tempBoolMask) {
        free(goodSamps); free(randIdx); free(tempBoolMask);
        return;
    }

    for (int ch = 0; ch < nchan; ++ch) {
        float *chanData = data + (size_t)ch * (size_t)nsamp;
        int *chanMask = mask + (size_t)ch * (size_t)nsamp;
        
        // Convert int mask to bool mask for this channel
        for (int k = 0; k < nsamp; k++) {
            tempBoolMask[k] = (chanMask[k] != 0);
        }
        
        substPixels(chanData, nsamp, tempBoolMask, goodSamps, randIdx);
    }
    free(goodSamps);
    free(randIdx);
    free(tempBoolMask);
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




/// @brief 兜底均值检测：对标准差检测未标记的通道检查均值离群
/// @param data 数据数组
/// @param nsamp 每个通道的样本数
/// @param nchan 通道数
/// @param channelFlagged 通道标记数组（输入/输出）
/// @param pointMask 点干扰标记位图，计算统计量时将跳过标记为真的像素（可选，可为 NULL）
/// @return 额外标记的通道数
int meanOutlierDetection(float *data, int nsamp, int nchan, int *channelFlagged, const bool *pointMask) {
    float *chan_means = (float *)malloc(nchan * sizeof(float));
    int mean_check_count = 0;
    
    // 计算未标记通道的均值
    for (int i = 0; i < nchan; i++) {
        if (!channelFlagged[i]) {  // 只对未标记通道
            double sum = 0;
            int count = 0;
            float *row = data + (size_t)i * nsamp;
            const bool *m = pointMask ? (pointMask + (size_t)i * nsamp) : NULL;
            for (int j = 0; j < nsamp; j++) {
                if (!m || !m[j]) {
                    sum += row[j];
                    count++;
                }
            }
            if (count > 0) {
                chan_means[mean_check_count++] = (float)(sum / count);
            }
        }
    }
    
    int additional_flagged = 0;
    if (mean_check_count >= 3) {  // 至少需要3个通道计算统计
        // 计算均值分布的中位数和标准差
        float mean_median = median(chan_means, mean_check_count);
        float mean_std;
        findMeanStd(chan_means, mean_check_count, NULL, &mean_std);
        
        // 标记均值离群的通道（超过中位数 ± 3*std）
        float upper_bound = mean_median + 3.0f * mean_std;
        float lower_bound = mean_median - 3.0f * mean_std;
        
        for (int i = 0; i < nchan; i++) {
            if (!channelFlagged[i]) {  // 只检查未标记的
                double sum = 0;
                int count = 0;
                float *row = data + (size_t)i * nsamp;
                const bool *m = pointMask ? (pointMask + (size_t)i * nsamp) : NULL;
                for (int j = 0; j < nsamp; j++) {
                    if (!m || !m[j]) {
                        sum += row[j];
                        count++;
                    }
                }
                if (count > 0) {
                    float mean = (float)(sum / count);
                    if (mean > upper_bound || mean < lower_bound) {
                        channelFlagged[i] = 1;  // 标记为异常
                        additional_flagged++;
                    }
                }
            }
        }
    }
    
    free(chan_means);
    return additional_flagged;
}

/* 线程局部保底均值缓冲区，避免在并行循环内频繁 malloc/free */
static _Thread_local float *tls_fallback_channel_means = NULL;
static _Thread_local size_t tls_fallback_channel_means_size = 0;
void setFallbackChannelMeansBuffer(float *buf, size_t size) {
    tls_fallback_channel_means = buf;
    tls_fallback_channel_means_size = size;
}

/* 全局保底均值检测 σ 倍数（默认 2.0，可外部配置） */
static float g_fallback_mean_nsigma = 2.0f;
void setFallbackMeanNSigma(float v) {
    if (v <= 0.0f) {
        g_fallback_mean_nsigma = 2.0f; // 回退到默认
    } else {
        g_fallback_mean_nsigma = v;
    }
}

/* Global toggles controlled by caller (default disabled) */
static int g_useIQRM = 0;
static int g_useCLFD = 0;
static int g_noBlock = 0;
static int g_noVertical = 0;
void setUseIQRM(int v) { g_useIQRM = (v != 0); }
void setUseCLFD(int v) { g_useCLFD = (v != 0); }
void setNoBlock(int v) { g_noBlock = (v != 0); }
void setNoVertical(int v) { g_noVertical = (v != 0); }

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
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold, float nsigma_in, int plot, const bool *pointMask)
{
    const int MAX_ITERATIONS = 30;   // Reduced maximum iterations for faster convergence
    const float STD_CHANGE_THRESHOLD = 0.01f;  // Relaxed standard deviation change rate threshold (1%)
    const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;  // Relaxed median change threshold
    const int MIN_ITERATIONS = 15;  // Minimum iterations before allowing early stop
    
    int i;
    
    // Save initial channel statistics for comparison
    float *initial_channel_stds = (float *)malloc(nchan * sizeof(float));
    if (!initial_channel_stds) {
        fprintf(stderr, "Memory allocation failed for initial_channel_stds in outChanDetection\n");
        return;
    }
    int initial_flagged_count = 0;
    
    // Count initially flagged channels
    for (i = 0; i < nchan; i++) {
        channelFlagged[i] = 0;  // Initialize to 0
    }
    
    // Calculate standard deviation for each channel, ignoring pixels in pointMask
    for (i = 0; i < nchan; i++)
    {
        double sum = 0, sum_sq = 0;
        int count = 0;
        float *row = data + (size_t)i * nsamp;
        const bool *m = pointMask ? (pointMask + (size_t)i * nsamp) : NULL;
        
        for (int j = 0; j < nsamp; j++) {
            if (!m || !m[j]) {
                float val = row[j];
                sum += val;
                sum_sq += (double)val * val;
                count++;
            }
        }
        
        if (count > 1) {
            double mean = sum / count;
            channel_stds[i] = (float)sqrt(fmax(0.0, (sum_sq / count) - (mean * mean)));
        } else {
            // Fallback: if entire channel is point-masked or too few samples, use dummy findMeanStd
            float dummy_m, dummy_s;
            findMeanStd(row, nsamp, &dummy_m, &dummy_s);
            channel_stds[i] = dummy_s;
        }
        initial_channel_stds[i] = channel_stds[i]; // Save initial value for comparison
    }
    
    // Fit Gaussian to initial channel stds to get mu and sigma
    // float fitted_mu, fitted_sigma;
    // fit_gaussian(initial_channel_stds, nchan, &fitted_mu, &fitted_sigma);  // Assume fit_gaussian is implemented
    
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
        free(channel_mask);
        free(valid_stds);
        return;
    }
    
    while (iter < MAX_ITERATIONS && valid_count >= 3)
    {
    // Calculate current median and std from that median in one pass
    findMedianStd(valid_stds, valid_count, &current_median, &current_std);
        
        // Check convergence conditions after first iteration
        if (iter > 0) {
            median_change = fabsf(current_median - last_median);
            std_change_rate = (last_std > 0) ? fabsf(current_std - last_std) / last_std : 0.0f;
            
            if (median_change < MEDIAN_CHANGE_THRESHOLD && std_change_rate < STD_CHANGE_THRESHOLD) {
                printf("Converged at iter %d (median_change=%.6f, std_change=%.6f)\n", iter, median_change, std_change_rate);
                break;  // Stop iterating if converged
            }
        }
        
        // Calculate bounds
        float upper_bound = current_median + channel_std_threshold * current_std;
        float lower_bound = current_median - channel_std_threshold * current_std;
        lower_bound = (lower_bound < 0) ? 0 : lower_bound; // Ensure non-negative lower bound
        
        // Flag new outlier channels and rebuild valid data array
        int new_outliers = 0;
        valid_count = 0;  // Reset and rebuild
        
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
        
        // Break if no new outliers found (algorithm has converged)
        if (new_outliers == 0 && iter >= MIN_ITERATIONS) {
            break;
        }
        
        // Update for next iteration
        last_median = current_median;
        last_std = current_std;
        iter++;
    }
    
    // Count final flagged channels
    int final_flagged_count = 0;
    for (i = 0; i < nchan; i++) {
        if (channelFlagged[i]) final_flagged_count++;
    }
    
    printf("outChannel detection completed (%d iterations, %d channels flagged)\n", iter, final_flagged_count);
    
    // Smooth the mask to remove isolated flagged channels
    smoothOutChanMask(channelFlagged, nchan, 2);
    
    // Recount after smoothing
    final_flagged_count = 0;
    for (i = 0; i < nchan; i++) {
        if (channelFlagged[i]) final_flagged_count++;
    }
    printf("After smoothing: %d channels flagged\n", final_flagged_count);
    
    // === 兜底均值检测 ===
    int additional_flagged = meanOutlierDetection(data, nsamp, nchan, channelFlagged, pointMask);
    final_flagged_count += additional_flagged;
    printf("Mean-based outlier check: flagged %d additional channels (total now: %d)\n", 
           additional_flagged, final_flagged_count);

    // === 保底：全局通道均值分布 ±2σ 检测（使用线程局部预分配缓冲区）===
    if (tls_fallback_channel_means && tls_fallback_channel_means_size >= (size_t)nchan && nchan >= 3) {
        float *buf = tls_fallback_channel_means;
        for (int ci = 0; ci < nchan; ++ci) {
            double sum = 0;
            int count = 0;
            float *row = data + (size_t)ci * (size_t)nsamp;
            const bool *m = pointMask ? (pointMask + (size_t)ci * (size_t)nsamp) : NULL;
            for (int k = 0; k < nsamp; k++) {
                if (!m || !m[k]) {
                    sum += row[k];
                    count++;
                }
            }
            if (count > 0) {
                buf[ci] = (float)(sum / count);
            } else {
                float ch_mean, ch_std_dummy;
                findMeanStd(row, nsamp, &ch_mean, &ch_std_dummy);
                buf[ci] = ch_mean;
            }
        }
        float global_mean, global_std;
        findMeanStd(buf, nchan, &global_mean, &global_std);
        if (global_std > 0.0f) {
            float thresh = g_fallback_mean_nsigma * global_std;
            float upper_b = global_mean + thresh;
            float lower_b = global_mean - thresh;
            int fallback_flagged = 0;
            for (int ci = 0; ci < nchan; ++ci) {
                if (!channelFlagged[ci]) {
                    float val = buf[ci];
                    if (val > upper_b || val < lower_b) {
                        channelFlagged[ci] = 1;
                        fallback_flagged++;
                    }
                }
            }
            if (fallback_flagged > 0) {
                final_flagged_count += fallback_flagged; // 仅在有新增时更新统计；不打印冗长提示
            }
        }
    }
    
    // Create final statistics array (mark flagged channels with negative values)
    float *final_channel_stds = (float *)malloc(nchan * sizeof(float));
    for (i = 0; i < nchan; i++) {
        if (channelFlagged[i]) {
            final_channel_stds[i] = -1.0f; // Mark as flagged
        } else {
            final_channel_stds[i] = channel_stds[i]; // Keep current value
        }
    }
    
    // Display comparison histogram if significant changes occurred and plotting is enabled
    if (total_flagged > 0 && plot) {
        printf("\n=== Displaying outChannel Detection Comparison ===\n");
        drawOutChannelComparisonHist(initial_channel_stds, final_channel_stds, nchan,
              initial_flagged_count, final_flagged_count, iter, channel_std_threshold, nsigma_in);
        printf("=== OutChannel Comparison Complete ===\n");
    }
    
    // Cleanup
    free(initial_channel_stds);
    free(final_channel_stds);
    free(channel_mask);
    free(valid_stds);
}



void drawChanStatHist(float *data, int nsamp, int nchan, int plot)
{
    int i;
    
    const char* stat_name = "STD";
    const char* stat_symbol = "\\gs";
    const char* stat_label = "Channel STD \\gs\\dj\\u";
    
    printf("\n=== %s Histogram Analysis ===\n", stat_name);
    
    // Allocate memory for channel statistics
    float *channel_stat = (float *)malloc(nchan * sizeof(float));
    if (!channel_stat) {
        fprintf(stderr, "Memory allocation failed for channel_stat in drawChanStatHist\n");
        return;
    }
     float *channel_median = (float *)malloc(nchan * sizeof(float));
     /* temp_data is used both for per-channel sample buffers (nsamp) and
         later as a workspace to hold nchan values (e.g. copying channel_stat
         into it). Allocate the maximum of the two to avoid out-of-bounds
         accesses when nchan > nsamp (observed when channel count grows).
     */
     int temp_len = (nsamp > nchan) ? nsamp : nchan;
     float *temp_data = (float *)malloc((size_t)temp_len * sizeof(float));
    
    // Calculate statistics for each channel
    for (i = 0; i < nchan; i++)
    {
        // Copy channel data for processing
        memcpy(temp_data, data + i * nsamp, nsamp * sizeof(float));
        
        // Calculate median of the channel
        channel_median[i] = median(temp_data, nsamp);
        
        if (nsamp > 1) {
            channel_stat[i] = stdFromMedian(data + i * nsamp, nsamp);
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
    for (i = 0; i < nchan; i++) {
        temp_data[i] = fabsf(channel_stat[i] - stat_median);
    }
    float dispersion_value = median(temp_data, nchan);
    stat_mad = dispersion_value;
    
    // Print statistics in original format
    printf("\n=== Channel STD σ_j Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("STD Median: %.6f\n", stat_median);
    printf("STD STD: %.6f\n", dispersion_value);
    
    if (!plot) {
        printf("%s histogram plotting disabled, skipping visualization.\n", stat_name);
        free(channel_stat);
        free(channel_median);
        free(temp_data);
        return;
    }

    // Ensure PGPLOT device available before any drawing
    if (!ensure_pgplot_device(NULL)) {
        printf("PGPLOT unavailable, skip visualization.\n");
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
    printf("\n=== Channel STD σ_j Statistics ===\n");
    printf("Total channels: %d\n", nchan);
    printf("STD Median: %.6f\n", stat_median);
    printf("STD STD: %.6f\n", dispersion_value);
    printf("STD Min: %.6f\n", stat_min);
    printf("STD Max: %.6f\n", stat_max);
    
    // Unified 11-threshold line system
    const int NUM_ALL_THRESHOLDS = 11;
    float all_thresh_values[11];
    const char* all_threshold_labels[11];
    
    // Set labels based on statistical method
    const char* std_labels[11] = {
        "-5*STD", "-4*STD", "-3*STD", "-2*STD", "-1*STD",
        "Median",
        "1*STD", "2*STD", "3*STD", "4*STD", "5*STD"
    };
    for (i = 0; i < 11; i++) all_threshold_labels[i] = std_labels[i];
    
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
    cpglab(stat_label, "Number of Channels", "Channel STD \\gs\\dj\\u Distribution");
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
        float x_curve = plot_min + (float)i * curve_step;
        float y_curve = main_hist_amplitude * expf(-(x_curve - global_fitted_mu) * (x_curve - global_fitted_mu) / (2 * global_fitted_sigma * global_fitted_sigma));
        if (i == 0) {
            cpgmove(x_curve, y_curve);
        } else {
            cpgdraw(x_curve, y_curve);
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
            
            float x_zoom, y_zoom;
            float zoom_curve_points = 200;
            for (i = 0; i < (int)zoom_curve_points; i++) {
                x_zoom = zoom_min + (float)i * (zoom_max - zoom_min) / (zoom_curve_points - 1.0f);
                y_zoom = gaus_with_amplitude(x_zoom, zoom_hist_amplitude, global_fitted_mu, global_fitted_sigma);
                
                if (i == 0) {
                    cpgmove(x_zoom, y_zoom);
                } else {
                    cpgdraw(x_zoom, y_zoom);
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
 * @return Total number of outliers detected
 */
int inChanOutlierIter(float *data, int nsamp, float Nsigma, 
                                   bool *mask, float *median_temp)
{
    const int MAX_ITERATIONS = 15;   // Increased maximum iterations
    const float STD_CHANGE_THRESHOLD = 0.0001f;  // Standard deviation change rate threshold (0.01%)
    const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;  // Median change threshold
    const float EPS_STD = 1e-12f;    // Tiny epsilon to avoid divide-by-zero and detect degenerate std
    
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
    
    if (valid_count < 3) return 0;
    
    while (iter < MAX_ITERATIONS && valid_count >= 3)
    {
        // Calculate current median and std from that median in one pass
        findMedianStd(median_temp, valid_count, &current_median, &current_std);

        // Early stop: degenerate standard deviation
        if (current_std <= EPS_STD) break;

        // Check convergence conditions after first iteration
        if (iter > 0) {
            median_change = fabsf(current_median - last_median);
            float denom = (last_std > EPS_STD) ? last_std : EPS_STD;
            std_change_rate = fabsf(current_std - last_std) / denom;
            if (median_change < MEDIAN_CHANGE_THRESHOLD && std_change_rate < STD_CHANGE_THRESHOLD) {
                break;  // Converged — avoid an extra full pass
            }
        }

        // Calculate N-sigma bounds
        float upper_bound = current_median + Nsigma * current_std;
        float lower_bound = current_median - Nsigma * current_std;

        // Flag new outliers and rebuild valid data array
        int new_outliers = 0;
        int new_valid_count = 0;  // Rebuild count

        for (int i = 0; i < nsamp; i++) {
            if (mask[i] == 0) {  // Currently unmasked
                if (data[i] > upper_bound || data[i] < lower_bound) {
                    mask[i] = 1;  // Flag as outlier
                    new_outliers++;
                    total_outliers++;
                } else {
                    median_temp[new_valid_count] = data[i];  // Keep in valid data
                    new_valid_count++;
                }
            }
        }

        // Early stop: no new outliers in this iteration
        if (new_outliers == 0) {
            break;
        }

        // Update for next iteration
        valid_count = new_valid_count;
        last_median = current_median;
        last_std = current_std;
        iter++;
    }

    return total_outliers;
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
int inChanDetection(float *data, int nsamp, int nchan, float Nsigma,
    bool *horizontalMask, int *channel_fully_flagged,
    float *scratch, size_t scratch_count)
{
    int totalOutliers = 0;

    int threads = omp_get_max_threads();
    if (threads < 1) threads = 1;
    size_t required = (size_t)threads * (size_t)nsamp;

    if (scratch && scratch_count >= required) {
    // #pragma omp parallel reduction(+:totalOutliers)
        {
            const int t = omp_get_thread_num();
            float *median_temp_local = scratch + (size_t)t * (size_t)nsamp;

            // #pragma omp for schedule(static)
            for (int i = 0; i < nchan; i++)
            {
                int channelOutliers = inChanOutlierIter(
                    data + (size_t)i * (size_t)nsamp, nsamp, Nsigma,
                    horizontalMask + (size_t)i * (size_t)nsamp, median_temp_local);
                totalOutliers += channelOutliers;
            }
        }
        return totalOutliers;
    }

    // Fallback: sequential processing with on-stack buffer if scratch is unavailable or undersized
    float *temp = (float *)malloc((size_t)nsamp * sizeof(float));
    if (!temp) {
        return totalOutliers; // give up silently if allocation fails
    }

    for (int i = 0; i < nchan; i++)
    {
        int channelOutliers = inChanOutlierIter(
            data + (size_t)i * (size_t)nsamp, nsamp, Nsigma,
            horizontalMask + (size_t)i * (size_t)nsamp, temp);
        totalOutliers += channelOutliers;
    }

    free(temp);
    return totalOutliers;
}


// If a channel has > ratio_thresh fraction of samples flagged in pointMask,
// treat it as point-dominated: clear that channel's horizontalMask and flaggedChans.
// Returns number of channels canceled.
int cancelHorizontalMaskForPointDominantChannels(
    const bool *pointMask,
    int nsamp,
    int nchan,
    float ratio_thresh,
    bool *horizontalMask,
    int *flaggedChans)
{
    if (nsamp <= 0 || nchan <= 0) return 0;
    int canceled = 0;
    for (int ch = 0; ch < nchan; ++ch) {
        int pm_count = 0;
        size_t base = (size_t)ch * (size_t)nsamp;
        for (int t = 0; t < nsamp; ++t) {
            if (pointMask[base + (size_t)t]) pm_count++;
        }
        float pm_ratio = (float)pm_count / (float)nsamp;
        if (pm_ratio > ratio_thresh) {
            // Clear horizontal mask for this channel
            for (int t = 0; t < nsamp; ++t) {
                horizontalMask[base + (size_t)t] = 0;
            }
            // Keep bookkeeping consistent
            if (flaggedChans) flaggedChans[ch] = 0;
            canceled++;
        }
    }
    return canceled;
}

/**
 * @brief Apply pixel substitution for flagged pixels in channels not killed by killThresh
 * @param data Data array (will be modified for pixel substitution)
 * @param globalMask RFI mask array (input only, not modified)
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param pixelsSubstituted Output: number of pixels substituted
 */
void inChanSubstitution(float *data, bool *globalMask, int nsamp, int nchan, int *pixelsSubstituted)
{
    *pixelsSubstituted = 0;
    printf("\n=== Pixel Substitution ===\n");
    
    int i;
    int localPixelsSubstituted = 0;
    
    for (i = 0; i < nchan; i++) {
        // Count masked pixels in this channel
        int channelMaskedCount = 0;
        for (int samp = 0; samp < nsamp; samp++) {
            int idx = samp + i * nsamp;
            if (globalMask[idx]) {
                channelMaskedCount++;
            }
        }
        
        // Skip channels with no flagged pixels
        if (channelMaskedCount == 0) continue;
        
        // Skip channels that are fully flagged (killed by killThresh)
        if (channelMaskedCount == nsamp) continue;
        
        // Substitute pixels in this channel
        float *channelData = data + i * nsamp;
        bool *channelMask = globalMask + i * nsamp;
        
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

/**
 * @brief Classify flagged channels into bright, dark, and complex based on channel means and continuity
 * @param data Data array (nsamp * nchan)
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param flaggedChans Array indicating which channels are flagged (1 if flagged, 0 otherwise)
 * @param brightMask Output mask for bright channels (set to 1 for entire bright flagged channels)
 * @param darkMask Output mask for dark channels (set to 1 for entire dark flagged channels)
 * @param complexMask Output mask for complex channels (set to 1 for entire complex flagged channels)
 */
void classifyChannels(float *data, int nsamp, int nchan, int *flaggedChans, bool *brightMask, bool *darkMask, bool *complexMask) {
    if (!(brightMask || darkMask || complexMask) || nchan <= 0 || !flaggedChans) return;

    const float eps = 1e-5f;

    float *channelMeans = (float *)malloc(nchan * sizeof(float));
    if (!channelMeans) {
        fprintf(stderr, "Memory allocation failed in classifyChannels\n");
        return;
    }

    double sumMeans = 0.0;
    for (int i = 0; i < nchan; i++) {
        double sum = 0.0;
        int base = i * nsamp;
        for (int j = 0; j < nsamp; j++) {
            sum += data[base + j];
        }
        float mean = (nsamp > 0) ? (float)(sum / nsamp) : 0.0f;
        channelMeans[i] = mean;
        sumMeans += mean;
    }

    float overallChannelMean = (nchan > 0) ? (float)(sumMeans / nchan) : 0.0f;

    int idx = 0;
    while (idx < nchan) {
        if (!flaggedChans[idx]) {
            idx++;
            continue;
        }

        int start = idx;
        float blockSum = 0.0f;
        while (idx < nchan && flaggedChans[idx]) {
            blockSum += channelMeans[idx];
            idx++;
        }
        int end = idx - 1;
        int blockLen = end - start + 1;

        bool *targetMask = NULL;
        if (blockLen <= 2) {
            float blockMean = blockSum / (float)blockLen;
            if (blockMean > overallChannelMean + eps && brightMask) {
                targetMask = brightMask;
            } else if (blockMean < overallChannelMean - eps && darkMask) {
                targetMask = darkMask;
            } else {
                targetMask = complexMask;
            }
        } else {
            int allAbove = 1;
            int allBelow = 1;
            for (int k = start; k <= end; k++) {
                float mean = channelMeans[k];
                if (!(mean > overallChannelMean + eps)) allAbove = 0;
                if (!(mean < overallChannelMean - eps)) allBelow = 0;
                if (!allAbove && !allBelow) break;
            }

            if (allAbove && brightMask) {
                targetMask = brightMask;
            } else if (allBelow && darkMask) {
                targetMask = darkMask;
            } else {
                targetMask = complexMask;
            }
        }

        if (targetMask) {
            for (int k = start; k <= end; k++) {
                int base = k * nsamp;
                for (int j = 0; j < nsamp; j++) {
                    targetMask[base + j] = 1;
                }
            }
        }
    }

    free(channelMeans);
}

void identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan,
    int iterationIndex, int plot, int doSubstitute,
    IdentNSigmaMasks *masks,
    float *finalMedian, float *finalStd, int cudaReady, int *flaggedChans,
    int *identSubst_goodSamps, int *identSubst_randIdxs, float *identSubst_medTemp,
    float *inChanScratch, size_t inChanScratchCount,
    int *clfd_mask_buf,
    float *vs_time_means_buf, unsigned char *vs_flag_time_buf)
{
    bool *horizontalMask = masks->horizontalMask;
    bool *verticalMask = masks->verticalMask;
    bool *blockMask = masks->blockMask;  // New: local pointer for blockMask
    bool *periodicMask = masks->periodicMask; // New: local pointer for periodicMask
    bool *globalMask = masks->globalMask;
    bool *pointMask = masks->pointMask;
    bool *brightMask = masks->chanBrightMask;
    bool *darkMask = masks->chanDarkMask;
    bool *complexMask = masks->chanComplexMask;
    memset(horizontalMask, 0, nsamp * nchan * sizeof(bool));
    memset(verticalMask, 0, nsamp * nchan * sizeof(bool));
    memset(blockMask, 0, nsamp * nchan * sizeof(bool));  // New: initialize blockMask
    memset(periodicMask, 0, nsamp * nchan * sizeof(bool)); // New: initialize periodicMask
    memset(globalMask, 0, nsamp * nchan * sizeof(bool));
    memset(pointMask, 0, nsamp * nchan * sizeof(bool));
    memset(brightMask, 0, nsamp * nchan * sizeof(bool));
    memset(darkMask, 0, nsamp * nchan * sizeof(bool));
    memset(complexMask, 0, nsamp * nchan * sizeof(bool));
    
    memset(flaggedChans, 0, sizeof(int) * nchan);
    memcpy(identSubst_medTemp, data, (size_t)nsamp * (size_t)nchan * sizeof(float));
    
    int i;
    
    printf("identSubstNSigma: doSubstitute=%d\n", doSubstitute);

    if (plot)
    {
        printf("=== Generating Channel STD sigma_j Histogram (Iteration %d) ===\n", iterationIndex);
        drawChanStatHist(data, nsamp, nchan, 1);
        printf("=== STD sigma_j Histogram Complete ===\n");
    }
    
    // === 1. inChannel Detection ===
    printf("=== Performing point-level (pixel) outlier detection ===\n");
    double inchan_start = omp_get_wtime();
    int pixelOutliers = 0;
    pixelOutliers = inChanDetection(data, nsamp, nchan, NSigmaInChan, pointMask,
        flaggedChans, inChanScratch, inChanScratchCount);
    
    double inchan_time = omp_get_wtime() - inchan_start;
    printf("Point-level detection: flagged %d outlier pixels (%.4f seconds)\n", pixelOutliers, inchan_time);
    int isolated_removed = eraseIsolatedPixels(pointMask, nsamp, nchan, 3);
    int dilated_added = dilateMask(pointMask, nsamp, nchan, 1, 1);
    printf("Post-processing: pointMask delta is %d\n", (dilated_added - isolated_removed));
    // Accumulate point-wise mask into global
    logicalOR(globalMask, pointMask, nsamp, nchan);
    
    // === 2. inChannel Pixel Substitution ===
    double subst_start = omp_get_wtime();
    int inChanPixelsSubstituted = 0;
    if (doSubstitute) {
        inChanSubstitution(data, pointMask, nsamp, nchan, &inChanPixelsSubstituted);
    } else {
        // detection-only mode: do not modify data
        if (plot || 1) {
            printf("inChannel substitution: skipped (detection-only mode)\n");
        }
    }
    double subst_time = omp_get_wtime() - subst_start;
    printf("inChannel substitution: replaced %d pixels (%.4f seconds)\n", inChanPixelsSubstituted, subst_time);

    // (Removed) killThresh: disabled per user request
    
    // === 4. outChannel Detection ===
    printf("=== Performing channel-level outlier detection ===\n");
    double outchan_start = omp_get_wtime();

    if (g_useIQRM) {
        // Use IQRM to generate a 2D mask (will allocate a temporary mask which we copy into horizontalMask)
        int *tmp_mask2d = NULL;
        float q_chan = 1.5f;
        float thr = 3.0f;
        int flagged_channels = IQRM(data, nsamp, nchan, q_chan, thr, &tmp_mask2d);
        if (tmp_mask2d) {
            // copy into horizontalMask (channel-major layout expected)
            for (int ch = 0; ch < nchan; ++ch) {
                for (int t = 0; t < nsamp; ++t) {
                    horizontalMask[ch * nsamp + t] = (tmp_mask2d[ch * nsamp + t] != 0);
                }
            }
            free(tmp_mask2d);
        }
        // populate flaggedChans from horizontalMask
        for (int ch = 0; ch < nchan; ++ch) {
            int flagged = 0;
            for (int t = 0; t < nsamp; ++t) if (horizontalMask[ch * nsamp + t]) { flagged = 1; break; }
            flaggedChans[ch] = flagged;
        }
        double outchan_time = omp_get_wtime() - outchan_start; (void)outchan_time;
        printf("IQRM-based outChannel detection done, flagged %d channels (approx)\n", flagged_channels);
    }
    else if (g_useCLFD) {
        // Use caller-provided CLFD mask buffer to avoid per-call malloc/free
        int *mask_buf = clfd_mask_buf;
        if (!mask_buf) {
            // fallback
            fprintf(stderr, "Warning: CLFD buffer not provided to identSubstNSigma; falling back to outChanDetection\n");
            float *channel_stds = (float *)malloc(nchan * sizeof(float));
            float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
            outChanDetection(data, nsamp, nchan, flaggedChans, channel_stds, channel_stds_temp, NSigmaOutChan, NSigmaInChan, plot, pointMask);
            free(channel_stds);
            free(channel_stds_temp);
            expandChannelMask(flaggedChans, horizontalMask, nsamp, nchan);
        } else {
            memset(mask_buf, 0, sizeof(int) * (size_t)nchan * (size_t)nsamp);
            CLFD(data, nsamp, nchan, 3.0f, NULL, 0, mask_buf);
            // copy int mask_buf into horizontalMask (bool) and flaggedChans
            for (int ch = 0; ch < nchan; ++ch) {
                int any_flag = 0;
                size_t base = (size_t)ch * (size_t)nsamp;
                for (int t = 0; t < nsamp; ++t) {
                    int v = mask_buf[base + (size_t)t];
                    horizontalMask[base + (size_t)t] = (v != 0);
                    if (v) any_flag = 1;
                }
                flaggedChans[ch] = any_flag;
            }
        }
    }
    else {
        float *channel_stds = (float *)malloc(nchan * sizeof(float));
        float *channel_stds_temp = (float *)malloc(nchan * sizeof(float));
        outChanDetection(data, nsamp, nchan, flaggedChans, channel_stds, channel_stds_temp, NSigmaOutChan, NSigmaInChan, plot, pointMask);
        double outchan_time = omp_get_wtime() - outchan_start;
        printf("outChannel detection completed (%.4f seconds)\n", outchan_time);
        // Expand 1D flaggedChans to 2D horizontalMask
        expandChannelMask(flaggedChans, horizontalMask, nsamp, nchan);
        free(channel_stds);
        free(channel_stds_temp);
    }

    // 新逻辑：按通道统计 pointMask 比例，若超过 30%，认为被点干扰主导，取消该通道的 horizontalMask 标记
    int canceled_horizontal_channels = cancelHorizontalMaskForPointDominantChannels(
        pointMask, nsamp, nchan, 0.30f, horizontalMask, flaggedChans);
    if (canceled_horizontal_channels > 0) {
        printf("Horizontal mask canceled on %d channels due to point-dominated (>30%%) interference.\n",
               canceled_horizontal_channels);
    }
    // Accumulate channel-wise mask into global（注意：在清理 horizontalMask 之后再合并）
    logicalOR(globalMask, horizontalMask, nsamp, nchan);

    // === 5. Out-Chan Cross-channel substitution for fully flagged channels ===
    int outChanPixelsSubstituted = 0;
    int flaggedChanCount = 0;
    for (i = 0; i < nchan; i++) if (flaggedChans[i]) flaggedChanCount++;
    if (flaggedChanCount > 0) {
        if (doSubstitute) {
            outChanSubstitution(data, flaggedChans, (const int*)pointMask, nsamp, nchan);
            // estimate substituted pixels as fully substituted channels * nsamp (conservative estimate)
            outChanPixelsSubstituted = flaggedChanCount * nsamp;
        } else {
            printf("outChannel substitution: skipped (detection-only mode), %d channels flagged\n", flaggedChanCount);
        }
    }
    printf("outChannel substitution: replaced %d pixels\n", outChanPixelsSubstituted);

    // === (NEW) 5.5 Vertical Stripe Detection ===
    if (!g_noVertical && nsamp >= 4 && nchan >= 8) { 
        // Detect broadband vertical stripes via time mean/std peak analysis.
        // Parameters chosen conservatively; can be externalized later.
        float vsigma_mean = 4.0f; // N-sigma threshold on time means
        int   v_min_run   = 1;    // minimum contiguous time samples to accept (1 = single sample)
        // Build exclusion mask: combine point-level and channel-level masks so vertical detection ignores already flagged RFI
        // Allocate a temporary bool array (nsamp*nchan) set true where pixel should be excluded
        // Directly pass pointMask and horizontalMask for exclusion; avoid allocating a combined excludeMask
        // Exclusion rule moved inside detectVerticalStripesByTimeProfiles
        detectVerticalStripesByTimeProfiles(
            data, nsamp, nchan,
            pointMask,
            horizontalMask,
            verticalMask,
            vsigma_mean,
            v_min_run,
            plot,
            vs_time_means_buf,
            vs_flag_time_buf);
        // Merge vertical stripes into global
        logicalOR(globalMask, verticalMask, nsamp, nchan);
        // Additional detector: repeated columns (duplicated time samples)
        // Use conservative near-zero thresholding
        float rep_abs_eps = 1e-6f;
        float rep_rel_sigma = 0.02f; // 更严格的相对阈值，避免过度匹配
        int   rep_min_run = 16; // require at least ~16 consecutive identical columns
        detectVerticalRepeatedColumns(
            data, nsamp, nchan,
            pointMask,
            horizontalMask,
            verticalMask,
            rep_abs_eps,
            rep_rel_sigma,
            rep_min_run,
            plot,
            vs_time_means_buf, // reuse as err buffer
            vs_flag_time_buf);
    } else {
        memset(verticalMask, 0, (size_t)nsamp * (size_t)nchan * sizeof(bool));
    }
    
    // === (NEW) 6. Block RFI Detection ===
    // Simple block detection using connected components on pointMask
    // Apply internal dilation (radius=3 iterations=1) to bridge sparse gaps before CCA
    if (!g_noBlock) {
        detectBlockRFI(pointMask, nsamp, nchan, blockMask,
                       5000, 0.5f,   // min_area, min_density tuned for large radar-like blocks
                       7, 1);        // dilate radius, iterations (adjust if over-merging)
        // High priority: directly overwrite globalMask
        for (int idx = 0; idx < nsamp * nchan; ++idx) {
            if (blockMask[idx]) globalMask[idx] = true;
        }
    } else {
        memset(blockMask, 0, (size_t)nsamp * (size_t)nchan * sizeof(bool));
    }

    // === (DISABLED) Periodic point RFI detection ===
    // 保留实现但不启用：如需启用，取消以下注释。
    // int min_period = 3;
    // int max_period = (nsamp > 200) ? 100 : (nsamp / 2 > 5 ? nsamp/2 : 5);
    // int min_pairs = 3;
    // float min_align_frac = 0.30f;
    // detectPeriodicPointRFI(pointMask, nsamp, nchan, periodicMask,
    //                        min_period, max_period, min_pairs, min_align_frac);
    // logicalOR(globalMask, periodicMask, nsamp, nchan);
    
    int nFlaggedChans = 0;
    for (i = 0; i < nchan; i++) {
        if (flaggedChans[i]) nFlaggedChans++;
    }
    
    if ((brightMask || darkMask || complexMask) && nchan > 0) {
        classifyChannels(identSubst_medTemp, nsamp, nchan, flaggedChans, brightMask, darkMask, complexMask);
    }

    printf("Final status: %d/%d channels fully flagged (%.2f%%)\n", 
           nFlaggedChans, nchan, (float)nFlaggedChans/nchan*100);
    printf("\n=== RFI Detection Statistics ===\n");
    // Accumulate other masks into global
    logicalOR(globalMask, verticalMask, nsamp, nchan);
    logicalOR(globalMask, brightMask, nsamp, nchan);
    logicalOR(globalMask, darkMask, nsamp, nchan);
    logicalOR(globalMask, complexMask, nsamp, nchan);

    // Recompute totals for reporting
    int globalFlagged = 0;
    for (i = 0; i < nsamp * nchan; i++) if (globalMask[i]) globalFlagged++;
    printf("Combined mask (all sources) flagged: %d/%d pixels (%.4f%%)\n", 
        globalFlagged, nsamp*nchan, (float)globalFlagged/(nsamp*nchan)*100);
    printf("=== End RFI Detection Statistics ===\n");
    
    // globalMask already equals union of all masks via logicalOR
    // finalMedian/finalStd are float* outputs; pass them directly (not by address)
    findMedianStd(identSubst_medTemp, nsamp * nchan, finalMedian, finalStd);

    // Calculate total pixels substituted
    int totalPixelsSubstituted = inChanPixelsSubstituted + outChanPixelsSubstituted;
    printf("\n=== Final Processing Summary ===\n");
    printf("  - inChannel pixels substituted: %d\n", inChanPixelsSubstituted);
    printf("  - outChannel pixels substituted: %d\n", outChanPixelsSubstituted);
    printf("  - Total pixels substituted: %d\n", totalPixelsSubstituted);

    // Removed killThresh-based warning

    // Buffers are managed by caller

    printf("### DEBUG: identSubstNSigma exiting with finalMedian=%.6f, finalStd=%.6f ###\n", *finalMedian, *finalStd);
    fflush(stdout);  // Ensure immediate output
}

/* -------------------------------------------------------------------------
 * New version: consistent-shape histogram (initial shape fixed)
 * Keeps original function above; this one only highlights removed vs kept
 * ------------------------------------------------------------------------- */
void drawOutChannelComparisonHist(float *initial_stats, float *final_stats, 
    int nchan, int initial_flagged_count, int final_flagged_count, 
    int iterations, float nsigma_out, float nsigma_in)
{
    const char* stat_name  = "STD";
    const char* stat_label = "Channel STD \\gs\\dj\\u";

    printf("\n=== OutChannel Comparison (CONSISTENT SHAPE) %s Histogram ===\n", stat_name);
    printf("Initial flagged (pre-existing): %d\n", initial_flagged_count);
    printf("Final flagged total: %d\n", final_flagged_count);
    printf("Newly flagged this pass: %d\n", final_flagged_count - initial_flagged_count);

    // 1. Compute initial median / dispersion (for reference line only)
    float *temp_data = (float *)malloc(nchan * sizeof(float));
    if (!temp_data) return;
    memcpy(temp_data, initial_stats, nchan * sizeof(float));
    float initial_median = median(temp_data, nchan);

    // 2. Determine min/max from initial only
    float overall_min = initial_stats[0], overall_max = initial_stats[0];
    for (int i = 1; i < nchan; i++) {
        if (initial_stats[i] < overall_min) overall_min = initial_stats[i];
        if (initial_stats[i] > overall_max) overall_max = initial_stats[i];
    }
    if (overall_max == overall_min) overall_max = overall_min + 1e-6f;

    int nbins = 100; // same bin count
    float bin_width = (overall_max - overall_min) / nbins;
    float *total_hist   = (float *)calloc(nbins, sizeof(float));
    float *kept_hist    = (float *)calloc(nbins, sizeof(float));
    float *flagged_hist = (float *)calloc(nbins, sizeof(float));
    if (!total_hist || !kept_hist || !flagged_hist) {
        free(temp_data);
        free(total_hist); free(kept_hist); free(flagged_hist);
        return;
    }

    // 3. Fill total shape from initial stats
    for (int i = 0; i < nchan; i++) {
        int bin = (int)((initial_stats[i] - overall_min) / bin_width);
        if (bin < 0) bin = 0; if (bin >= nbins) bin = nbins - 1;
        total_hist[bin]++;
    }

    // 4. Split into kept / flagged by inspecting final_stats (<0 flagged)
    for (int i = 0; i < nchan; i++) {
        int bin = (int)((initial_stats[i] - overall_min) / bin_width);
        if (bin < 0) bin = 0; if (bin >= nbins) bin = nbins - 1;
        if (final_stats[i] < 0) flagged_hist[bin]++; else kept_hist[bin]++;
    }

    // 5. Scaling
    float max_count = 0.0f;
    for (int b = 0; b < nbins; b++) if (total_hist[b] > max_count) max_count = total_hist[b];
    if (max_count <= 0) max_count = 1.0f;

    // 6. Plot
    cpgpage();
    cpgvstd();
    cpgsch(1.2);
    cpgswin(overall_min, overall_max, 0, max_count * 1.15f);
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    char title[256];
    sprintf(title, "OutChannel %s Distribution (Shape Fixed)", stat_name);
    cpgmtxt("T", 1.0, 0.5, 0.5, title);
    cpgmtxt("B", 2.0, 0.5, 0.5, stat_label);
    cpgmtxt("L", 2.0, 0.5, 0.5, "Number of Channels");

    // Outline of total distribution
    cpgsci(7); // grey
    cpgsls(2);
    for (int b = 0; b < nbins; b++) {
        if (total_hist[b] > 0) {
            float x1 = overall_min + b * bin_width;
            float x2 = overall_min + (b + 1) * bin_width;
            float y  = total_hist[b];
            cpgmove(x1, 0); cpgdraw(x1, y); cpgdraw(x2, y); cpgdraw(x2, 0);
        }
    }

    // Kept channels (red filled bars)
    cpgsci(2); cpgsls(1);
    for (int b = 0; b < nbins; b++) {
        if (kept_hist[b] > 0) {
            float x1 = overall_min + b * bin_width;
            float x2 = overall_min + (b + 1) * bin_width;
            cpgrect(x1, x2, 0, kept_hist[b]);
        }
    }

    // Flagged channels (green outline)
    cpgsci(3);
    for (int b = 0; b < nbins; b++) {
        if (flagged_hist[b] > 0) {
            float x1 = overall_min + b * bin_width;
            float x2 = overall_min + (b + 1) * bin_width;
            float y  = flagged_hist[b];
            cpgmove(x1, 0); cpgdraw(x1, y); cpgdraw(x2, y); cpgdraw(x2, 0);
        }
    }

    // Initial median line
    cpgsci(1);
    cpgmove(initial_median, 0); cpgdraw(initial_median, max_count * 1.15f);
    cpgptxt(initial_median, max_count * 1.08f, 0.0, 0.0, "Initial Median");

    // Legend & stats
    float lx = overall_min + (overall_max - overall_min) * 0.65f;
    float ly = max_count * 1.05f;
    float dy = max_count * 0.05f;
    cpgsci(7); cpgptxt(lx, ly, 0.0, 0.0, "Outline: Total (initial)");
    cpgsci(2); cpgptxt(lx, ly - dy, 0.0, 0.0, "Red: Kept");
    cpgsci(3); cpgptxt(lx, ly - 2*dy, 0.0, 0.0, "Green: Removed");
    cpgsci(1);
    char stats_text[160];
    sprintf(stats_text, "Removed: %d  Kept: %d  Iter=%d", final_flagged_count - initial_flagged_count,
            nchan - final_flagged_count, iterations);
    cpgptxt(lx, ly - 3*dy, 0.0, 0.0, stats_text);
    sprintf(stats_text, "T_chan=%.2f  T_point=%.2f", nsigma_out, nsigma_in);
    cpgptxt(lx, ly - 4*dy, 0.0, 0.0, stats_text);

    // Cleanup
    free(temp_data);
    free(total_hist);
    free(kept_hist);
    free(flagged_hist);
    printf("Consistent-shape %s histogram (new) completed!\n", stat_name);
}

/* -------------------------------------------------------------------------
 * Broadband vertical stripe detection by time-profile peaks
 * ------------------------------------------------------------------------- */
void detectVerticalStripesByTimeProfiles(
    const float *data,
    int nsamp,
    int nchan,
    const bool *pointMask,
    const bool *horizontalMask,
    bool *verticalMask,
    float nsigma_mean,
    int min_run,
    int plot,
    float *time_means,
    unsigned char *flag_time)
{
    memset(verticalMask, 0, (size_t)nsamp * (size_t)nchan * sizeof(bool));
    if (!time_means || !flag_time) {
        fprintf(stderr, "detectVerticalStripesByTimeProfiles: NULL scratch buffers provided.\n");
        return;
    }

    // Compute per-time-sample mean & std across channels
    // data layout: channel-major (channel * nsamp + t)
    // We'll iterate time index outer for better locality when gathering across channels.
    for (int t = 0; t < nsamp; ++t) {
        double sum = 0.0;
        int count = 0;
        // First pass: mean over unexcluded pixels
        for (int c = 0; c < nchan; ++c) {
            size_t idx = (size_t)c * (size_t)nsamp + (size_t)t;
            if ((pointMask && pointMask[idx]) || (horizontalMask && horizontalMask[idx])) continue;
            sum += (double)data[c * nsamp + t];
            count++;
        }
        if (count == 0) {
            time_means[t] = 0.0f;
            continue;
        }
        double mean = sum / (double)count;
        time_means[t] = (float)mean;
    }

    // Simple (non-robust) statistics: mean and standard deviation of time-means for sigma-clip
    float mean_means, std_means;
    findMeanStd(time_means, nsamp, &mean_means, &std_means);

    float thr_mean = mean_means + nsigma_mean * std_means;

    // Initial flag arrays
    memset(flag_time, 0, (size_t)nsamp);

    for (int t = 0; t < nsamp; ++t) {
        if (time_means[t] > thr_mean) {
            flag_time[t] = 1;
        }
    }

    // Merge adjacent runs shorter than min_run (if min_run>1). We'll perform run-length pass.
    if (min_run > 1) {
        int start = 0;
        while (start < nsamp) {
            while (start < nsamp && !flag_time[start]) start++;
            if (start >= nsamp) break;
            int end = start + 1;
            while (end < nsamp && flag_time[end]) end++;
            int run_len = end - start;
            if (run_len < min_run) { // clear short run
                for (int k = start; k < end; ++k) flag_time[k] = 0;
            }
            start = end;
        }
    }

    // Write vertical mask: for each flagged time index, mark all channels
    int flagged_times = 0;
    for (int t = 0; t < nsamp; ++t) {
        if (!flag_time[t]) continue;
        flagged_times++;
        for (int c = 0; c < nchan; ++c) {
            verticalMask[c * nsamp + t] = true;
        }
    }

    if (flagged_times > 0) {
        printf("Vertical stripe detection: flagged %d time indices (%.2f%%) mean_mean=%.3f std_mean=%.3f thr_mean=%.3f\n",
               flagged_times, (float)flagged_times / nsamp * 100.0f,
               mean_means, std_means,
               thr_mean);
    } else {
        printf("Vertical stripe detection: no time indices flagged mean_mean=%.3f std_mean=%.3f thr_mean=%.3f\n",
               mean_means, std_means,
               thr_mean);
    }
}

/* -------------------------------------------------------------------------
 * Detect vertical "string" interference from duplicated time samples
 * ------------------------------------------------------------------------- */
void detectVerticalRepeatedColumns(
    const float *data,
    int nsamp,
    int nchan,
    const bool *pointMask,
    const bool *horizontalMask,
    bool *verticalMask,
    float abs_epsilon,
    float rel_sigma,
    int min_run,
    int plot,
    float *err_buf,
    unsigned char *flag_time)
{
    if (!data || !verticalMask || nsamp <= 1 || nchan <= 0) return;

    // 忽略其他掩码，按用户要求直接做双指针相邻列比较。
    // 列相等判据：mean(|col(t)-col(s)|) <= max(abs_epsilon, rel_sigma * mean( (|col(t)|+|col(s)|)/2 ))

    int t = 1;
    while (t < nsamp) {
    if (columns_similar_cols(data, nsamp, nchan, t, t - 1, abs_epsilon, rel_sigma)) {
            int anchor = t;        // 后指针固定为较新的这一列
            int start = t - 1;     // 初始起点
            int end = t;           // 初始终点

            // 向右扩展：前指针前进，直到不相似
            int u = end + 1;
            while (u < nsamp && columns_similar_cols(data, nsamp, nchan, u, anchor, abs_epsilon, rel_sigma)) { end = u; u++; }

            // 可选向左扩展：包含更早相同列（避免从中间才触发时丢失左边界）
            int v = start - 1;
            while (v >= 0 && columns_similar_cols(data, nsamp, nchan, v, anchor, abs_epsilon, rel_sigma)) { start = v; v--; }

            // 最小长度过滤（如果需要）。min_run<=1 则不启用过滤
            if (min_run < 1) min_run = 1;
            if ((end - start + 1) >= min_run) {
                for (int k = start; k <= end; ++k) {
                    for (int c = 0; c < nchan; ++c) {
                        verticalMask[(size_t)c * (size_t)nsamp + (size_t)k] = true;
                    }
                }
                printf("Vertical repeat (two-pointer): [%d,%d] len=%d\n", start, end, end - start + 1);
            }
            t = end + 1; // 跳过已处理片段
        } else {
            t++;
        }
    }
}

/* -------------------------------------------------------------------------
 * Simple block RFI detection using connected components
 * ------------------------------------------------------------------------- */
void detectBlockRFI(
    const bool *binaryMask, // input binary mask (e.g., pointMask)
    int nsamp, int nchan,
    bool *blockMask, // output block RFI mask
    int min_area, float min_density, // simple thresholds
    int dilate_radius, int dilate_iterations // extra dilation on a local copy
) {
    memset(blockMask, 0, (size_t)nsamp * (size_t)nchan * sizeof(bool));

    // Make a local working copy so we can apply extra dilation without touching the original mask
    bool *workMask = (bool *)malloc((size_t)nsamp * (size_t)nchan * sizeof(bool));
    if (!workMask) return;
    memcpy(workMask, binaryMask, (size_t)nsamp * (size_t)nchan * sizeof(bool));
    if (dilate_radius > 0 && dilate_iterations > 0) {
        // Note: dilateMask signature is (mask, width=nsamp, height=nchan, radius, iterations)
        dilateMask(workMask, nsamp, nchan, dilate_radius, dilate_iterations);
    }

    int *componentLabels = (int *)malloc((size_t)nsamp * (size_t)nchan * sizeof(int));
    if (!componentLabels) { free(workMask); return; }
    memset(componentLabels, 0, (size_t)nsamp * (size_t)nchan * sizeof(int));
    
    int label = 1;
    // Stack for flood fill (to avoid recursion)
    int *stack = (int *)malloc((size_t)nsamp * (size_t)nchan * sizeof(int));
    if (!stack) { free(componentLabels); free(workMask); return; }
    
    // Directions for 8-connectivity
    int dirs[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    
    for (int c = 0; c < nchan; ++c) {
        for (int t = 0; t < nsamp; ++t) {
            if (workMask[c * nsamp + t] && componentLabels[c * nsamp + t] == 0) {
                // Start new component
                int top = -1;
                stack[++top] = c * nsamp + t;  // push
                componentLabels[c * nsamp + t] = label;
                int pixel_count = 0;
                int min_c = c, max_c = c, min_t = t, max_t = t;
                
                while (top >= 0) {
                    int idx = stack[top--];  // pop
                    int cc = idx / nsamp;
                    int tt = idx % nsamp;
                    pixel_count++;
                    
                    // Update bounds
                    if (cc < min_c) min_c = cc;
                    if (cc > max_c) max_c = cc;
                    if (tt < min_t) min_t = tt;
                    if (tt > max_t) max_t = tt;
                    
                    // Check 8 neighbors
                    for (int d = 0; d < 8; ++d) {
                        int nc = cc + dirs[d][0];
                        int nt = tt + dirs[d][1];
                        if (nc >= 0 && nc < nchan && nt >= 0 && nt < nsamp &&
                            workMask[nc * nsamp + nt] && componentLabels[nc * nsamp + nt] == 0) {
                            componentLabels[nc * nsamp + nt] = label;
                            stack[++top] = nc * nsamp + nt;  // push
                        }
                    }
                }
                
                // Evaluate component
                int width = max_c - min_c + 1;
                int height = max_t - min_t + 1;
                int bb_area = width * height;
                float density = (float)pixel_count / (float)bb_area;
                float wh_ratio = (height > 0) ? ((float)width / (float)height) : 0.0f;
                // New constraint: width/height must be in [1/5, 5]
                const float RATIO_MIN = 0.2f;
                const float RATIO_MAX = 5.0f;
                int ratio_ok = (wh_ratio >= RATIO_MIN && wh_ratio <= RATIO_MAX);
                
                if (pixel_count >= min_area && density >= min_density && ratio_ok) {
                    // Mark bounding box as block RFI
                    for (int bc = min_c; bc <= max_c; ++bc) {
                        for (int bt = min_t; bt <= max_t; ++bt) {
                            blockMask[bc * nsamp + bt] = true;
                        }
                    }
                }
                
                label++;
            }
        }
    }
    
    free(stack);
    free(componentLabels);
    free(workMask);
}

/* -------------------------------------------------------------------------
 * Periodic point RFI detection (subset of pointMask)
 * ------------------------------------------------------------------------- */
void detectPeriodicPointRFI(
    const bool *pointMask,
    int nsamp, int nchan,
    bool *periodicMask,
    int min_period, int max_period,
    int min_pairs,
    float min_align_frac)
{
    if (!pointMask || !periodicMask || nsamp <= 0 || nchan <= 0) return;
    memset(periodicMask, 0, (size_t)nsamp * (size_t)nchan * sizeof(bool));

    if (min_period < 2) min_period = 2; // ignore period=1 (too trivial)
    if (max_period <= min_period) max_period = min_period + 1;
    if (max_period > nsamp - 1) max_period = nsamp - 1;
    if (max_period < min_period) return;
    if (min_pairs < 2) min_pairs = 2;
    if (min_align_frac < 0.0f) min_align_frac = 0.0f;
    if (min_align_frac > 1.0f) min_align_frac = 1.0f;

    int total_periodic_pixels = 0;
    // Per-channel search
    for (int c = 0; c < nchan; ++c) {
        const bool *row = pointMask + (size_t)c * (size_t)nsamp;
        int flagged_count = 0;
        for (int t = 0; t < nsamp; ++t) if (row[t]) flagged_count++;
        if (flagged_count < min_pairs) continue; // not enough points to form periodic structure

        int best_period = 0;
        int best_score_pairs = 0;
        float best_align_frac = 0.0f;

        // Precompute indices of flagged points to accelerate pair scan
        int *flag_times = (int *)malloc((size_t)flagged_count * sizeof(int));
        if (!flag_times) return; // abort all if allocation fails
        int fc = 0;
        for (int t = 0; t < nsamp; ++t) if (row[t]) flag_times[fc++] = t;

        // For each candidate period T, count pairs (t, t+T) where both flagged
        for (int T = min_period; T <= max_period; ++T) {
            int pair_count = 0;
            // Use two-pointer membership test on sorted flag_times (already sorted by construction)
            int i = 0, j = 0;
            while (i < flagged_count && j < flagged_count) {
                int a = flag_times[i];
                int b_target = a + T;
                // Advance j until flag_times[j] >= b_target
                while (j < flagged_count && flag_times[j] < b_target) j++;
                if (j < flagged_count && flag_times[j] == b_target) {
                    pair_count++;
                }
                i++;
            }

            if (pair_count >= min_pairs) {
                float align_frac = (float)pair_count / (float)flagged_count; // fraction of points participating
                if (align_frac >= min_align_frac) {
                    // Prefer higher pair_count; tie-breaker: higher align_frac; then smaller T
                    if (pair_count > best_score_pairs ||
                        (pair_count == best_score_pairs && align_frac > best_align_frac) ||
                        (pair_count == best_score_pairs && fabsf(align_frac - best_align_frac) < 1e-6f && T < best_period)) {
                        best_period = T;
                        best_score_pairs = pair_count;
                        best_align_frac = align_frac;
                    }
                }
            }
        }

        if (best_period > 0 && best_score_pairs >= min_pairs && best_align_frac >= min_align_frac) {
            // Mark periodic points: only those that have partner at +best_period (keep subset to avoid over-marking)
            bool *dst = periodicMask + (size_t)c * (size_t)nsamp;
            int pairs_marked = 0;
            int i = 0, j = 0;
            while (i < flagged_count && j < flagged_count) {
                int a = flag_times[i];
                int b_target = a + best_period;
                while (j < flagged_count && flag_times[j] < b_target) j++;
                if (j < flagged_count && flag_times[j] == b_target) {
                    dst[a] = true;
                    dst[b_target] = true; // include partner pixel
                    pairs_marked++;
                }
                i++;
            }
            total_periodic_pixels += pairs_marked * 2; // approximate
        }
        free(flag_times);
    }

    if (total_periodic_pixels > 0) {
        printf("Periodic point RFI: marked ~%d pixels (subset of pointMask)\n", total_periodic_pixels);
    } else {
        printf("Periodic point RFI: no periodic structures detected (criteria unmet)\n");
    }
}