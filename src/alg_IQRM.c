#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "findStats.h"

#ifndef IQRM_GEOMETRIC_FACTOR
#define IQRM_GEOMETRIC_FACTOR 1.5
#endif

int calcQuartiles(const float *x, int n, float *q1, float *med, float *q3) {
    float *buf = (float*)malloc(n * sizeof(float));
    int m = 0;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (v == v) buf[m++] = v; /* v==v is false only for NaN */
    }
    if (m == 0) {free(buf); return 0;}
    qsort(buf, m, sizeof(float), cmp_float);
    *q1  = percentile(buf, m, 25.0f);
    *med = percentile(buf, m, 50.0f);
    *q3  = percentile(buf, m, 75.0f);
    free(buf);
    return 1;
}

/* Compute per-channel std on channel-major data: data[ch*nsamp + t] */
static void compute_channel_std(const float *data_by_chan, int nsamp, int nchan, float *stds_out) {
    for (int ch = 0; ch < nchan; ++ch) {
        const float *col = data_by_chan + (size_t)ch * (size_t)nsamp;
        double sum = 0.0, sum2 = 0.0; int cnt = 0;
        for (int t = 0; t < nsamp; ++t) {
            float v = col[t];
            sum += v; sum2 += (double)v * (double)v; cnt++;
        }
        float s = 0.0f;
        if (cnt > 1) {
            double mean = sum / cnt;
            double var = (sum2 - 2.0 * mean * sum + mean * mean * cnt) / cnt;
            if (var < 0.0) var = 0.0;
            s = (float)sqrt(var);
        }
        stds_out[ch] = s;
    }
}

/* ---------------------- Outlier mask (Tukey) ------------------------- */
void outlier_mask_diff(const float *d, int n, float threshold, unsigned char *m_out) {
    float q1, med, q3;
    if (!calcQuartiles(d, n, &q1, &med, &q3)) {
        /* No valid data => no outliers */
        for (int i = 0; i < n; ++i) m_out[i] = 0;
        return;
    }
    float std = (q3 - q1) / 1.349f; /* Gaussian std estimate from IQR */
    if (!(std > 0.0f)) std = 1e-12f;   /* avoid division by zero */
    for (int i = 0; i < n; ++i) {
        float v = d[i];
        if (v != v) { m_out[i] = 0; continue; }
        m_out[i] = (unsigned char)((v - med) > threshold * std);
    }
}

/* ---------------------- Lagged difference ---------------------------- */
void lagged_diff(const float *input, int num_elements, int lag, float *differences) {
    /* differences[i] = input[i] - input[i - lag] with boundary extension:
       if i - lag < 0 use input[0]; if i - lag >= num_elements use input[num_elements-1]
     */
    for (int current_index = 0; current_index < num_elements; ++current_index) {
        int lagged_index = current_index - lag;
        if (lagged_index < 0) lagged_index = 0;
        else if (lagged_index >= num_elements) lagged_index = num_elements - 1;
        differences[current_index] = input[current_index] - input[lagged_index];
    }
}

/* ---------------------- Main IQRM function --------------------------- */
int alg_iqrm_mask(float *x, int n, int radius, float threshold,
              int *ignore, int ignore_count,
              unsigned char *mask_out) {
    /* Parameter checks retained */
    if (!x || !mask_out || n <= 0) return 1;
    if (radius <= 0) return 2;
    if (!(threshold > 0.0)) return 3;

    for (int i = 0; i < n; ++i) mask_out[i] = 0;

    size_t matrix_bytes = (size_t)n * (size_t)n;
    unsigned char *votes_matrix = (unsigned char*)calloc(matrix_bytes, 1);
    int *votes_cast_count = (int*)calloc((size_t)n, sizeof(int));
    int *votes_received_count = (int*)calloc((size_t)n, sizeof(int));
    float *diff = (float*)malloc((size_t)n * sizeof(float));
    unsigned char *m = (unsigned char*)malloc((size_t)n * sizeof(unsigned char));

    /* Geometric lag generation (mirrors Python genlags) */
    int lag = 1;
    while (lag <= radius) {
        for (int sign_iter = 0; sign_iter < 2; ++sign_iter) {
            int signed_lag = (sign_iter == 0) ? lag : -lag;
            lagged_diff(x, n, signed_lag, diff);
            outlier_mask_diff(diff, n, threshold, m);
            for (int i = 0; i < n; ++i) {
                if (!m[i]) continue;
                int caster = i - signed_lag; /* j = i - lag */
                if (caster < 0) caster = 0; else if (caster >= n) caster = n - 1;
                /* Add directed edge caster -> i if not already present */
                size_t idx = (size_t)caster * (size_t)n + (size_t)i;
                if (!votes_matrix[idx]) {
                    votes_matrix[idx] = 1;
                    votes_cast_count[caster] += 1;
                    votes_received_count[i] += 1;
                }
            }
        }
        int next_lag = (int)(IQRM_GEOMETRIC_FACTOR * lag);
        if (next_lag <= lag) next_lag = lag + 1;
        lag = next_lag;
    }

    /* Final masking logic */
    for (int i = 0; i < n; ++i) {
        int received = votes_received_count[i];
        if (received == 0) continue; /* cannot be flagged */
        /* iterate over all possible casters j; break early if condition met */
        for (int j = 0; j < n; ++j) {
            size_t idx = (size_t)j * (size_t)n + (size_t)i;
            if (!votes_matrix[idx]) continue; /* no edge j->i */
            if (votes_cast_count[j] < received) { /* condition */
                mask_out[i] = 1;
                break;
            }
        }
    }

    /* Force ignored channels to masked */
    for (int k = 0; k < ignore_count; ++k) {
        int idx = ignore[k];
        if (idx >= 0 && idx < n) mask_out[idx] = 1;
    }

    /* Cleanup */
    free(votes_matrix); 
    free(votes_cast_count); 
    free(votes_received_count);
    free(diff); 
    free(m);
    return 0;
}

/* ---------------------- Channel Std IQR detection ------------------- */
int iqrmChanStd(const float *data_by_chan, int nsamp, int nchan,
                         float tukey_q, unsigned char *chan_mask_out) {
    if (!(tukey_q > 0.0f)) tukey_q = 1.5f;
    float *stds = (float*)malloc((size_t)nchan * sizeof(float));
    float *sorted = (float*)malloc((size_t)nchan * sizeof(float));

    // 1) compute per-channel std
    compute_channel_std(data_by_chan, nsamp, nchan, stds);

    // 2) quartiles across channels
    memcpy(sorted, stds, (size_t)nchan * sizeof(float));
    qsort(sorted, nchan, sizeof(float), cmp_float);
    float q1 = percentile(sorted, nchan, 25.0f);
    float q3 = percentile(sorted, nchan, 75.0f);
    float iqr = q3 - q1;
    float vmin, vmax;
    if (!(iqr > 0.0f)) { vmin = -FLT_MAX; vmax = FLT_MAX; }
    else { vmin = q1 - tukey_q * iqr; vmax = q3 + tukey_q * iqr; }

    // 3) flag
    int flagged = 0;
    for (int ch = 0; ch < nchan; ++ch) {
        unsigned char f = (stds[ch] < vmin || stds[ch] > vmax) ? 1 : 0;
        chan_mask_out[ch] = f; flagged += f;
    }

    free(stds); 
    free(sorted);
    return flagged;
}

/* ---------------------- Utilities ------------------- */
void expandChanMask(const unsigned char *chan_mask, int nchan, int nsamp, int *mask2d) {
    memset(mask2d, 0, (size_t)nchan * (size_t)nsamp * sizeof(int));
    for (int ch = 0; ch < nchan; ++ch) {
        if (!chan_mask[ch]) continue;
        int base = ch * nsamp; // Base index for 2D mask
        for (int t = 0; t < nsamp; ++t) mask2d[base + t] = 1;
    }
}

/* ---------------------- Combined: Channel IQR + IQRM ------------------- */
int IQRM(const float *data_by_chan, int nsamp, int nchan,
                                 float tukey_q, float iqrm_threshold,
                                 int **mask2d_out) {
    if (!(tukey_q > 0.0f)) tukey_q = 1.5f;
    if (!(iqrm_threshold > 0.0f)) iqrm_threshold = 3.0f;

    // Compute per-channel std once
    float *chan_std = (float*)malloc((size_t)nchan * sizeof(float));
    compute_channel_std(data_by_chan, nsamp, nchan, chan_std);

    // 1) Channel Std IQR on precomputed std
    unsigned char *mask_std = (unsigned char*)calloc((size_t)nchan, sizeof(unsigned char));
    float *sorted = (float*)malloc((size_t)nchan * sizeof(float)); // Sorted array for quartiles
    memcpy(sorted, chan_std, (size_t)nchan * sizeof(float));
    qsort(sorted, nchan, sizeof(float), cmp_float);
    float q1 = percentile(sorted, nchan, 25.0f);
    float q3 = percentile(sorted, nchan, 75.0f);
    float iqr = q3 - q1;
    float vmin, vmax;
    vmin = q1 - tukey_q * iqr; 
    vmax = q3 + tukey_q * iqr;
    for (int ch = 0; ch < nchan; ++ch) {
        mask_std[ch] = (chan_std[ch] < vmin || chan_std[ch] > vmax) ? 1 : 0;
    }

    // 2) IQRM with channel std as feature
    float *chan_feature = chan_std; /* reuse */
    int iqrm_radius = nchan / 10;  // Fixed radius as 1/10 of nchan
    if (iqrm_radius < 1) iqrm_radius = 1;  // Ensure minimum radius of 1
    unsigned char *mask_iqrm = (unsigned char*)calloc((size_t)nchan, sizeof(unsigned char));
    alg_iqrm_mask(chan_feature, nchan, iqrm_radius, iqrm_threshold, NULL, 0, mask_iqrm);

    // 3) Merge OR
    unsigned char *mask_1d = (unsigned char*)calloc((size_t)nchan, sizeof(unsigned char));
    int flagged = 0;
    for (int ch = 0; ch < nchan; ++ch) { mask_1d[ch] = (mask_std[ch] || mask_iqrm[ch]) ? 1 : 0; flagged += mask_1d[ch]; }

    // 4) Expand to 2D (allocate and return)
    int *mask2d = NULL;
    if (mask2d_out) {
        mask2d = (int*)malloc((size_t)nchan * (size_t)nsamp * sizeof(int));
        expandChanMask(mask_1d, nchan, nsamp, mask2d);

        // Add pixel-level IQRM for unmarked channels
        for (int ch = 0; ch < nchan; ++ch) {
            if (mask_1d[ch]) continue; // channel already masked, all pixels 1
            const float *time_series = data_by_chan + ch * (size_t)nsamp;
            unsigned char *pixel_mask = (unsigned char*)malloc(nsamp * sizeof(unsigned char));
            int pixel_radius = nsamp / 10;
            if (pixel_radius < 1) pixel_radius = 1;
            alg_iqrm_mask(time_series, nsamp, pixel_radius, iqrm_threshold, NULL, 0, pixel_mask);
            for (int t = 0; t < nsamp; ++t) {
                mask2d[ch * nsamp + t] = pixel_mask[t];
            }
            free(pixel_mask);
        }

        *mask2d_out = mask2d;
    }

    free(mask_1d);
    free(mask_iqrm);
    free(sorted);
    free(mask_std);
    free(chan_std);
    return flagged;
}
