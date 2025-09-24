#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef IQRM_GEOMETRIC_FACTOR
#define IQRM_GEOMETRIC_FACTOR 1.5
#endif

int cmp_float(const void *a, const void *b) {
    float da = *(const float*)a;
    float db = *(const float*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Linear interpolation percentile on sorted array arr of length m (m > 0) */
float percentile_linear(const float *arr, int m, float p) {
    if (m <= 0) return NAN;
    if (p <= 0.0) return arr[0];
    if (p >= 100.0) return arr[m - 1];
    float rank = p / 100.0f * (m - 1);
    int lo = (int)floor(rank);
    int hi = (int)ceil(rank);
    float w = rank - lo;
    if (hi == lo) return arr[lo];
    return arr[lo] + (arr[hi] - arr[lo]) * w;
}

int compute_quartiles(const float *x, int n, float *q1, float *med, float *q3) {
    float *buf = (float*)malloc(n * sizeof(float));
    int m = 0;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (!isnan(v)) buf[m++] = v;
    }
    if (m == 0) {free(buf); return 0;}
    qsort(buf, m, sizeof(float), cmp_float);
    *q1  = percentile_linear(buf, m, 25.0f);
    *med = percentile_linear(buf, m, 50.0f);
    *q3  = percentile_linear(buf, m, 75.0f);
    free(buf);
    return 1;
}

/* ---------------------- Outlier mask (Tukey) ------------------------- */
void outlier_mask_diff(const float *d, int n, float threshold, unsigned char *m_out) {
    float q1, med, q3;
    if (!compute_quartiles(d, n, &q1, &med, &q3)) {
        /* No valid data => no outliers */
        for (int i = 0; i < n; ++i) m_out[i] = 0;
        return;
    }
    float std = (q3 - q1) / 1.349f; /* Gaussian std estimate from IQR */
    if (!(std > 0.0f)) std = 1e-12f;   /* avoid division by zero */
    for (int i = 0; i < n; ++i) {
        float v = d[i];
        if (isnan(v)) { m_out[i] = 0; continue; }
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
int iqrm_mask(float *x, int n, int radius, float threshold,
              int *ignore, int ignore_count,
              unsigned char *mask_out) {
    /* 参数检查保留 */
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
