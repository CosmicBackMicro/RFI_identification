#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Comparison for qsort
int compare(const void* a, const void* b) {
    return (*(float*)a > *(float*)b) - (*(float*)a < *(float*)b);
}

// Calculate percentile
float percentile(float* arr, int n, float p) {
    float* sorted = (float*)malloc(n * sizeof(float));
    memcpy(sorted, arr, n * sizeof(float));
    qsort(sorted, n, sizeof(float), compare);
    int idx = (int)(p * (n - 1));
    float result = sorted[idx];
    free(sorted);
    return result;
}

// CLFD (formerly profile_mask) for 2D time-frequency array
// Input: data (nsamp x nchan, column-major), mask (pre-allocated, same shape)
// Output: mask written in-place
void CLFD(float* data, int nsamp, int nchan, float q, int* zap_freqs, int zap_len, int* mask) {
    int* zap_mask = (int*)calloc(nchan, sizeof(int));
    for (int k = 0; k < zap_len; k++) {
        if (zap_freqs[k] >= 0 && zap_freqs[k] < nchan) {
            zap_mask[zap_freqs[k]] = 1;
        }
    }

    float* chan_features = (float*)malloc(nsamp * sizeof(float));

    for (int j = 0; j < nchan; j++) {
        if (zap_mask[j]) {
            for (int i = 0; i < nsamp; i++) {
                mask[j * nsamp + i] = 1;
            }
            continue;
        }

        for (int i = 0; i < nsamp; i++) {
            chan_features[i] = data[j * nsamp + i];
        }

        float q1 = percentile(chan_features, nsamp, 0.25f);
        float q3 = percentile(chan_features, nsamp, 0.75f);
        float iqr_val = q3 - q1;
        float min_val = q1 - q * iqr_val;
        float max_val = q3 + q * iqr_val;

        for (int i = 0; i < nsamp; i++) {
            mask[j * nsamp + i] = (chan_features[i] < min_val || chan_features[i] > max_val) ? 1 : 0;
        }
    }
    free(chan_features);
    free(zap_mask);
}

// Spike mask for 2D time-frequency array
// Input: data (nsamp x nchan, column-major), mask/replacement (pre-allocated)
// Output: mask and replacement written in-place
void spike_mask(float* data, int nsamp, int nchan, float q, int* zap_freqs, int zap_len, int* mask, float* replacement) {
    int* zap_mask = (int*)calloc(nchan, sizeof(int));
    for (int k = 0; k < zap_len; k++) {
        if (zap_freqs[k] >= 0 && zap_freqs[k] < nchan) {
            zap_mask[zap_freqs[k]] = 1;
        }
    }

    int valid_channels = 0;
    for (int j = 0; j < nchan; j++) {
        if (!zap_mask[j]) valid_channels++;
    }

    // 计算 baseline：每个通道的中位数（沿时间轴）
    float* baselines = (float*)malloc(nchan * sizeof(float));
    for (int j = 0; j < nchan; j++) {
        float* chan_data = (float*)malloc(nsamp * sizeof(float));
        for (int i = 0; i < nsamp; i++) {
            chan_data[i] = data[j * nsamp + i];
        }
        baselines[j] = percentile(chan_data, nsamp, 0.5f);  // 中位数
        free(chan_data);
    }

    // 计算 subtracted
    float* subtracted = (float*)malloc(nsamp * nchan * sizeof(float));
    for (int j = 0; j < nchan; j++) {
        for (int i = 0; i < nsamp; i++) {
            subtracted[j * nsamp + i] = data[j * nsamp + i] - baselines[j];
        }
    }

    float* q1_arr = (float*)malloc(nchan * sizeof(float));
    float* med_arr = (float*)malloc(nchan * sizeof(float));
    float* q3_arr = (float*)malloc(nchan * sizeof(float));
    float* bin_data = (float*)malloc(nsamp * sizeof(float));
    for (int j = 0; j < nchan; j++) {
        for (int i = 0; i < nsamp; i++) {
            bin_data[i] = subtracted[j * nsamp + i];
        }
        q1_arr[j] = percentile(bin_data, nsamp, 0.25f);
        med_arr[j] = percentile(bin_data, nsamp, 0.5f);
        q3_arr[j] = percentile(bin_data, nsamp, 0.75f);
    }
    free(bin_data);

    float* iqr_arr = (float*)malloc(nchan * sizeof(float));
    float* vmin_arr = (float*)malloc(nchan * sizeof(float));
    float* vmax_arr = (float*)malloc(nchan * sizeof(float));
    for (int j = 0; j < nchan; j++) {
        iqr_arr[j] = q3_arr[j] - q1_arr[j];
        vmin_arr[j] = q1_arr[j] - q * iqr_arr[j];
        vmax_arr[j] = q3_arr[j] + q * iqr_arr[j];
    }

    for (int j = 0; j < nchan; j++) {
        for (int i = 0; i < nsamp; i++) {
            mask[j * nsamp + i] = (subtracted[j * nsamp + i] < vmin_arr[j] || subtracted[j * nsamp + i] > vmax_arr[j]) ? 1 : 0;
        }
    }

    for (int j = 0; j < nchan; j++) {
        for (int i = 0; i < nsamp; i++) {
            replacement[j * nsamp + i] = baselines[j] + med_arr[j] / valid_channels;
        }
    }

    free(baselines);
    free(subtracted);
    free(q1_arr);
    free(med_arr);
    free(q3_arr);
    free(iqr_arr);
    free(vmin_arr);
    free(vmax_arr);
    free(zap_mask);
}