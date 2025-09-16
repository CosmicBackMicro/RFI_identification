#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "plot.h"
#include "findStats.h"
#include "ReadFASTData.h"
#include "omp.h"

// int partition(float *arr, int left, int right) {
//     float pivot = arr[(left + right) / 2];  // Select pivot
//     int i = left - 1, j = right + 1;

//     while (1) {
//         // Scan from left to find the first element >= pivot
//         while (i < right) {
//             i++;
//             if (arr[i] >= pivot) break;
//         }
//         // Scan from right to find the first element <= pivot
//         while (j > left) {
//             j--;
//             if (arr[j] <= pivot) break;
//         }
//         // Check if pointers meet or cross
//         if (i >= j) return j;
//         // Swap elements
//         float temp = arr[i];
//         arr[i] = arr[j];
//         arr[j] = temp;
//     }
// }

// float quickselect(float *arr, int left, int right, int k)
// {
//     if (left == right)
//         return arr[left];

//     int p = partition(arr, left, right);

//     if (k <= p)
//     {
//         return quickselect(arr, left, p, k);
//     }
//     else
//     {
//         return quickselect(arr, p + 1, right, k);
//     }
// }

// void findMedian(float *arr, int size, float *median)
// {
//     float *tmp = malloc(size * sizeof(float));
//     memcpy(tmp, arr, size * sizeof(float));
    
// #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         int threads = omp_get_num_threads();
//         int chunk = (size + threads - 1) / threads;
//         int left = tid * chunk;
//         int right = (left + chunk > size) ? size - 1 : left + chunk - 1;
//         quickselect(tmp, left, right, left + (right - left) / 2);
//     }
//     *median = quickselect(tmp, 0, size - 1, size / 2);
//     free(tmp);
// }

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float median(float *arr, int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low)        /* One element only */
            return arr[median];

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

float mad(float *arr, int n) {
    float *temp_arr = (float *)malloc(n * sizeof(float));
    memcpy(temp_arr, arr, n * sizeof(float));
    float median_value = median(temp_arr, n);
    free(temp_arr);

    float *abs_deviations = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        abs_deviations[i] = fabsf(arr[i] - median_value);
    }

    float *temp_deviations = (float *)malloc(n * sizeof(float));
    memcpy(temp_deviations, abs_deviations, n * sizeof(float));
    float mad_value = median(temp_deviations, n);
    mad_value *= 1.4826f;

    free(abs_deviations);
    free(temp_deviations);
    return mad_value;
}

float stdFromMedian(float *arr, int n) {
    if (n <= 1) {
        return 0.0f;
    }
    
    // Calculate median without modifying original array
    float *temp_arr = (float *)malloc(n * sizeof(float));
    memcpy(temp_arr, arr, n * sizeof(float));
    float median_value = median(temp_arr, n);
    free(temp_arr);
    
    // Calculate standard deviation from median
    float sum_squared_dev = 0.0f;
    for (int i = 0; i < n; i++) {
        float deviation = arr[i] - median_value;
        sum_squared_dev += deviation * deviation;
    }
    
    float mean_squared_dev = sum_squared_dev / n;
    return sqrtf(mean_squared_dev);
}

#undef ELEM_SWAP

void findMeanStd(float *arr, int size, float *mean, float *std)
{

    int i;
    /* === First pass for mean === */
    float sum = 0.0f;
// #pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < size; i++)
    {
        sum += arr[i];
    }
    float calculated_mean = sum / size;
    if (mean != NULL)
    {
        *mean = calculated_mean;
    }

    /* === Second pass for stddev === */
    if (std == NULL)
        return;
    float variance = 0.0f;
// #pragma omp parallel for reduction(+ : variance)
    for (i = 0; i < size; i++)
    {
        float diff = arr[i] - calculated_mean;
        variance += diff * diff;
    }
    variance /= size;

    *std = sqrtf(variance);
}

void findMinMax(float *arr, int size, float *min, float *max)
{
    float local_min = arr[0], local_max = arr[0];
// #pragma omp parallel
    {
        float thread_min = arr[0], thread_max = arr[0];
// #pragma omp for nowait
        for (int i = 1; i < size; i++)
        {
            if (arr[i] < thread_min)
                thread_min = arr[i];
            if (arr[i] > thread_max)
                thread_max = arr[i];
        }
// #pragma omp critical
        {
            if (thread_min < local_min)
                local_min = thread_min;
            if (thread_max > local_max)
                local_max = thread_max;
        }
    }
    *min = local_min;
    *max = local_max;
}

void calc8bitHist(float *data, int size)
{
    const int numBins = 256;
    int *hist = (int *)calloc(numBins, sizeof(int));

    /* === Accumulate histogram === */
    for (int i = 0; i < size; i++)
    {
        int val = (int)round(data[i]);
        if (val < 0)
            val = 0;
        if (val > 255)
            val = 255;
        hist[val]++;
    }

    /* === Find X bounds === */
    float mean = 0.0, sigma = 0.0;
    findMeanStd(data, size, &mean, &sigma);
    float lowerBound = floor(mean - 5 * sigma);
    float upperBound = ceil(mean + 5 * sigma);
    lowerBound = (lowerBound < 0) ? 0 : lowerBound;
    upperBound = (upperBound > 255) ? 255 : upperBound;
    if (upperBound <= lowerBound)
    {
        lowerBound = 0;
        upperBound = 255;
    }

    /* === Find Y bounds === */
    int maxFreq = 0;
    for (int i = lowerBound; i <= upperBound; i++)
    {
        if (hist[i] > maxFreq)
            maxFreq = hist[i];
    }

    plot8bitHist(hist, lowerBound, upperBound, mean, sigma, maxFreq);
    free(hist);
}