#pragma once

// void findMedian(float *arr, int size, float *median);
void findMeanStd(float *arr, int size, float *mean, float *std);
void findMinMax(float *arr, int size, float *min, float *max);
void calc8bitHist(float *data, int size);
float median(float *arr, int n);
float mad(float *arr, int n);
float stdFromMedian(float *arr, int n);
float stdFromKnownMedian(const float *arr, int n, float median_value);
void findMedianStd(float *arr, int n, float *median, float *std);
float percentile(const float *arr, int m, float p);
int cmp_float(const void *a, const void *b);