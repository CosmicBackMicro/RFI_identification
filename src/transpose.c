#include <omp.h>
#include <fftw3.h>

#include "transpose.h"

/// @brief Transpose a block of data from frequency contiguous to time contiguous.
/// Theoretically it can be used vice versa.
/// @param array Array to be transposed.
/// @param nsamp Number of time samples in `array`.
/// @param nchan Number of frequency channels in `array`.
/// @param arrayT Transposed array.
// void transpose(const float *array, int nsamp, int nchan, float *arrayT)
// {
//     int i, j;
// #pragma omp parallel for private(i, j)
//     for (i = 0; i < nsamp; i++)
//     {
// #pragma omp simd
//         for (j = 0; j < nchan; j++)
//         {
//             arrayT[j * nsamp + i] = array[i * nchan + j];
//         }
//     }
// }

fftwf_plan plan_transpose_f(int rows, int cols, float *in, float *out)
{
    const unsigned flags = FFTW_ESTIMATE;
    fftwf_iodim howmany_dims[2];
    howmany_dims[0].n = rows;
    howmany_dims[0].is = cols;
    howmany_dims[0].os = 1;
    howmany_dims[1].n = cols;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = rows;
    return fftwf_plan_guru_r2r(
        0, // rank
        NULL, // dims
        2, // howmany_rank 
        howmany_dims,
        in, out, 
        NULL, // kind
        flags);
}
void transpose(float *array, int nsamp, int nchan, float *arrayT)
{
    static fftwf_plan tplan;
    static int last_nsamp = -1, last_nchan = -1;
    
    // Create plan only if dimensions changed or plan doesn't exist
    if (tplan == NULL || nsamp != last_nsamp || nchan != last_nchan) {
        if (tplan != NULL) {
            fftwf_destroy_plan(tplan);
        }
        tplan = plan_transpose_f(nsamp, nchan, array, arrayT);
        last_nsamp = nsamp;
        last_nchan = nchan;
    }
    
    fftwf_execute_r2r(tplan, array, arrayT);
}

void transpose_int(const int *array, int nsamp, int nchan, int *arrayT)
{
    int i, j;
    #pragma omp parallel for collapse(2)
    for (i = 0; i < nsamp; i++)
    {
        for (j = 0; j < nchan; j++)
        {
            arrayT[j * nsamp + i] = array[i * nchan + j];
        }
    }
}