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
static inline void cpu_transpose_fallback(const float *array, int nsamp, int nchan, float *arrayT) {
    for (int i = 0; i < nsamp; ++i) {
        for (int j = 0; j < nchan; ++j) {
            arrayT[j * nsamp + i] = array[i * nchan + j];
        }
    }
}

void transpose(float *array, int nsamp, int nchan, float *arrayT)
{
    // Thread-local plan cache with up to two shapes (covers our typical (nsamp,nchan) and (nchan,nsamp))
    static _Thread_local fftwf_plan tplan1 = NULL, tplan2 = NULL;
    static _Thread_local int p1_ns = -1, p1_nc = -1;
    static _Thread_local int p2_ns = -1, p2_nc = -1;

    fftwf_plan plan = NULL;
    if (tplan1 && p1_ns == nsamp && p1_nc == nchan) {
        plan = tplan1;
    } else if (tplan2 && p2_ns == nsamp && p2_nc == nchan) {
        plan = tplan2;
    } else {
        // Create a new plan in an empty slot (do not destroy existing plans to avoid races)
        if (!tplan1) {
            tplan1 = plan_transpose_f(nsamp, nchan, array, arrayT);
            p1_ns = nsamp; p1_nc = nchan;
            plan = tplan1;
        } else if (!tplan2) {
            tplan2 = plan_transpose_f(nsamp, nchan, array, arrayT);
            p2_ns = nsamp; p2_nc = nchan;
            plan = tplan2;
        } else {
            // Fallback safely without modifying existing plans
            cpu_transpose_fallback(array, nsamp, nchan, arrayT);
            return;
        }
    }

    fftwf_execute_r2r(plan, array, arrayT);
}

void transpose_int(const int *array, int nsamp, int nchan, int *arrayT)
{
    int i, j;
    // #pragma omp parallel for collapse(2)
    for (i = 0; i < nsamp; i++)
    {
        for (j = 0; j < nchan; j++)
        {
            arrayT[j * nsamp + i] = array[i * nchan + j];
        }
    }
}