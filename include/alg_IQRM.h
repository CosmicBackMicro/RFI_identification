#ifndef ALG_IQRM_H
#define ALG_IQRM_H

#include <stdlib.h>

/* Main IQRM function */
int alg_iqrm_mask(float *x, int n, int radius, float threshold,
                  int *ignore, int ignore_count,
                  unsigned char *mask_out);

/*
 * Channel-level Std IQR detection
 * Inputs:
 *  - data_by_chan: transposed data laid out as nchan blocks of length nsamp (i.e., data[ch*nsamp + t])
 *  - nsamp: number of time samples per channel
 *  - nchan: number of channels
 *  - tukey_q: Tukey IQR multiplier (e.g., 1.5 for classic, 2.0 for conservative)
 * Output:
 *  - chan_mask_out: length nchan, 0/1 per channel; 1 means flagged as outlier
 * Return: number of flagged channels (>=0), or negative on error
 */
int iqrmChanStd(const float *data_by_chan, int nsamp, int nchan,
                             float tukey_q, unsigned char *chan_mask_out);

/* Expand 1D channel mask to 2D mask (channel-major layout). mask2d must be allocated by caller. */
void expandChanMask(const unsigned char *chan_mask, int nchan, int nsamp, int *mask2d);

/*
 * Combined Channel IQR + IQRM detection. Returns number of flagged channels.
 * If iqrm_radius <= 0, a default radius is chosen based on nchan (approx nchan/10 with floor of 4).
 * If mask2d_out is non-NULL, an int* buffer will be allocated and returned via mask2d_out; caller must free it with free().
 * If used_radius_out is non-NULL, it will receive the actual radius used.
 */
int IQRM(const float *data_by_chan, int nsamp, int nchan,
                            float tukey_q, float iqrm_threshold,
                            int **mask2d_out);

#endif /* ALG_IQRM_H */