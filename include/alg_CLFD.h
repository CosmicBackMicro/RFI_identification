#ifndef ALG_CLFD_H
#define ALG_CLFD_H

#include <stdlib.h>

/* CLFD (formerly profile_mask): generate channel-time mask for 2D time-frequency array */
void CLFD(float* data, int nsamp, int nchan, float q, int* zap_freqs, int zap_len, int* mask);

/* Spike mask for 2D time-frequency array */
void spike_mask(float* data, int nsamp, int nchan, float q, int* zap_freqs, int zap_len, int* mask, float* replacement);

#endif /* ALG_CLFD_H */