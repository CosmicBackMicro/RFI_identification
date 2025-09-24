#ifndef ALG_IQRM_H
#define ALG_IQRM_H

#include <stdlib.h>

/* Main IQRM function */
int iqrm_mask(float *x, int n, int radius, float threshold,
              int *ignore, int ignore_count,
              unsigned char *mask_out);

#endif /* ALG_IQRM_H */