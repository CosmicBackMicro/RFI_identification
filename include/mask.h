#pragma once
#include "identification.h" // for IdentNSigmaMasks

// Mask-only utility APIs
// - writeIndexMaskPNG: save a 2D mask (nchan x nsamp) as grayscale PNG
// - mergeMask2D: merge multiple masks into a single result (writes i+1 for mask i hits)
// - expandChannelMask: expand 1D channel flags to 2D mask (whole channel to 1)
// - logicalOR: element-wise OR from mask into globalMask

void writeIndexMaskPNG(int *mask, int nsamp, int nchan, char *filename);

void mergeMask2D(int *masks[], int nmasks, int nsamp, int nchan, int *result);

void expandChannelMask(const int *channelFlagged, int *mask2D, int nsamp, int nchan);

void logicalOR(int *globalMask, const int *mask, int nsamp, int nchan);

// Convenience API: write all masks in IdentNSigmaMasks to PNGs with consistent naming
// Filenames will be written to: `${datasetPath}mask_<name>_<index>.png`
void writeAllMasksPNG(const IdentNSigmaMasks *masks, int nsamp, int nchan,
					  const char *datasetPath, int index);

// Allocation helpers for IdentNSigmaMasks
void allocIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan);
void freeIdentNSigmaMasks(IdentNSigmaMasks *m);
