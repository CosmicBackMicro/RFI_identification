 #pragma once
 #include "identification.h" // for IdentNSigmaMasks

// 清空所有掩码数组
void clearIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan);

// Mask-only utility APIs
// - writeIndexMaskPNG: save a binary 2D mask (nchan x nsamp, values 0/1) as grayscale PNG
// - writeClassIndexMaskPNG: save a class-index 2D mask (values 0..255) as grayscale PNG
// - mergeMask2D: merge multiple masks into a single result (writes i+1 for mask i hits)
// - expandChannelMask: expand 1D channel flags to 2D mask (whole channel to 1)
// - logicalOR: element-wise OR from mask into globalMask

void writeIndexMaskPNG(const bool *mask, int nsamp, int nchan, char *filename);
void writeClassIndexMaskPNG(const int *indexMask, int nsamp, int nchan, char *filename);

void mergeMask2D(int *masks[], int nmasks, int nsamp, int nchan, int *result);

void expandChannelMask(const int *channelFlagged, bool *mask2D, int nsamp, int nchan);

void logicalOR(bool *globalMask, const bool *mask, int nsamp, int nchan);

// Convenience API: write all masks in IdentNSigmaMasks to PNGs with consistent naming
// Filenames will be written to: `${datasetPath}mask_<name>_<index>.png`
void writeAllMasksPNG(const IdentNSigmaMasks *masks, int nsamp, int nchan,
					  const char *datasetPath, int index, int merge);

// Smooth outChannel mask by removing isolated flagged channels
void smoothOutChanMask(int *channelFlagged, int nchan, int N);

// Allocation helpers for IdentNSigmaMasks
void allocIdentNSigmaMasks(IdentNSigmaMasks *m, int nsamp, int nchan);
void freeIdentNSigmaMasks(IdentNSigmaMasks *m);
