#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ReadFASTData.h"
#include "cpgplot.h"
#include "psrPalett.h"
#include "plot.h"
#include "findStats.h"

#define PALETT_GREY 1
#define PALETT_BLUE 2
#define PALETT_HEAT 3
#define PALETT_GOLD 4
#define PALETT_ALIEN_GLOW 6
#define PALETT_COLD 7
#define PALETT_PLASMA 8
#define PALETT_CUBE_HELIX 9
#define PALETT_VIRIDIS 10

int ClosestFreqIdx(const float *freqArray, int arraySize, float targetFreq)
{
    if (arraySize <= 0 || freqArray == NULL)
    {
        return -1; // Error: invalid input
    }

    int closestIndex = 0;
    float minDiff = fabsf(freqArray[0] - targetFreq);

    for (int i = 1; i < arraySize; i++)
    {
        float currentDiff = fabsf(freqArray[i] - targetFreq);
        if (currentDiff < minDiff)
        {
            minDiff = currentDiff;
            closestIndex = i;
        }
    }
    return closestIndex;
}

void plotDownsampLongTimeAbs(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock)
{
    int nsampPlot = m->nsampBinned * numReads;
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbinBinned;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory ===
    float *dsDataT_deRFI = malloc(sizeof(float) * m->nsampBinned * numReads * m->nchanBinned);
    float *dsTimeProfile = malloc(sizeof(float) * m->nsampBinned * numReads);
    float *dsFreqProfile = malloc(sizeof(float) * m->nchanBinned * numReads);
    int *dsChannelMask = calloc(m->nchanBinned, sizeof(int));

    getProfile(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    palett(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===

    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(0.1, 0.7, 0.1, 0.7);
    float mean, std, nsig = 5.0;
    findMeanStd(dsDataT, nsampPlot * nchanPlot, &mean, &std);

    float tr[6] = {
        startTime + currentBlock * nsampPlot * tbinPlot, tstep,
        0, dsFreqArray[plotStartChan],
        0, fstep};

    cpgswin(tmin, tmax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("ABINST", 0, 0, "ABINPST", 0, 0);

    cpgimag(dsDataT + plotStartChan * nsampPlot,  // Data pointer offset
            nsampPlot,                            // Number of time samples
            plotEndChan - plotStartChan + 1,      // Number of channels
            1, nsampPlot,                         // X range
            1, plotEndChan - plotStartChan + 1,   // Y range
            mean - nsig * std, mean + nsig * std, // Z range
            tr);
    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(0.7, 0.85, 0.1, 0.7);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile, dsFreqArray + plotStartChan);

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Intensity");
    cpgmtxt("T", 1.0, 0.5, 0.5, "SED Curve");
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    cpgwedg("RI", 2.0, 2.5, mean - nsig * std, mean + nsig * std, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(0.1, 0.7, 0.7, 0.9);
    float time[nsampPlot];
    for (int i = 0; i < nsampPlot; i++)
    {
        time[i] = startTime + currentBlock * nsampPlot * tbinPlot + i * tstep;
    }
    float timeProfileMin, timeProfileMax;
    findMinMax(dsTimeProfile, nsampPlot, &timeProfileMin, &timeProfileMax);
    timeProfileMin -= 0.1 * (timeProfileMax - timeProfileMin);
    timeProfileMax += 0.1 * (timeProfileMax - timeProfileMin);
    cpgswin(tmin, tmax, timeProfileMin, timeProfileMax);
    cpgsch(0.7);
    cpgbox("BCMST", 0, 0, "BCNST", 0, 0);
    cpgline(nsampPlot, time, dsTimeProfile); // Use downsampled time array
    cpglab("", "Intensity", "Time (s)");
    char *title_part1 = malloc(256 * sizeof(char));
    char *title_part2 = malloc(256 * sizeof(char));
    sprintf(title_part1, m->filename);
    sprintf(title_part2, "downsamp %d for time, and %d for freq, tbin = %.4e s, channel width = %.4e MHz",
            m->binFactorTime, m->binFactorFreq, m->tbinBinned, m->chan_bwBinned);
    cpgmtxt("T", 4.0, 0.5, 0.5, title_part1);
    cpgmtxt("T", 3.0, 0.5, 0.5, title_part2);
    free(title_part1);
    free(title_part2);
    cpgsch(1.0);

    // === Clean up ===
    free(dsDataT_deRFI);
    free(dsTimeProfile);
    free(dsFreqProfile);
    free(dsChannelMask);
}
void plotDownsampSED(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, float *baseline)
{
    int nsampPlot = m->nsampBinned; // Use binned samples for plotting
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbinBinned;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory ===
    float *dsDataT_deRFI = malloc(sizeof(float) * m->nsampBinned * m->nchanBinned);
    float *dsTimeProfile = malloc(sizeof(float) * m->nsampBinned);
    float *dsFreqProfile = malloc(sizeof(float) * m->nchanBinned);
    int *dsChannelMask = calloc(m->nchanBinned, sizeof(int));

    getProfile(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);
    // getProfileStd(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    palett(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===
    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(0.1, 0.7, 0.1, 0.7);
    float mean, std, nsig = 5.0;
    findMeanStd(dsDataT, nsampPlot * nchanPlot, &mean, &std);

    float tr[6] = {
        startTime + currentBlock * nsampPlot * tbinPlot, tstep,
        0, dsFreqArray[plotStartChan],
        0, fstep};

    cpgswin(tmin, tmax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("ABINST", 0, 0, "ABINPST", 0, 0);

    cpgimag(dsDataT + plotStartChan * nsampPlot,  // Data pointer offset
            nsampPlot,                            // Number of time samples
            plotEndChan - plotStartChan + 1,      // Number of channels
            1, nsampPlot,                         // X range
            1, plotEndChan - plotStartChan + 1,   // Y range
            mean - nsig * std, mean + nsig * std, // Z range
            tr);
    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(0.7, 0.85, 0.1, 0.7);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    // cpgswin(freqProfileMin, freqProfileMax, fmin, fmax);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile, dsFreqArray + plotStartChan);
    // cpgline(plotEndChan - plotStartChan + 1, baseline, dsFreqArray + plotStartChan);
    // cpgline(nchanPlot, dsFreqProfile, dsFreqArray + plotStartChan);

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Intensity");
    cpgmtxt("T", 1.0, 0.5, 0.5, "SED Curve");
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    cpgwedg("RI", 2.0, 2.5, mean - nsig * std, mean + nsig * std, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(0.1, 0.7, 0.7, 0.9);
    // Calculate time axis
    float time[nsampPlot];
    for (int i = 0; i < nsampPlot; i++)
    {
        time[i] = startTime + currentBlock * nsampPlot * tbinPlot + i * tstep;
    }
    float timeProfileMin, timeProfileMax;
    findMinMax(dsTimeProfile, nsampPlot, &timeProfileMin, &timeProfileMax);
    timeProfileMin -= 0.1 * (timeProfileMax - timeProfileMin);
    timeProfileMax += 0.1 * (timeProfileMax - timeProfileMin);
    cpgswin(tmin, tmax, timeProfileMin, timeProfileMax);
    cpgsch(0.7);
    cpgbox("BCMST", 0, 0, "BCNST", 0, 0);
    cpgline(nsampPlot, time, dsTimeProfile); // Use downsampled time array
    cpglab("", "Intensity", "Time (s)");
    char *title_part1 = malloc(256 * sizeof(char));
    char *title_part2 = malloc(256 * sizeof(char));
    sprintf(title_part1, m->filename);
    sprintf(title_part2, "downsamp %d for time, and %d for freq, tbin = %.4e s, channel width = %.4e MHz",
            m->binFactorTime, m->binFactorFreq, m->tbinBinned, m->chan_bwBinned);
    cpgmtxt("T", 4.0, 0.5, 0.5, title_part1);
    cpgmtxt("T", 3.0, 0.5, 0.5, title_part2);
    free(title_part1);
    free(title_part2);
    cpgsch(1.0);

    // === Clean up ===
    free(dsDataT_deRFI);
    free(dsTimeProfile);
    free(dsFreqProfile);
    free(dsChannelMask);
}

void plotDataAndMask(Metadata *m, int numBuffs, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, int *mask)
{
    // int nsampPlot = m->nsampBinned * numBuffs;
    int nsampPlot = m->nsampBinned; // Use binned samples for plotting
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbinBinned;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory ===
    float *dsDataT_deRFI = malloc(sizeof(float) * nsampPlot * nchanPlot);
    float *dsTimeProfile = malloc(sizeof(float) * nsampPlot);
    float *dsFreqProfile = malloc(sizeof(float) * nsampPlot);
    int *dsChannelMask = calloc(m->nchanBinned, sizeof(int));

    getProfile(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);
    // getProfileStd(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    palett(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===
    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(0.1, 0.7, 0.1, 0.7);
    float mean, std, nsig = 5.0;
    findMeanStd(dsDataT, nsampPlot * nchanPlot, &mean, &std);

    float tr[6] = {
        startTime + currentBlock * nsampPlot * tbinPlot, tstep,
        0, dsFreqArray[plotStartChan],
        0, fstep};

    cpgswin(tmin, tmax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);

    cpgsch(0.7);
    cpgbox("ABINST", 0, 0, "ABINPST", 0, 0);

    cpgimag(dsDataT + plotStartChan * nsampPlot,
            nsampPlot,
            plotEndChan - plotStartChan + 1,
            1, nsampPlot,
            1, plotEndChan - plotStartChan + 1,
            mean - nsig * std, mean + nsig * std,
            tr);

    plotIndexMask(fmin, nchanPlot, chan_bwPlot, tmin, nsampPlot, tbinPlot, mask, plotStartChan, plotEndChan);

    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(0.7, 0.85, 0.1, 0.7);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile, dsFreqArray + plotStartChan);

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Intensity");
    cpgmtxt("T", 1.0, 0.5, 0.5, "SED Curve");
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    cpgwedg("RI", 2.0, 2.5, mean - nsig * std, mean + nsig * std, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(0.1, 0.7, 0.7, 0.9);
    // 计算时间轴
    float time[nsampPlot];
    for (int i = 0; i < nsampPlot; i++)
    {
        time[i] = startTime + currentBlock * nsampPlot * tbinPlot + i * tstep;
    }
    float timeProfileMin, timeProfileMax;
    findMinMax(dsTimeProfile, nsampPlot, &timeProfileMin, &timeProfileMax);
    timeProfileMin -= 0.1 * (timeProfileMax - timeProfileMin);
    timeProfileMax += 0.1 * (timeProfileMax - timeProfileMin);
    cpgswin(tmin, tmax, timeProfileMin, timeProfileMax);
    cpgsch(0.7);
    cpgbox("BCMST", 0, 0, "BCNST", 0, 0);
    cpgline(nsampPlot, time, dsTimeProfile);
    cpglab("", "Intensity", "Time (s)");
    char *title_part1 = malloc(256 * sizeof(char));
    char *title_part2 = malloc(256 * sizeof(char));
    sprintf(title_part1, m->filename);
    sprintf(title_part2, "downsamp %d for time, and %d for freq, tbin = %.4e s, channel width = %.4e MHz",
            m->binFactorTime, m->binFactorFreq, tbinPlot, chan_bwPlot);
    cpgmtxt("T", 4.0, 0.5, 0.5, title_part1);
    cpgmtxt("T", 3.0, 0.5, 0.5, title_part2);
    free(title_part1);
    free(title_part2);
    cpgsch(1.0);

    // === Clean up ===
    free(dsDataT_deRFI);
    free(dsTimeProfile);
    free(dsFreqProfile);
    free(dsChannelMask);
}

void plotDownsampSEDStd(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, float *baseline)
{
    int nsampPlot = m->nsampBinned; // Use binned samples for plotting
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbinBinned;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory ===
    float *dsDataT_deRFI = malloc(sizeof(float) * m->nsampBinned * m->nchanBinned);
    float *dsTimeProfile = malloc(sizeof(float) * m->nsampBinned);
    float *dsFreqProfile = malloc(sizeof(float) * m->nchanBinned);
    int *dsChannelMask = calloc(m->nchanBinned, sizeof(int));

    getProfileStd(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    palett(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===
    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(0.1, 0.7, 0.1, 0.7);
    float mean, std, nsig = 5.0;
    findMeanStd(dsDataT, nsampPlot * nchanPlot, &mean, &std);

    float tr[6] = {
        startTime + currentBlock * nsampPlot * tbinPlot, tstep,
        0, dsFreqArray[plotStartChan],
        0, fstep};

    cpgswin(tmin, tmax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("ABINST", 0, 0, "ABINPST", 0, 0);

    cpgimag(dsDataT + plotStartChan * nsampPlot,  // Data pointer offset
            nsampPlot,                            // Number of time samples
            plotEndChan - plotStartChan + 1,      // Number of channels
            1, nsampPlot,                         // X range
            1, plotEndChan - plotStartChan + 1,   // Y range
            mean - nsig * std, mean + nsig * std, // Z range
            tr);
    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(0.7, 0.85, 0.1, 0.7);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile, dsFreqArray + plotStartChan);

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Value");
    cpgmtxt("T", 1.0, 0.5, 0.5, "Channel StdDev.");
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    cpgwedg("RI", 2.0, 2.5, mean - nsig * std, mean + nsig * std, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(0.1, 0.7, 0.7, 0.9);
    // Calculate time axis
    float time[nsampPlot];
    for (int i = 0; i < nsampPlot; i++)
    {
        time[i] = startTime + currentBlock * nsampPlot * tbinPlot + i * tstep;
    }
    float timeProfileMin, timeProfileMax;
    findMinMax(dsTimeProfile, nsampPlot, &timeProfileMin, &timeProfileMax);
    timeProfileMin -= 0.1 * (timeProfileMax - timeProfileMin);
    timeProfileMax += 0.1 * (timeProfileMax - timeProfileMin);
    cpgswin(tmin, tmax, timeProfileMin, timeProfileMax);
    cpgsch(0.7);
    cpgbox("BCMST", 0, 0, "BCNST", 0, 0);
    cpgline(nsampPlot, time, dsTimeProfile); // Use downsampled time array
    cpglab("", "Time StdDev.", "Time (s)");
    char *title_part1 = malloc(256 * sizeof(char));
    char *title_part2 = malloc(256 * sizeof(char));
    sprintf(title_part1, m->filename);
    sprintf(title_part2, "downsamp %d for time, and %d for freq, tbin = %.4e s, channel width = %.4e MHz",
            m->binFactorTime, m->binFactorFreq, m->tbinBinned, m->chan_bwBinned);
    cpgmtxt("T", 4.0, 0.5, 0.5, title_part1);
    cpgmtxt("T", 3.0, 0.5, 0.5, title_part2);
    free(title_part1);
    free(title_part2);
    cpgsch(1.0);

    // === Clean up ===
    free(dsDataT_deRFI);
    free(dsTimeProfile);
    free(dsFreqProfile);
    free(dsChannelMask);
}

void plotDataAndMaskStd(Metadata *m, int numBuffs, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, int *mask)
{
    int nsampPlot = m->nsampBinned; // Use binned samples for plotting
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbinBinned;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory ===
    float *dsDataT_deRFI = malloc(sizeof(float) * nsampPlot * nchanPlot);
    float *dsTimeProfile = malloc(sizeof(float) * nsampPlot);
    float *dsFreqProfile = malloc(sizeof(float) * nsampPlot);
    int *dsChannelMask = calloc(m->nchanBinned, sizeof(int));

    getProfileStd(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, 1);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    palett(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===
    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(0.1, 0.7, 0.1, 0.7);
    float mean, std, nsig = 5.0;
    findMeanStd(dsDataT, nsampPlot * nchanPlot, &mean, &std);

    float tr[6] = {
        startTime + currentBlock * nsampPlot * tbinPlot, tstep,
        0, dsFreqArray[plotStartChan],
        0, fstep};

    cpgswin(tmin, tmax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);

    cpgsch(0.7);
    cpgbox("ABINST", 0, 0, "ABINPST", 0, 0);

    cpgimag(dsDataT + plotStartChan * nsampPlot,
            nsampPlot,
            plotEndChan - plotStartChan + 1,
            1, nsampPlot,
            1, plotEndChan - plotStartChan + 1,
            mean - nsig * std, mean + nsig * std,
            tr);

    plotIndexMask(fmin, nchanPlot, chan_bwPlot, tmin, nsampPlot, tbinPlot, mask, plotStartChan, plotEndChan);

    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(0.7, 0.85, 0.1, 0.7);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile, dsFreqArray + plotStartChan);

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Value");
    cpgmtxt("T", 1.0, 0.5, 0.5, "Channel StdDev.");
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    cpgwedg("RI", 2.0, 2.5, mean - nsig * std, mean + nsig * std, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(0.1, 0.7, 0.7, 0.9);
    // 计算时间轴
    float time[nsampPlot];
    for (int i = 0; i < nsampPlot; i++)
    {
        time[i] = startTime + currentBlock * nsampPlot * tbinPlot + i * tstep;
    }
    float timeProfileMin, timeProfileMax;
    findMinMax(dsTimeProfile, nsampPlot, &timeProfileMin, &timeProfileMax);
    timeProfileMin -= 0.1 * (timeProfileMax - timeProfileMin);
    timeProfileMax += 0.1 * (timeProfileMax - timeProfileMin);
    cpgswin(tmin, tmax, timeProfileMin, timeProfileMax);
    cpgsch(0.7);
    cpgbox("BCMST", 0, 0, "BCNST", 0, 0);
    cpgline(nsampPlot, time, dsTimeProfile);
    cpglab("", "Time StdDev.", "Time (s)");
    char *title_part1 = malloc(256 * sizeof(char));
    char *title_part2 = malloc(256 * sizeof(char));
    sprintf(title_part1, m->filename);
    sprintf(title_part2, "downsamp %d for time, and %d for freq, tbin = %.4e s, channel width = %.4e MHz",
            m->binFactorTime, m->binFactorFreq, tbinPlot, chan_bwPlot);
    cpgmtxt("T", 4.0, 0.5, 0.5, title_part1);
    cpgmtxt("T", 3.0, 0.5, 0.5, title_part2);
    free(title_part1);
    free(title_part2);
    cpgsch(1.0);

    // === Clean up ===
    free(dsDataT_deRFI);
    free(dsTimeProfile);
    free(dsFreqProfile);
    free(dsChannelMask);
}


void plotIndexMask(
    float fmin, int nchanPlot, float chan_bwPlot,
    float tmin, int nsampPlot, float tbinPlot,
    int *mask,
    int plotStartChan, int plotEndChan)
{
    for (int i = plotStartChan; i <= plotEndChan; i++)
    {
        for (int j = 0; j < nsampPlot; j++)
        {
            int idx = i * nsampPlot + j;
            if (!mask[idx])
                continue;

            float x1 = tmin + (j + 0.0) * tbinPlot;
            float x2 = tmin + (j + 1.0) * tbinPlot;
            float y1 = fmin + (i - plotStartChan + 0.0) * chan_bwPlot;
            float y2 = fmin + (i - plotStartChan + 1.0) * chan_bwPlot;

            // 设置颜色
            int color_index;
            switch (mask[idx])
            {
            case 1:
                color_index = 4;
                break;
            case 2:
                color_index = 3;
                break;
            case 3:
                color_index = 5;
                break;
            default:
                color_index = mask[idx] % 16 + 1;
            }
            cpgsci(color_index);
            cpgrect(x1, x2, y1, y2);
        }
    }
    cpgsci(1);
}

void calcHist(float *data, int size, int numBins)
{
    float *hist = (float *)malloc(sizeof(float) * numBins);
    // Initialize histogram
    memset(hist, 0, sizeof(float) * numBins); // More effecient init

    // Calculate bin width
    float minVal = 0, maxVal = 127;
    // findMinMax(data, size, &minVal, &maxVal);


    if (maxVal == minVal)
    {
        maxVal = minVal + 1.0f;
    }

    float binWidth = (maxVal - minVal) / numBins;

    // Fill histogram with improved edges
    for (int i = 0; i < size; i++)
    {
        int binIndex = (int)((data[i] - minVal) / binWidth);

        // Handle when data[i] is exactly maxVal
        if (data[i] == maxVal)
        {
            binIndex = numBins - 1;
        }
        // Ensure binIndex is valid
        if (binIndex >= 0 && binIndex < numBins)
        {
            hist[binIndex]++;
        }
    }

    float maxHist = 0.0f;
    for (int i = 0; i < numBins; i++)
    {
        if (hist[i] > maxHist)
        {
            maxHist = hist[i];
        }
    }

    // cpgplot visualization
    cpgpage();
    cpgmtxt("T", 3.0, 0.35, 0.5, "Histogram");
    cpgsvp(0.1, 0.9, 0.1, 0.9);
    cpgsci(1);

    // Adjust Y range
    float yMax = (maxHist > 0) ? maxHist * 1.1f : 1.0f;
    cpgswin(minVal, maxVal, 0.0f, yMax);

    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(2);

    printf("Min: %f, Max: %f, BinWidth: %f\n", minVal, maxVal, binWidth);
    float binCenter;
    for (int i = 0; i < numBins; i++)
    {
        binCenter = minVal + (i + 0.5f) * binWidth;
        float leftEdge = minVal + i * binWidth;
        float rightEdge = minVal + (i + 1) * binWidth;
        // Use edge instead of center
        cpgrect(leftEdge, rightEdge, 0.0f, hist[i]);
        printf("Bin %d: %f\n", i, hist[i]);
    }

    cpgsci(1);
    cpgmtxt("B", 2.0, 0.5, 0.5, "Value");
    cpgmtxt("L", 2.0, 0.5, 0.5, "Frequency");
    cpgmtxt("T", 3.0, 0.5, 0.5, "Histogram of Data");
    free(hist);
}

void plot8bitHist(int *hist, float lowerBound, float upperBound, float mean, float sigma, int maxBin)
{
    /* === Draw rectangles for the hist === */
    cpgpage();
    cpgsvp(0.1, 0.9, 0.1, 0.9);
    cpgsci(1);
    cpgswin(lowerBound, upperBound + 1, 0, maxBin * 1.1);
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    cpgsci(2);
    for (int i = lowerBound; i <= upperBound; i++)
    {
        cpgrect(i, i + 1, 0, hist[i]);
    }

    /* === Title and axis labels === */
    cpgsci(1);
    char title[100];
    snprintf(title, sizeof(title), "8-bit Histogram (Range: %.1f~%.1f, Mean=%.4f, StdDev=%.4f)",
             lowerBound, upperBound, mean, sigma);
    cpgmtxt("T", 2.5, 0.5, 0.5, title);
    cpgmtxt("B", 2.0, 0.5, 0.5, "Value (0-255)");
    cpgmtxt("L", 2.0, 0.5, 0.5, "Frequency");

    /* === Draw vertical line at mean === */
    cpgsls(2);
    cpgsci(4);
    cpgslw(2);                   // Dashed line, blue, thickness 3
    cpgmove(mean, 0);            // Move to (mean, 0)
    cpgdraw(mean, maxBin * 1.1); // Draw to (mean, maxFreq * 1.1)
    cpgsls(1);
    cpgsci(1);
    cpgslw(1); // Reset
}