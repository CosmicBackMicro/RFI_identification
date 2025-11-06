#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ReadFASTData.h"
#include "cpgplot.h"
#include "psrPalett.h"
#include "plot.h"
#include "findStats.h"
#include "identification.h"

#define PALETT_GREY 1
#define PALETT_BLUE 2
#define PALETT_HEAT 3
#define PALETT_GOLD 4
#define PALETT_ALIEN_GLOW 6
#define PALETT_COLD 7
#define PALETT_PLASMA 8
#define PALETT_CUBE_HELIX 9
#define PALETT_VIRIDIS 10

// Ensure PGPLOT device is opened once.
int ensure_pgplot_device(const char *device)
{
    static int opened = 0;
    if (opened) return 1;

    const char *dev = device;
    if (!dev || !dev[0]) {
        dev = "output/plot.ps/VCPS";
    }
    int id = cpgopen(dev);
    if (id <= 0) {
        fprintf(stderr, "PGPLOT: failed to open device '%s'\n", dev);
        return 0;
    }
    cpgask(0); // do not pause between pages
    opened = 1;
    return 1;
}

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

void clampPlotRange(float *data, int dataSize, float nsig, float *zmin, float *zmax)
{
    float mean, std;
    findMeanStd(data, dataSize, &mean, &std);
    *zmin = mean - nsig * std;
    *zmax = mean + nsig * std;
}

void setupPalette(int palettType, double contrast, double brightness)
{
    palett(palettType, contrast, brightness);
}

void drawColorBar(float zmin, float zmax, const char *label)
{
    cpgwedg("RI", 2.0, 2.5, zmin, zmax, label);
}

void plotDownsampLongTimeAbs(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock)
{
    if (!ensure_pgplot_device(NULL)) {
        fprintf(stderr, "PGPLOT unavailable, skip plotDownsampLongTimeAbs.\n");
        return;
    }
    // === Global Layout Configuration ===
    float globalMargin = 0.1;  // Symmetric margin for left/right/top/bottom
    float mainPanelRight = 0.7;  // Main panel right boundary
    float mainPanelTop = 0.7;    // Main panel top boundary
    
    // === Panel Layout Configuration ===
    float mainPanel_x1 = globalMargin, mainPanel_x2 = mainPanelRight, mainPanel_y1 = globalMargin, mainPanel_y2 = mainPanelTop;
    float rightPanel_x1 = mainPanelRight, rightPanel_x2 = 1.0 - globalMargin, rightPanel_y1 = globalMargin, rightPanel_y2 = mainPanelTop;
    float topPanel_x1 = globalMargin, topPanel_x2 = mainPanelRight, topPanel_y1 = mainPanelTop, topPanel_y2 = 1.0 - globalMargin;

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

    getProfile(dsDataT, nsampPlot, nchanPlot, dsFreqProfile, dsTimeProfile, NULL);

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    setupPalette(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===

    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(mainPanel_x1, mainPanel_x2, mainPanel_y1, mainPanel_y2);
    float zmin, zmax, nsig = 5.0;
    clampPlotRange(dsDataT, nsampPlot * nchanPlot, nsig, &zmin, &zmax);

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
            zmin, zmax,                           // Z range
            tr);
    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(rightPanel_x1, rightPanel_x2, rightPanel_y1, rightPanel_y2);
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
    drawColorBar(zmin, zmax, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(topPanel_x1, topPanel_x2, topPanel_y1, topPanel_y2);
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

/**
 * @brief Advanced time-frequency plot with selectable panel statistics and optional mask overlay
 * @param m Metadata structure
 * @param numReads Number of reads  
 * @param dsDataT Downsampled data array (transposed)
 * @param dsFreqArray Frequency array
 * @param startTime Start time
 * @param currentBlock Current block index
 * @param baseline Baseline array
 * @param topPanelMode Mode for top panel: 0=mean, 1=std
 * @param rightPanelMode Mode for right panel: 0=mean, 1=std
 * @param mask Optional mask array for RFI flagging (NULL to skip mask plotting)
 */
void plotTimeFreqSED(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, float *baseline, int topPanelMode, int rightPanelMode, bool *mask, int *flaggedChans)
{
    if (!ensure_pgplot_device(NULL)) {
        fprintf(stderr, "PGPLOT unavailable, skip plotTimeFreqSED.\n");
        return;
    }
    // === Global Layout Configuration ===
    float globalMargin = 0.1;  // Symmetric margin for left/right/top/bottom
    float mainPanelRight = 0.7;  // Main panel right boundary
    float mainPanelTop = 0.7;    // Main panel top boundary
    
    // === Panel Layout Configuration ===
    float mainPanel_x1 = globalMargin, mainPanel_x2 = mainPanelRight, mainPanel_y1 = globalMargin, mainPanel_y2 = mainPanelTop;
    float rightPanel_x1 = mainPanelRight, rightPanel_x2 = 1.0 - globalMargin, rightPanel_y1 = globalMargin, rightPanel_y2 = mainPanelTop;
    float topPanel_x1 = globalMargin, topPanel_x2 = mainPanelRight, topPanel_y1 = mainPanelTop, topPanel_y2 = 1.0 - globalMargin;

    // Effective plotting geometry:
    // We may aggregate multiple SUBINT rows per iteration when binFactorTime>1.
    // The caller passes `numReads` as the number of SUBINT rows aggregated this iteration
    // (often m.blocksPerRead * m.binFactorTime). The actual binned time samples per plot
    // should therefore be (nsblk * numReads) / binFactorTime, while the time resolution is
    // tbin * binFactorTime so the total time span expands by binFactorTime.
    int binT = (m->binFactorTime > 0) ? m->binFactorTime : 1;
    int nsampPlot = (m->nsblk * numReads) / binT;
    int nchanPlot = m->nchanBinned;
    float tbinPlot = m->tbin * binT;
    float chan_bwPlot = m->chan_bwBinned;

    float tmin = startTime + currentBlock * nsampPlot * tbinPlot + 0.5 * tbinPlot;
    float tstep = tbinPlot;
    float tmax = startTime + currentBlock * nsampPlot * tbinPlot + (nsampPlot - 0.5) * tbinPlot;
    float fmin = dsFreqArray[0] + 0.5 * chan_bwPlot;
    float fstep = chan_bwPlot;
    float fmax = dsFreqArray[nchanPlot - 1] - 0.5 * chan_bwPlot;

    // == Allocate memory for profiles (align sizes to current plot geometry) ===
    float *dsTimeProfile_mean = (float *)malloc(sizeof(float) * nsampPlot);
    float *dsFreqProfile_mean = (float *)malloc(sizeof(float) * nchanPlot);
    float *dsTimeProfile_std  = (float *)malloc(sizeof(float) * nsampPlot);
    float *dsFreqProfile_std  = (float *)malloc(sizeof(float) * nchanPlot);

    // Calculate both mean and std profiles
    getProfile(dsDataT, nsampPlot, nchanPlot, dsFreqProfile_mean, dsTimeProfile_mean, mask);
    getProfileStd(dsDataT, nsampPlot, nchanPlot, dsFreqProfile_std, dsTimeProfile_std, mask);

    // Select profiles based on panel modes
    float *dsTimeProfile = (topPanelMode == 0) ? dsTimeProfile_mean : dsTimeProfile_std;
    float *dsFreqProfile = (rightPanelMode == 0) ? dsFreqProfile_mean : dsFreqProfile_std;
    
    const char *topPanelLabel = (topPanelMode == 0) ? "Time Mean \\gm\\di" : "Time StdDev. \\gs\\di";
    const char *rightPanelLabel = (rightPanelMode == 0) ? "Channel Mean \\gm\\dj" : "Channel StdDev. \\gs\\dj";

    int palettType = 3;
    double contrast = 1.0;
    double brightness = 0.4;
    setupPalette(palettType, contrast, brightness);

    // === Main Panel: Time-Frequency Plot ===
    float plotStartFreq = 1000.0, plotEndFreq = 1500.0;
    int plotStartChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotStartFreq);
    int plotEndChan = ClosestFreqIdx(dsFreqArray, nchanPlot, plotEndFreq);
    plotStartChan = (plotStartChan < 0) ? 0 : plotStartChan;
    plotEndChan = (plotEndChan > nchanPlot) ? nchanPlot : plotEndChan;

    cpgsvp(mainPanel_x1, mainPanel_x2, mainPanel_y1, mainPanel_y2);
    float zmin, zmax, nsig = 5.0;
    clampPlotRange(dsDataT, nsampPlot * nchanPlot, nsig, &zmin, &zmax);

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
            zmin, zmax, // Z range
            tr);

    // === Optional Mask Overlay ===
    if (mask != NULL) {
        plotIndexMask(fmin, nchanPlot, chan_bwPlot, tmin, nsampPlot, tbinPlot, mask, plotStartChan, plotEndChan);
    }

    cpglab("", "Frequency (MHz)", "");
    cpgmtxt("B", 2.5, 0.5, 0.5, "Time (s)");
    cpgsch(1.0);

    // === Right Panel: Time-integrated Profile ===
    cpgsvp(rightPanel_x1, rightPanel_x2, rightPanel_y1, rightPanel_y2);
    float freqProfileMin, freqProfileMax;
    findMinMax(dsFreqProfile, nchanPlot, &freqProfileMin, &freqProfileMax);
    freqProfileMin -= 0.1 * (freqProfileMax - freqProfileMin);
    freqProfileMax += 0.1 * (freqProfileMax - freqProfileMin);
    cpgswin(freqProfileMin, freqProfileMax, fmin + plotStartChan * chan_bwPlot, fmax - (nchanPlot - plotEndChan + 1) * chan_bwPlot);
    cpgsch(0.7);
    cpgbox("BCNST", 0, 0, "BCMST", 0, 0);
    
    // Draw frequency profile with gaps for fully flagged channels
    if (flaggedChans != NULL) {
        // Draw profile with gaps for fully flagged channels
        int segmentStart = -1;
        for (int i = plotStartChan; i <= plotEndChan + 1; i++) {
            int isChannelFlagged = (i <= plotEndChan) ? flaggedChans[i] : 1; // Treat end as flagged to close last segment
            
            if (!isChannelFlagged && segmentStart == -1) {
                // Start a new segment
                segmentStart = i;
            } else if ((isChannelFlagged || i > plotEndChan) && segmentStart != -1) {
                // End current segment and draw it
                int segmentLength = i - segmentStart;
                if (segmentLength > 0) {
                    cpgline(segmentLength, dsFreqProfile + segmentStart, dsFreqArray + segmentStart);
                }
                segmentStart = -1;
            }
        }
    } else {
        // Fallback to original continuous line if no flaggedChans info
        cpgline(plotEndChan - plotStartChan + 1, dsFreqProfile + plotStartChan, dsFreqArray + plotStartChan);
    }

    cpgmtxt("R", 2.1, 0.5, 0.5, "Frequency (MHz)");
    cpgmtxt("B", 3.0, 0.5, 0.5, "Value");
    cpgmtxt("T", 1.0, 0.5, 0.5, rightPanelLabel);
    cpgsch(1.0);

    // === Right Edge: Color Bar ===
    drawColorBar(zmin, zmax, "Intensity");

    // === Top Panel: Frequency-integrated Profile ===
    cpgsvp(topPanel_x1, topPanel_x2, topPanel_y1, topPanel_y2);
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
    cpglab("", topPanelLabel, "Time (s)");
    
    // Simplified title without panel mode details
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
    free(dsTimeProfile_mean);
    free(dsFreqProfile_mean);
    free(dsTimeProfile_std);
    free(dsFreqProfile_std);
}


void plotIndexMask(
    float fmin, int nchanPlot, float chan_bwPlot,
    float tmin, int nsampPlot, float tbinPlot,
    bool *mask,
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

            cpgsci(4); // 
            cpgrect(x1, x2, y1, y2);
        }
    }
    cpgsci(1);
}

void calcHist(float *data, int size, int numBins)
{
    if (!ensure_pgplot_device(NULL)) {
        fprintf(stderr, "PGPLOT unavailable, skip calcHist plotting.\n");
        return;
    }
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
    for (int i = 0; i < numBins; i++)
    {
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
    if (!ensure_pgplot_device(NULL)) {
        fprintf(stderr, "PGPLOT unavailable, skip plot8bitHist.\n");
        return;
    }
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

void plotAllMasks(Metadata *m, int blocksPerRead, float *outDataT, float *dsFreqArray, int startTime, int numiter, IdentNSigmaMasks *maskSet, int *flaggedChans) {
    // List of masks to plot, excluding globalMask
    bool *masks[] = {
        maskSet->horizontalMask, 
        maskSet->verticalMask, 
        maskSet->pointMask, 
        maskSet->chanBrightMask, 
        maskSet->chanDarkMask, 
        maskSet->chanComplexMask};
    char *maskNames[] = {"horizontalMask", "verticalMask", "pointMask", "chanBrightMask", "chanDarkMask", "chanComplexMask"};
    int numMasks = 6;

    for (int i = 0; i < numMasks; i++) {
        cpgpage();
        char title[100];
        snprintf(title, sizeof(title), "Mask: %s", maskNames[i]);
        cpgmtxt("T", 4.0, 0.35, 0.5, title);
        plotTimeFreqSED(m, blocksPerRead, outDataT, dsFreqArray, startTime, numiter, NULL, 1, 1, masks[i], flaggedChans);
    }
}