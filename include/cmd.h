#pragma once
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>

#include "ReadFASTData.h"

static struct option long_options[] = {
    {"filename", required_argument, NULL, 'i'},         // --filename
    {"startTime", optional_argument, NULL, 'S'},        // --startTime
    {"timeDuration", required_argument, NULL, 'd'},     // --timeDuration
    {"blocksPerRead", required_argument, NULL, 'n'},    // --blocksPerRead
    {"savePlot", required_argument, NULL, 's'},         // --savePlot
    {"binFactorTime", required_argument, NULL, 't'},    // --binFactorTime
    {"binFactorFreq", required_argument, NULL, 'f'},    // --binFactorFreq
    {"generateMasks", required_argument, NULL, 'M'},    // --generateMasks
    {"datasetPath", optional_argument, NULL, 'p'},      // --datasetPath
    {"plot", required_argument, NULL, 'P'},             // --plot
    {"write", required_argument, NULL, 'W'},            // --write
    {"doSubstitution", required_argument, NULL, 'e'},   // --doSubstitution
    {"doSumThreshold", required_argument, NULL, 'r'},   // --doSumThreshold
    {"help", no_argument, NULL, 'h'},                   // --help
    {NULL, 0, NULL, 0}                                  // End Mark for Options
};

int parseCommandLineArguments(int argc, char *argv[], Metadata *m);