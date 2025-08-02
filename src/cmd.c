#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include "cmd.h"

// Ensure path ends with '/'
static void normalize_path(char *path) {
    size_t len = strlen(path);
    if (len > 0 && path[len - 1] != '/') {
        strcat(path, "/");
    }
}

// Call this function after parsing the command line arguments
int check_and_create_dataset_path(Metadata *m) {
    // Rectify dataset path to ensure it ends with a '/'
    normalize_path(m->datasetPath);

    // Check if the dataset path exists
    if (access(m->datasetPath, F_OK) == -1) {
        // Attempt to create the directory with permissions 0755
        if (mkdir(m->datasetPath, 0755) == -1) {
            fprintf(stderr, "Error: Failed to create directory '%s': %s\n",
                    m->datasetPath, strerror(errno));
            return -1;
        }
    } else {
        // If it exists, check if it's a directory
        struct stat st;
        if (stat(m->datasetPath, &st) == -1 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Error: '%s' exists but is not a directory\n",
                    m->datasetPath);
            return -1;
        }
    }
    return 0;
}
int parseCommandLineArguments(int argc, char *argv[], Metadata *m) {
    // Define long options
    static struct option long_options[] = {
        {"filename", required_argument, 0, 'i'},
        {"startTime", required_argument, 0, 'S'},
        {"timeDuration", required_argument, 0, 'd'},
        {"savePlot", required_argument, 0, 's'},
        {"binFactorTime", required_argument, 0, 't'},
        {"binFactorFreq", required_argument, 0, 'f'},
        {"doSumThreshold", required_argument, 0, 'r'},
        {"doSubstitution", required_argument, 0, 'e'},
        {"generateMasks", required_argument, 0, 'M'},
        {"datasetPath", required_argument, 0, 'p'},
        {"blocksPerRead", required_argument, 0, 'n'},
        {"plot", required_argument, 0, 'P'},
        {"write", required_argument, 0, 'W'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    // Default values
    m->binFactorTime = 1;
    m->binFactorFreq = 1;
    m->savePlot = 1;
    m->blocksPerRead = 1;
    m->generateMasks = 0;
    m->datasetPath = "./dataset/";
    m->startTime = 0.0f;
    m->blocksPerRead = 0;
    m->timeDuration = 0.0f;
    m->doSubstitution = 1;
    m->doSumThreshold = 1;
    m->enableCuda = 1;  // Default: enable CUDA if available

    int opt;
    while ((opt = getopt_long(argc, argv, "i:S:d:s:t:f:r:e:M:p:n:P:W:c:h", long_options, NULL))) {
        if (opt == -1) break;

        switch (opt) {
            case 'i':
                m->filename = optarg;
                break;
            case 'S':
                m->startTime = atof(optarg);
                break;
            case 'd':
                m->timeDuration = atof(optarg);
                break;
            case 'n':
                m->blocksPerRead = atoi(optarg);
                break;
            case 's':
                m->savePlot = atoi(optarg);
                break;
            case 't':
                m->binFactorTime = atoi(optarg);
                break;
            case 'f':
                m->binFactorFreq = atoi(optarg);
                break;
            case 'M':
                m->generateMasks = atoi(optarg);
                break;
            case 'p':
                m->datasetPath = optarg;
                break;
            case 'P':
                m->plot = atoi(optarg);
                break;
            case 'W':
                m->write = atoi(optarg);
                break;
            case 'e':
                m->doSubstitution = atoi(optarg);
                break;
            case 'r':
                m->doSumThreshold = atoi(optarg);
                break;
            case 'c':
                m->enableCuda = atoi(optarg);
                break;
            case 'h':
                printf("Usage: %s [OPTIONS]\n", argv[0]);
                printf("Options:\n");
                printf("  -i, --filename=FILENAME       Input FITS file\n");
                printf("  -S, --startTime=TIME          Start time in seconds, if unspecified is 0.\n");
                printf("  -d, --timeDuration=DURATION   Time duration in seconds\n");
                printf("  -n, --blocksPerRead=BLOCKS    Number of blocks to read at once\n");
                printf("  -s, --savePlot=MODE           Save plots to a PostScript?\n");
                printf("  -t, --binFactorTime=FACTOR    Time binning factor\n");
                printf("  -f, --binFactorFreq=FACTOR    Frequency binning factor\n");
                printf("  -M, --generateMasks=MASKS     Whether to generate masks\n");
                printf("  -p, --datasetPath=PATH        Path to dataset\n");
                printf("  -P, --plot=MODE               Plot or not\n");
                printf("  -c, --enableCuda=MODE         Enable CUDA acceleration (1=enable, 0=disable)\n");
                printf("  -h, --help                    Show this help message\n");
                break;
                return 1; // Return non-zero to indicate no further processing
        }
    }

    check_and_create_dataset_path(m);

    // Verify required parameters
    if (m->filename == NULL) {
        fprintf(stderr, "Error: Input filename is required\n");
        return -1;
    }

    return 0; // Success
}