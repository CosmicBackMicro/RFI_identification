#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>
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
        {"writeBack", required_argument, 0, 'B'},
        {"writeMasks", required_argument, 0, 'k'},
    {"enableIQRM", required_argument, 0, 'q'},
    {"enableCLFD", required_argument, 0, 'l'},
        {"enableCuda", required_argument, 0, 'c'},
        {"inChanNSigma", required_argument, 0, 'I'},
        {"outChanNSigma", required_argument, 0, 'O'},
        {"ncpus", required_argument, 0, 'T'},
        {"fallbackMeanNSigma", required_argument, 0, 'F'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    // Default values
    m->binFactorTime = 1;
    m->binFactorFreq = 1;
    m->savePlot = 1;
    m->blocksPerRead = 1;  // Default to 1 block per read
    m->generateMasks = 0;
    m->datasetPath = "./dataset/";
    m->startTime = 0.0f;
    m->timeDuration = 0.0f;
    m->doSubstitution = 1;
    m->doSumThreshold = 1;
    m->enableCuda = 1;  // Default: enable CUDA if available
    m->writeBack = 0;    // Default: do not write back to original file
    m->writeMasks = 1;   // Default: write mask images
    m->enableIQRM = 0;   // Default: IQRM disabled
    m->enableCLFD = 0;   // Default: CLFD disabled
    m->NSigmaInChan = 3.0f;   // Default NSigma thresholds
    m->NSigmaOutChan = 3.0f;
    m->FallbackMeanNSigma = 2.0f; // 默认均值兜底 2σ
    m->ncpus = 20; // default threads

    int opt;
    while ((opt = getopt_long(argc, argv, "i:S:d:s:t:f:r:e:M:p:n:P:W:B:k:c:I:O:F:T:h", long_options, NULL))) {
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
            case 'B':
                m->writeBack = atoi(optarg);
                break;
            case 'k':
                m->writeMasks = atoi(optarg);
                break;
            case 'q':
                m->enableIQRM = atoi(optarg);
                break;
            case 'l':
                m->enableCLFD = atoi(optarg);
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
            case 'I':
                m->NSigmaInChan = atof(optarg);
                break;
            case 'O':
                m->NSigmaOutChan = atof(optarg);
                break;
            case 'F':
                m->FallbackMeanNSigma = atof(optarg);
                if (m->FallbackMeanNSigma <= 0.0f) m->FallbackMeanNSigma = 2.0f; // 合理性保护
                break;
            case 'T':
                m->ncpus = atoi(optarg);
                if (m->ncpus <= 0) m->ncpus = 1;
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
                printf("  -W, --write=MODE              Write processed data to separate FITS files\n");
                printf("  -B, --writeBack=MODE          Write back modified data to original FITS file (dangerous!)\n");
                printf("  -k, --writeMasks=MODE         Write mask images to PNG files\n");
                printf("  -q, --enableIQRM=MODE         Enable IQRM channel detection (1=enable, 0=disable)\n");
                printf("  -l, --enableCLFD=MODE         Enable CLFD channel detection (1=enable, 0=disable)\n");
                printf("  -c, --enableCuda=MODE         Enable CUDA acceleration (1=enable, 0=disable)\n");
                printf("  -I, --inChanNSigma=VALUE      NSigma threshold for in-channel outlier detection (default 3.0)\n");
                printf("  -O, --outChanNSigma=VALUE     NSigma threshold for cross-channel detection (default 3.0)\n");
                printf("  -F, --fallbackMeanNSigma=VAL   Fallback mean-based channel sigma clip (default 2.0)\n");
                printf("  -T, --ncpus=N                 Number of CPU threads to use (default 20)\n");
                printf("  -h, --help                    Show this help message\n");
                return 1; // Return non-zero to indicate no further processing
        }
    }

    check_and_create_dataset_path(m);

    // Verify required parameters
    if (m->filename == NULL) {
        fprintf(stderr, "Error: Input filename is required\n");
        return -1;
    }

    // Check if the parent directory exists
    char *filename_copy = strdup(m->filename);
    char *parent_dir = dirname(filename_copy);
    struct stat dir_st;
    if (stat(parent_dir, &dir_st) == -1) {
        fprintf(stderr, "Error: Parent directory '%s' does not exist\n", parent_dir);
        free(filename_copy);
        return -1;
    }
    if (!S_ISDIR(dir_st.st_mode)) {
        fprintf(stderr, "Error: '%s' is not a directory\n", parent_dir);
        free(filename_copy);
        return -1;
    }
    free(filename_copy);

    // Check if the input file exists
    struct stat st;
    if (stat(m->filename, &st) == -1) {
        fprintf(stderr, "Error: Input file '%s' does not exist\n", m->filename);
        return -1;
    }

    // Safety check for writeBack
    if (m->writeBack) {
        printf("ATTENTION! --writeBack will modify the original data! If you know what you are doing, type the full sentence 'I know what I am doing!' and press Enter.\n");
        char input[256];
        if (fgets(input, sizeof(input), stdin) == NULL) {
            fprintf(stderr, "Error reading input\n");
            return -1;
        }
        // Remove newline
        input[strcspn(input, "\n")] = 0;
        if (strcmp(input, "I know what I am doing!") != 0) {
            fprintf(stderr, "Confirmation failed. Exiting.\n");
            return -1;
        }
    }

    // Enforce mutual exclusivity: IQRM and CLFD cannot both be enabled
    if (m->enableIQRM && m->enableCLFD) {
        fprintf(stderr, "Error: --enableIQRM and --enableCLFD are mutually exclusive. Please enable only one.\n");
        return -1;
    }

    return 0; // Success
}