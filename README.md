# deRFI - Radio Frequency Interference Detection and Mitigation

A high-performance RFI (Radio Frequency Interference) detection and mitigation tool for radio astronomy data, specifically designed for FAST telescope data processing.

## Features

- **Advanced Statistical Analysis**: MAD (Median Absolute Deviation) and STD-based RFI detection
- **Dual Histogram Visualization**: Main view and zoomed (0-0.25 range) histograms with Gaussian curve fitting
- **Channel-level Processing**: Individual channel median subtraction and statistical analysis
- **Multi-threading Support**: OpenMP parallelization for efficient processing
- **Flexible Thresholding**: Configurable N-sigma criteria for RFI detection
- **Real-time Visualization**: PGPLOT-based graphical output for monitoring

## Technical Highlights

- **Gaussian Curve Fitting**: Automatic fitting of Gaussian curves to histogram distributions
- **Adaptive Binning**: Smart histogram binning to handle quantized data
- **Robust Statistics**: Median-based calculations for improved outlier resistance
- **Memory Efficient**: Optimized memory usage for large datasets

## Dependencies

- CFITSIO library for FITS file handling
- PGPLOT for graphical output
- OpenMP for parallel processing
- Standard C libraries

## Building

```bash
make clean
make -j4
```

## Usage

```bash
./build/ReadFASTData [options] input_file.fits
```

## Project Structure

- `src/`: Source code files
- `include/`: Header files
- `build/`: Build output directory
- `Makefile`: Build configuration

## Recent Enhancements

- Added Gaussian curve fitting to MAD histograms
- Implemented channel median subtraction functionality
- Enhanced visualization with dual histogram views
- Improved statistical calculation methods

## License

This project is developed for radio astronomy research purposes.
