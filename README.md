# LFP Extraction Tool

## Overview
The LFP Extraction Tool is designed to efficiently extract Local Field Potential (LFP) data from large electrophysiology recordings stored in .dat files. The tool down-samples the data to 1250 Hz and applies a low-pass filter at 450 Hz, ensuring that the resulting LFP data is suitable for further analysis.

## Features
- Down-sampling of raw data from 30 kHz to 1250 Hz
- Application of a low-pass filter at 450 Hz
- Chunk processing to handle large files efficiently
- Utilization of RTX A4000 GPU for accelerated computations
- Memory-efficient processing of large datasets
- Progress tracking during extraction

## Project Structure
```
lfp-extraction-tool
├── src
│   ├── extract_lfp.py       # Main script for LFP extraction
│   ├── utils
│   │   ├── __init__.py      # Initializes the utils package
│   │   ├── gpu_utils.py     # GPU utility functions
│   │   └── filters.py       # Functions for filtering data
│   └── config.py            # Configuration settings
├── requirements.txt         # Required Python packages
├── run.py                   # Entry point for the application
├── compare_files.py         # Tool for validating LFP extraction quality
└── README.md                # Project documentation
```

## Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) with:
  - CUDA Toolkit 11.x or 12.x installed
  - Compatible NVIDIA drivers
- For CPU-only operation:
  - NumPy and SciPy will be used instead
- Required Python packages:
  - NumPy (>=1.20.0)
  - SciPy (>=1.7.0)
  - CuPy (>=10.0.0, optional for GPU acceleration)
  - tqdm (>=4.0.0, for progress tracking)

## Installation

1. First clone the repository:
```bash
git clone https://github.com/MingzeDou/LFP-extraction-electrophysiology
cd lfp-extraction-tool
```

2. Set up a conda environment (recommended):
```bash
conda create -n lfp-env python=3.11
conda activate lfp-env
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

For GPU acceleration (optional but recommended):
```bash
# For CUDA 13
conda install -c conda-forge cupy
```
If you encounter issues, you can verify your CUDA version and install the matching cupy:
```bash
nvcc --version  # Check CUDA version
# OR
conda install -c conda-forge cupy cudatoolkit=13.0  # Explicitly specify version
```

## Usage

Basic usage:
```bash
python run.py --input_file input.dat --output_file output.lfp
```

Advanced usage with all options:
```bash
python run.py \
  --input_file input.dat \
  --output_file output.lfp \
  --chunk_size 499998720 \  # ~ process in chunks, keep in mind that this should be the multiple of num_channels
  --num_channels 384
```

### Command Line Options
- `--input_file`: Path to input .dat file (required)
- `--output_file`: Path for output .lfp file (required)
- `--chunk_size`: Processing chunk size in bytes (default: 100MB)
- `--num_channels`: Number of channels in recording (default: 384)

### Common Issues
1. **GPU Out of Memory**:
   - Reduce chunk_size (try 50MB)
   - Ensure no other GPU processes are running
   
2. **File Not Found**:
   - Use absolute paths
   - Check file permissions

3. **CUDA Errors**:
   - Verify CUDA toolkit and driver versions
   - Try CPU-only mode by uninstalling CuPy

## Configuration

Key parameters in `src/config.py`:

```python
# Signal Processing
SAMPLE_RATE_ORIGINAL = 30000  # Hz (30kHz)
TARGET_SAMPLING_RATE = 1250   # Hz 
CUTOFF_FREQUENCY = 450        # Hz (should be < TARGET_SAMPLING_RATE/2)

# Processing
CHUNK_SIZE = 499998720  # process in chunks (adjust based on GPU memory)
N_CHANNELS = 384        # Number of recording channels

# File Paths (override with command line args)
INPUT_FILE = "path/to/input.dat"
OUTPUT_FILE = "path/to/output.lfp"
```

Important Notes:
- The downsampling factor (original_rate/target_rate) must be an integer
- Cutoff frequency should be less than half the target sampling rate (Nyquist)
- Optimal chunk size depends on available GPU memory

## Validation with compare_files.py

The LFP Extraction Tool includes a validation script that helps you verify the quality and accuracy of your extracted LFP data by comparing it with the original raw data.

### Features of compare_files.py

- Time domain comparison of raw and downsampled signals
- Frequency domain analysis including power spectral density
- Coherence analysis between original and extracted signals
- Frequency band power comparison (Delta, Theta, Alpha, Beta, Gamma)
- Signal-to-noise ratio (SNR) calculation
- Detailed metrics and statistical comparison

### Using the validation tool

Basic usage:
```bash
python compare_files.py raw_data.dat processed_data.lfp
```

Advanced usage:
```bash
python compare_files.py raw_data.dat processed_data.lfp \
  --num-channels 384 \
  --num-samples 60000 \
  --analysis detailed \
  --channel 10 \
  --output analysis_report.txt
```

### Command Line Options

- `dat_file`: Path to the original .dat file (required)
- `lfp_file`: Path to the generated .lfp file (required)
- `--num-channels`: Number of channels in recordings (default: 384)
- `--num-samples`: Number of samples to analyze (default: 30000)
- `--analysis`: Type of analysis to perform:
  - `basic`: Simple signal correlation across multiple channels
  - `detailed`: Comprehensive analysis of a single channel
  - `frequency`: Frequency band analysis of a single channel
  - `all`: All analysis types (default)
- `--channel`: Channel number to analyze in detail (default: 0)
- `--output`: Path to save summary results (optional)

### Example Analysis Output

The analysis script generates both visual and numerical outputs:

1. For basic analysis:
   - Correlation values between raw and processed signals
   - Time-domain plot showing signal overlays for multiple channels

2. For detailed analysis:
   - Time-domain comparison
   - Power spectral density plots
   - Coherence analysis
   - Summary statistics (correlation, SNR, coherence metrics)

3. For frequency band analysis:
   - Power comparison across canonical frequency bands
   - Visualization of power differences

### Interpreting Results

- **High correlation values** (>0.9) indicate good fidelity in the downsampling and filtering process
- **Similar power spectral densities** confirm frequency content is preserved appropriately
- **Strong coherence** (especially in the 0-200Hz range) validates that the signal timing relationships are maintained
- **Comparable frequency band power** confirms that the relative energy in each band is preserved

## Performance Considerations

1. **GPU vs CPU**:
   - GPU acceleration provides 5-10x speedup
   - CPU mode works but is slower

2. **Chunk Size**:
   - Larger chunks = better performance
   - But requires more GPU memory
   - Recommended starting points:
     - GPU: 100-500MB chunks
     - CPU: 50-100MB chunks

3. **Monitoring**:
   - Progress is shown every 10 chunks
   - Memory is freed periodically
   - Watch for CUDA out-of-memory errors

4. **Multi-channel Data**:
   - Processing scales linearly with channel count
   - Very high channel counts may require smaller chunks

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.
