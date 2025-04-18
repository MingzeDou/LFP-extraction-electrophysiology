# LFP Extraction Tool

## Overview
The LFP Extraction Tool is designed to efficiently extract Local Field Potential (LFP) data from large, multi-channel electrophysiology recordings stored in binary (.dat) files. It offers two processing paths: a GPU-accelerated version for maximum speed on compatible hardware, and a CPU-only version for broader accessibility.

## Motivation
Extracting LFP signals (typically < 500 Hz) from broadband neural recordings (often sampled at 30 kHz or higher) is a fundamental step in electrophysiology data analysis. Standard tools may struggle with the sheer size of modern datasets (often hundreds of GBs or TBs). This tool aims to provide fast, memory-efficient, and validated solutions using Python.

## Features
- Efficient down-sampling of raw data (e.g., 30 kHz to 1250 Hz)
- Low-pass filtering to isolate LFP frequencies (e.g., < 450 Hz)
- Chunk processing to handle large files without loading entirely into memory.
- **GPU Path (`GPU_process`):** Uses CuPy for significant speedup (5-10x or more) on NVIDIA GPUs. Requires CUDA and CuPy installation. Filter state propagation ensures signal continuity between chunks.
- **CPU Path (`CPU_process`):** Uses NumPy/SciPy for processing on standard CPUs. Handles chunk boundaries using data overlap to avoid artifacts.
- Progress tracking during extraction.
- Includes validation scripts (`compare_files.py` in each directory) to compare extracted LFP with downsampled raw data.

## How it Works

### GPU Path (`GPU_process`)
1.  **Reading Data:** Reads the input `.dat` file in chunks.
2.  **GPU Transfer:** Transfers the data chunk to the GPU using CuPy.
3.  **Filtering:** Each chunk is low-pass filtered on the GPU using a Butterworth filter implemented via `cupyx.scipy.signal`. Filter state (`zi`) is propagated between chunks to ensure signal continuity without explicit overlap handling in the main loop.
4.  **Downsampling:** The filtered data is downsampled on the GPU to the target LFP sampling rate using `cupyx.scipy.signal.decimate`, which includes an anti-aliasing filter.
5.  **CPU Transfer:** Transfers the processed LFP data back to the CPU.
6.  **Writing Data:** The processed LFP data (as int16) is written to the output `.lfp` file.

### CPU Path (`CPU_process`)
1.  **Reading Data:** Reads the input `.dat` file in chunks.
2.  **Overlap:** To prevent filtering artifacts at chunk boundaries, a small overlap (e.g., 0.5 seconds of data) is read and processed between consecutive chunks.
3.  **Filtering:** Each chunk (including overlap) is low-pass filtered using a Butterworth filter implemented via `scipy.signal`. Filter state can optionally be propagated.
4.  **Downsampling:** The filtered data is downsampled to the target LFP sampling rate using `scipy.signal.decimate`, which includes an anti-aliasing filter.
5.  **Overlap Removal:** The overlapping section (now downsampled) is removed from the beginning of the processed chunk before writing.
6.  **Writing Data:** The processed LFP data (as int16) is written to the output `.lfp` file.

## Project Structure
```
lfp-extraction-tool
├── GPU_process/         # GPU-accelerated implementation
│   ├── src/
│   │   ├── extract_lfp.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── filters.py
│   │   ├── config.py
│   ├── run.py
│   ├── compare_files.py
│   └── requirements.txt     # Python requirements for GPU path
├── CPU_process/         # CPU-only implementation
│   ├── src/
│   │   ├── extract_lfp.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── filters.py
│   │   ├── config.py
│   ├── run.py
│   ├── compare_files.py
│   └── requirements.txt     # Python requirements for CPU path
└── README.md            # This file
```

## Requirements
- Python 3.12
- **For GPU Acceleration (`GPU_process` directory):**
    - See `GPU_process/requirements.txt`.
    - Requires CUDA-compatible NVIDIA GPU, CUDA Toolkit, and compatible drivers.
    - `cupy` installation is separate and depends on your CUDA version (see Installation).
- **For CPU-Only Operation (`CPU_process` directory):**
    - See `CPU_process/requirements.txt`.
- **Other Optional Packages:**
    - `h5py` (Included in both requirements files, but not used by core scripts)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MingzeDou/LFP-extraction-electrophysiology
    cd LFP-extraction-electrophysiology
    ```

2.  Set up a virtual environment (conda recommended):
    ```bash
    conda create -n lfp-env python=3.12
    conda activate lfp-env
    ```

3.  **Install requirements based on your needs:**

    *   **For GPU Processing:** Navigate to the `GPU_process` directory and install its requirements.
        ```bash
        cd GPU_process
        pip install -r requirements.txt
        # This installs numpy, scipy, tqdm, pycuda, matplotlib, pandas, tabulate, h5py
        ```
        **Then, install CuPy (Crucial Step):**
        *   **Using Conda (Recommended):** Find the correct command for your CUDA version from the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge). Example for CUDA 12.x:
            ```bash
            conda install -c conda-forge cupy cudatoolkit=12.0
            ```
        *   **Using Pip:** Find the correct command for your CUDA version. Example for CUDA 12.x:
            ```bash
            pip install cupy-cuda12x
            ```
        *   Verify CuPy installation:
            ```python -c "import cupy; print(cupy.show_config())" ```
        ```bash
        cd .. # Return to the main directory
        ```

    *   **For CPU Processing:** Navigate to the `CPU_process` directory and install its requirements.
        ```bash
        cd CPU_process
        pip install -r requirements.txt
        # This installs numpy, scipy, tqdm, matplotlib, pandas, tabulate, h5py
        ```
        ```bash
        cd .. # Return to the main directory
        ```

## Usage

Choose the appropriate directory based on whether you want to use GPU or CPU processing.

### GPU Accelerated Extraction (`GPU_process`)

Navigate to the `GPU_process` directory:
```bash
cd GPU_process
```

Basic usage (uses defaults from `src/config.py`):
```bash
python run.py --input_file path/to/your/input.dat --output_file path/to/your/output.lfp
```

Specify all options:
```bash
python run.py \
  --input_file path/to/your/input.dat \
  --output_file path/to/your/output.lfp \
  --chunk_size 1073741824 \ # Chunk size in bytes (e.g., 1GB). Adjust based on GPU RAM.
  --num_channels 385
```

### CPU Only Extraction (`CPU_process`)

Navigate to the `CPU_process` directory:
```bash
cd CPU_process
```

Basic usage (uses defaults from `src/config.py`):
```bash
python run.py --input_file path/to/your/input.dat --output_file path/to/your/output.lfp
```

Specify all options:
```bash
python run.py \
  --input_file path/to/your/input.dat \
  --output_file path/to/your/output.lfp \
  --chunk_size 1073741824 \ # Chunk size in bytes (e.g., 1GB). Adjust based on system RAM.
  --num_channels 385
```

### Command Line Options (Apply to both `run.py` scripts)
- `--input_file`: Path to input raw data file (`.dat`) (required).
- `--output_file`: Path for output LFP file (`.lfp`) (required).
- `--chunk_size`: Processing chunk size in bytes. Must be divisible by `num_channels * 2`. (Default: `1073741824` (1GB)). Larger chunks can be faster but require more memory (System RAM for CPU, GPU RAM for GPU).
- `--num_channels`: Number of channels in the recording (Default: `385`).

### Common Issues
1.  **GPU Out of Memory (`GPU_process`)**:
    *   Reduce `--chunk_size` (e.g., `536870912` for 512MB). **Ensure the new size is still divisible by `num_channels * 2`**.
    *   Ensure no other processes are heavily using the GPU.
2.  **File Not Found**:
    *   Use absolute paths for input/output files or ensure relative paths are correct from the directory you are running the script (`GPU_process` or `CPU_process`).
    *   Check file permissions.
3.  **CUDA/CuPy Errors (`GPU_process`)**:
    *   Verify CUDA toolkit, NVIDIA driver, and CuPy versions are compatible. See CuPy installation guide.
    *   Ensure `cupy` is installed correctly in your environment.
    *   If errors persist and you cannot resolve them, use the CPU-only version by running the scripts in the `CPU_process` directory. The `GPU_process` scripts *require* a working CuPy installation.
4.  **Chunk Size Divisibility**: Ensure `--chunk_size` is a multiple of `num_channels * 2` (bytes per frame). The scripts attempt to adjust this, but setting it correctly is best.

## Configuration

Default parameters can be adjusted in the `config.py` file within the respective `GPU_process` or `CPU_process` directory.

### `GPU_process/src/config.py`
```python
# Signal Processing
SAMPLE_RATE_ORIGINAL = 30000  # Hz (Raw data sample rate)
TARGET_SAMPLING_RATE = 1250   # Hz (Desired LFP sample rate)
CUTOFF_FREQUENCY = 450        # Hz (Low-pass filter cutoff)

# Processing
# IMPORTANT: CHUNK_SIZE must be divisible by N_CHANNELS * 2 (bytes per frame)
CHUNK_SIZE = 1073741824 # Bytes (1GB) - Adjust based on GPU memory
N_CHANNELS = 385        # Number of recording channels

# File Paths (can be overridden by command line args)
INPUT_FILE = "path/to/input.dat"
OUTPUT_FILE = "path/to/output.lfp"
```

### `CPU_process/src/config.py`
```python
# Signal Processing
SAMPLE_RATE_ORIGINAL = 30000  # Hz (Raw data sample rate)
TARGET_SAMPLING_RATE = 1250   # Hz (Desired LFP sample rate)
CUTOFF_FREQUENCY = 450        # Hz (Low-pass filter cutoff)

# Processing
# IMPORTANT: CHUNK_SIZE must be divisible by N_CHANNELS * 2 (bytes per frame)
CHUNK_SIZE = 1073741824 # Bytes (1GB) - Adjust based on system RAM
N_CHANNELS = 385        # Number of recording channels
OVERLAP_SECONDS = 0.5   # Seconds of overlap between chunks for CPU processing

# File Paths (can be overridden by command line args)
INPUT_FILE = "path/to/input.dat"
OUTPUT_FILE = "path/to/output.lfp"
```

**Important Notes:**
- The downsampling factor (`SAMPLE_RATE_ORIGINAL / TARGET_SAMPLING_RATE`) must be an integer.
- `CUTOFF_FREQUENCY` should be less than half the `TARGET_SAMPLING_RATE` (Nyquist theorem).
- `CHUNK_SIZE` significantly impacts performance and memory usage. Experiment to find the optimal value for your system.
- `OVERLAP_SECONDS` (CPU path only) determines how much data is re-processed at chunk boundaries to avoid artifacts. 0.1-0.5 seconds is usually sufficient.

## Validation with compare_files.py

Both `GPU_process` and `CPU_process` directories contain a `compare_files.py` script to help verify the LFP extraction quality by comparing the output `.lfp` file against the original `.dat` file. **Run the script from the same directory (`GPU_process` or `CPU_process`) where you ran the extraction.**

### Features
- Time domain comparison (plots overlay of downsampled raw vs processed LFP).
- Frequency domain analysis (Power Spectral Density - PSD).
- Coherence analysis between signals.
- Frequency band power comparison.
- Signal-to-noise ratio (SNR) estimation.
- Calculation of correlation coefficients.
- Option to save analysis metrics to a text file.

### Usage
Navigate to the relevant directory (`GPU_process` or `CPU_process`):
```bash
cd GPU_process  # or cd CPU_process
```

Usage:
```bash
python compare_files.py path/to/raw_data.dat path/to/processed_data.lfp \
  --num-channels 385 \
  --num-samples 60000 \ # Number of raw samples to analyze
  --analysis detailed \ # 'basic', 'detailed', 'frequency', or 'all'
  --channel 10 \        # Channel index for detailed/frequency analysis
  --output analysis_report.txt # Optional output file for metrics
```
*(See script arguments using `python compare_files.py -h` for more details)*

### Interpreting Results
- **High correlation** (>0.95 expected) indicates good signal preservation.
- **Similar PSD shapes** below the cutoff frequency.
- **High coherence** (~1.0) below the cutoff frequency.
- **Comparable power** in relevant frequency bands (Delta, Theta, etc.).

## Performance Considerations
*   **GPU vs CPU**: The `GPU_process` path offers substantial speedup if you have a compatible NVIDIA GPU and CuPy installed. The `CPU_process` path provides a reliable alternative.
    *   *Example:* On a system with an NVIDIA RTX A4000 GPU and an Intel i9-13900KF CPU, the GPU implementation was observed to be approximately **3.3 times faster** than the CPU implementation. Actual speedup will vary based on hardware specifics and data size.
*   **Chunk Size**: Larger chunks reduce overhead but increase memory demand (GPU RAM for `GPU_process`, system RAM for `CPU_process`). Find the sweet spot for your hardware, **ensuring the chunk size is divisible by `num_channels * 2`**.
*   **Overlap (`CPU_process` only)**: Larger overlap increases computation slightly but improves robustness against edge artifacts in the CPU version.

## Contributing
Contributions are welcome! If you find bugs, have suggestions, or want to add features (e.g., support for other file formats, different filter types), please feel free to:
1.  Open an issue on the GitHub repository to discuss the change.
2.  Fork the repository, make your changes, and submit a pull request.

Please ensure code is well-commented and, if adding features, consider adding corresponding tests or validation steps.

## License
This project is licensed under the MIT License.

## Citation

If you use this software in your research, please cite it using the DOI provided below.

**Example Citation (Version 1.0):**

> Dou, Mingze. (2025). LFP Extraction Toolkit for Electrophysiology Data (Version 1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15243756

**BibTeX Entry:**

```bibtex
@software{Dou_LFP_Extraction_2024_15243756,
  author       = {Dou, Mingze},
  title        = {{LFP Extraction Toolkit for Electrophysiology Data}},
  month        = april, 
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.15243756},
  url          = {https://doi.org/10.5281/zenodo.15243756}
}
