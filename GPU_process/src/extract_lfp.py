import os
import numpy as np
import time  # Add time import

# Import the updated filter functions
from src.utils.filters import apply_filter_and_downsample
from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY

# Import CuPy - Raise error if unavailable
try:
    import cupy as cp

    # Test CUDA functionality
    a = cp.array([1, 2, 3])
    a.sum()
    # Simple kernel test - just check compilation, don't call
    cp.RawKernel('extern "C" __global__ void example() {}', "example")
    print(f"Using CuPy version: {cp.__version__}")
    print(f"GPU device: {cp.cuda.get_device_id()}")
except ImportError as e:
    raise ImportError(
        f"CuPy failed to import: {e}. Please install CuPy or use the CPU_process scripts."
    ) from e
except Exception as e:  # Catch other potential CUDA errors
    raise RuntimeError(
        f"CuPy initialization failed ({type(e).__name__}: {e}). Ensure CUDA drivers and toolkit are compatible. Otherwise, use the CPU_process scripts."
    ) from e


def free_gpu_memory():
    """Free GPU memory."""
    # Assumes cp is available if this function is called
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        # print("GPU memory freed.") # Optional: for debugging
    except Exception as e:
        print(f"Error freeing GPU memory: {e}")


def extract_lfp(input_file, output_file, chunk_size, num_channels):
    """
    Extract LFP from raw data file (.dat) using GPU acceleration and overlapped chunking.
    Uses zero-phase filtering (filtfilt) by loading chunks with padding.

    Args:
        input_file: Path to input raw data file (.dat)
        output_file: Path to output LFP file
        chunk_size: Target size of chunks to process (in bytes).
                    Will be converted to samples.
        num_channels: Number of channels in the data.
    """
    # Validate inputs
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing data file: {input_file} to {output_file}")
    print(f"Using Overlapped Chunking for Zero-Phase Filtering on GPU.")

    # Data type and sizes
    dtype_input = np.int16
    bytes_per_sample = np.dtype(dtype_input).itemsize

    # Calculate file structure
    file_size_bytes = os.path.getsize(input_file)
    total_samples = file_size_bytes // (num_channels * bytes_per_sample)

    # Calculate chunk size in samples (per channel)
    # chunk_size input is roughly bytes for all channels
    # We want a manageable number of TIME samples to load at once
    # Let's target ~1GB of VRAM usage or use the provided chunk_size
    # chunk_size (bytes) / (num_channels * 2 bytes/sample) = samples_per_chunk
    samples_per_chunk_target = chunk_size // (num_channels * bytes_per_sample)

    # Ensure chunk size is divisible by downsampling factor to keep alignment simple
    downsample_factor = int(SAMPLE_RATE_ORIGINAL / TARGET_SAMPLING_RATE)
    if samples_per_chunk_target % downsample_factor != 0:
        samples_per_chunk_target -= samples_per_chunk_target % downsample_factor

    # Define Padding (Context) for stable filtering
    # 0.5 seconds of context is usually sufficient for LFP filters
    pad_samples = int(0.5 * SAMPLE_RATE_ORIGINAL)

    print(f"Total Samples: {total_samples}")
    print(f"Chunk Size (Time Samples): {samples_per_chunk_target}")
    print(f"Padding (Time Samples): {pad_samples}")

    # Open Input File using Memmap (Read-Only) - Fast & Low Memory
    # Shape: (Time, Channels)
    data_map = np.memmap(
        input_file, dtype=dtype_input, mode="r", shape=(total_samples, num_channels)
    )

    start_time = time.time()

    # Open Output File
    # We will append to it.
    with open(output_file, "wb") as f_out:

        chunk_idx = 0
        current_sample = 0

        while current_sample < total_samples:
            # 1. Define Standard Block
            block_start = current_sample
            block_end = min(current_sample + samples_per_chunk_target, total_samples)
            block_len = block_end - block_start

            # 2. Define Loaded Region (Block + Padding)
            load_start = max(0, block_start - pad_samples)
            load_end = min(total_samples, block_end + pad_samples)

            # 3. Load Data from Memmap
            # Note: memmap access is standard numpy slicing
            data_chunk_cpu = data_map[load_start:load_end, :]

            # Convert to Float32 for GPU
            data_chunk_float = data_chunk_cpu.astype(np.float32)

            # 4. Transfer to GPU
            try:
                data_chunk_gpu = cp.asarray(data_chunk_float)
            except Exception as e:
                # Fallback or Error
                free_gpu_memory()
                raise RuntimeError(f"GPU OOM on Chunk {chunk_idx}: {e}")

            # 5. Apply Zero-Phase Filter & Downsample
            # This function now does: sosfilt -> flip -> sosfilt -> flip -> slice
            processed_gpu = apply_filter_and_downsample(
                data_chunk_gpu,
                CUTOFF_FREQUENCY,
                SAMPLE_RATE_ORIGINAL,
                TARGET_SAMPLING_RATE,
            )

            # 6. Trim Padding from Output
            # We need to calculate where the 'valid' block is in the downsampled space

            # Calculate input offsets relative to the loaded chunk
            valid_start_input_rel = block_start - load_start
            valid_end_input_rel = block_end - load_start  # Exclusive

            # Convert these offsets to output (downsampled) space
            # Note: The downsample operation is a slice [::factor]
            # So index i in output corresponds to index i*factor in input
            valid_start_output = int(valid_start_input_rel / downsample_factor)
            valid_end_output = int(valid_end_input_rel / downsample_factor)

            # Extract the valid region
            result_gpu = processed_gpu[valid_start_output:valid_end_output, :]

            # 7. Write to Disk
            result_cpu = cp.asnumpy(result_gpu).astype(np.int16)
            f_out.write(result_cpu.tobytes())

            # Cleanup
            del (
                data_chunk_cpu,
                data_chunk_float,
                data_chunk_gpu,
                processed_gpu,
                result_gpu,
                result_cpu,
            )

            # Progress update
            chunk_idx += 1
            current_sample = block_end

            if chunk_idx % 1 == 0:
                progress = (current_sample / total_samples) * 100
                print(f"Processed chunk {chunk_idx}: {progress:.1f}%")
                free_gpu_memory()

    free_gpu_memory()
    end_time = time.time()
    print(f"\nProcessing complete in {end_time - start_time:.2f} seconds.")
    print(f"Saved to {output_file}")
