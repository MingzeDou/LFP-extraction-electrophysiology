import os
import numpy as np
import time # Add time import
# Import the updated filter functions
from src.utils.filters import low_pass_filter, downsample
from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY

# Import CuPy - Raise error if unavailable
try:
    import cupy as cp
    # Test CUDA functionality
    a = cp.array([1, 2, 3])
    a.sum()
    # Simple kernel test - just check compilation, don't call
    cp.RawKernel('extern "C" __global__ void example() {}', 'example')
    print(f"Using CuPy version: {cp.__version__}")
    print(f"GPU device: {cp.cuda.get_device_id()}")
except ImportError as e:
    raise ImportError(f"CuPy failed to import: {e}. Please install CuPy or use the CPU_process scripts.") from e
except Exception as e: # Catch other potential CUDA errors
    raise RuntimeError(f"CuPy initialization failed ({type(e).__name__}: {e}). Ensure CUDA drivers and toolkit are compatible. Otherwise, use the CPU_process scripts.") from e


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


def process_chunk(data_chunk_gpu, filter_states=None):
    """
    Process a chunk of data on the GPU: filter then downsample.
    Relies on filter state propagation for continuity between chunks.

    Args:
        data_chunk_gpu: Input data chunk (CuPy array, float32, samples x channels)
        filter_states: Initial filter state from the previous chunk (optional, list of CuPy arrays)

    Returns:
        tuple: (processed_data (CuPy array), next_filter_state (list of CuPy arrays))
    """
    # Calculate cutoff frequency (ensure it's not too high)
    cutoff = min(TARGET_SAMPLING_RATE * 0.45, CUTOFF_FREQUENCY)

    # 1. Filter first at original sample rate using CuPy
    # The updated low_pass_filter operates only on GPU
    filtered_data, next_filter_state = low_pass_filter(
        data_chunk_gpu, cutoff, SAMPLE_RATE_ORIGINAL, initial_zi=filter_states
    )

    # 2. Then downsample using CuPy
    # The updated downsample operates only on GPU
    downsampled_data = downsample(
        filtered_data, TARGET_SAMPLING_RATE, SAMPLE_RATE_ORIGINAL
    )

    return downsampled_data, next_filter_state


def extract_lfp(input_file, output_file, chunk_size, num_channels):
    """
    Extract LFP from raw data file (.dat) using GPU acceleration via CuPy.
    Raises errors if CuPy is unavailable or GPU operations fail.

    Args:
        input_file: Path to input raw data file (.dat)
        output_file: Path to output LFP file
        chunk_size: Size of chunks to process in bytes.
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
    print(f"Using GPU acceleration via CuPy.") # Always using GPU now

    # Data type and bytes per sample/frame
    dtype = np.int16
    bytes_per_sample = np.dtype(dtype).itemsize
    bytes_per_frame = num_channels * bytes_per_sample

    # Adjust chunk size to be divisible by bytes_per_frame for clean reshaping
    if chunk_size % bytes_per_frame != 0:
        chunk_size = chunk_size - (chunk_size % bytes_per_frame)
        print(f"Adjusted chunk size to {chunk_size} bytes to be divisible by frame size ({bytes_per_frame} bytes).")

    # Get file size for progress tracking
    file_size = os.path.getsize(input_file)
    bytes_processed = 0

    # Initialize filter state (will be managed by low_pass_filter)
    filter_states = None

    start_time = time.time() # Record start time

    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        chunk_count = 0
        while True:
            # Read a chunk of data
            data_bytes = f_in.read(chunk_size)
            if not data_bytes:
                break # End of file

            bytes_processed += len(data_bytes)

            # Convert buffer to numpy array
            data_chunk_np = np.frombuffer(data_bytes, dtype=dtype)

            # Handle incomplete frames at the very end of the file
            remainder = len(data_chunk_np) % num_channels
            if remainder != 0:
                print(f"Warning: Trimming {remainder} samples from final incomplete frame.")
                data_chunk_np = data_chunk_np[:-remainder]

            # Reshape to (samples, channels)
            data_chunk_np = data_chunk_np.reshape(-1, num_channels)

            # --- Check for empty chunk after reshape ---
            if data_chunk_np.shape[0] == 0:
                print(f"Warning: Skipping empty chunk {chunk_count}.")
                chunk_count += 1 # Increment chunk count even if skipped
                continue # Skip processing and read next chunk

            # Convert to float32 for processing (potential scaling could be added here if needed)
            data_chunk_float = data_chunk_np.astype(np.float32)

            # --- GPU Acceleration ---
            # Transfer data to GPU
            try:
                data_chunk_gpu = cp.asarray(data_chunk_float)
            except Exception as e:
                raise RuntimeError(f"Error transferring chunk {chunk_count} to GPU: {e}. "
                                   "Check GPU memory and compatibility. Consider using the CPU_process scripts.") from e

            # --- Process Chunk ---
            # Pass the GPU array and current filter state
            lfp_chunk_gpu, filter_states = process_chunk(data_chunk_gpu, filter_states)

            # --- Data Conversion and Writing ---
            # Transfer result back to CPU
            try:
                lfp_chunk_np = cp.asnumpy(lfp_chunk_gpu)
            except Exception as e:
                 raise RuntimeError(f"Error transferring chunk {chunk_count} result from GPU: {e}. "
                                    "Check GPU memory. Consider using the CPU_process scripts.") from e

            # Ensure it's the correct type before writing
            lfp_data_int16 = lfp_chunk_np.astype(dtype)
            f_out.write(lfp_data_int16.tobytes())

            # --- Cleanup and Progress ---
            # Explicitly delete intermediate arrays to potentially help memory management
            # Note: CuPy arrays (data_chunk_gpu, lfp_chunk_gpu) should be garbage collected
            del data_chunk_np, data_chunk_float, data_chunk_gpu, lfp_chunk_gpu, lfp_chunk_np, lfp_data_int16

            chunk_count += 1
            if chunk_count % 10 == 0: # Report progress every 10 chunks
                progress = (bytes_processed / file_size) * 100
                print(f"Processed {chunk_count} chunks... ({progress:.1f}% complete)")
                # Free GPU memory periodically
                if chunk_count % 50 == 0: # Free every 50 chunks
                    free_gpu_memory()

    # Final GPU memory cleanup
    free_gpu_memory()

    end_time = time.time() # Record end time
    duration = end_time - start_time
    print(f"\nTotal processing time: {duration:.2f} seconds") # Print duration

    print(f"LFP extraction complete. Output saved to {output_file}")
