import os
import numpy as np
# Import the updated filter functions
from src.utils.filters import low_pass_filter, downsample
from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY

# Use NumPy as default processor
xp = np
GPU_AVAILABLE = False
# Keep track of the CuPy module if available
cp = None

# Try to initialize CuPy
try:
    # Import cupy here
    import cupy
    cp = cupy # Assign to cp variable if successful
    # Test CUDA functionality
    a = cp.array([1, 2, 3])
    a.sum()
    # Simple kernel test
    cp.RawKernel('extern "C" __global__ void example() {}', 'example')()
    xp = cp # Set xp to cupy
    GPU_AVAILABLE = True
    print(f"Using CuPy version: {cp.__version__}")
    print(f"GPU device: {cp.cuda.get_device_id()}")
except ImportError:
    print("CuPy not found, using NumPy instead.")
except Exception as e: # Catch other potential CUDA errors
    print(f"CuPy initialization failed ({type(e).__name__}: {e}), using NumPy instead.")
    cp = None # Ensure cp is None if initialization fails
    xp = np
    GPU_AVAILABLE = False


def free_gpu_memory():
    """Free GPU memory if CuPy is available and initialized."""
    if GPU_AVAILABLE and cp: # Check if cp was successfully imported
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            # print("GPU memory freed.") # Optional: for debugging
        except Exception as e:
            print(f"Error freeing GPU memory: {e}")


def process_chunk(data_chunk_xp, filter_states=None):
    """
    Process a chunk of data (already on CPU/GPU): filter then downsample.
    Relies on filter state propagation for continuity between chunks.

    Args:
        data_chunk_xp: Input data chunk (NumPy or CuPy array, float32, samples x channels)
        filter_states: Initial filter state from the previous chunk (optional)

    Returns:
        tuple: (processed_data (xp array), next_filter_state)
    """
    # Calculate cutoff frequency (ensure it's not too high)
    cutoff = min(TARGET_SAMPLING_RATE * 0.45, CUTOFF_FREQUENCY)

    # 1. Filter first at original sample rate using the appropriate array module (xp)
    # The updated low_pass_filter handles multi-channel directly and accepts xp
    filtered_data, next_filter_state = low_pass_filter(
        data_chunk_xp, cutoff, SAMPLE_RATE_ORIGINAL, xp, initial_zi=filter_states
    )

    # 2. Then downsample using the appropriate array module (xp)
    # The updated downsample handles multi-channel directly and accepts xp
    downsampled_data = downsample(
        filtered_data, TARGET_SAMPLING_RATE, SAMPLE_RATE_ORIGINAL, xp
    )

    return downsampled_data, next_filter_state


def extract_lfp(input_file, output_file, chunk_size, num_channels):
    """
    Extract LFP from raw data file (.dat), handling chunking and GPU acceleration.
    
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
    if GPU_AVAILABLE:
        print(f"Using GPU acceleration via CuPy.")
    else:
        print("Using CPU (NumPy) for processing.")

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

            # Convert to float32 for processing (potential scaling could be added here if needed)
            data_chunk_float = data_chunk_np.astype(np.float32)

            # --- GPU Acceleration ---
            # Transfer data to GPU if available
            if GPU_AVAILABLE:
                try:
                    data_chunk_xp = xp.asarray(data_chunk_float)
                except Exception as e:
                    print(f"Error transferring chunk {chunk_count} to GPU: {e}. Falling back to NumPy for this chunk.")
                    data_chunk_xp = data_chunk_float # Use the NumPy float version
                    # Consider disabling GPU for future chunks if errors persist
            else:
                data_chunk_xp = data_chunk_float # Already a NumPy array

            # --- Process Chunk ---
            # Pass the xp array (CPU or GPU) and current filter state
            lfp_chunk_xp, filter_states = process_chunk(data_chunk_xp, filter_states)

            # --- Data Conversion and Writing ---
            # Convert back to int16 for storage
            # If data is on GPU, transfer back to CPU first
            if GPU_AVAILABLE and xp is cp:
                try:
                    lfp_chunk_np = cp.asnumpy(lfp_chunk_xp)
                except Exception as e:
                     print(f"Error transferring chunk {chunk_count} result from GPU: {e}. Skipping write for this chunk.")
                     continue # Skip writing this chunk if transfer fails
            else:
                lfp_chunk_np = lfp_chunk_xp # Already a NumPy array

            # Ensure it's the correct type before writing
            lfp_data_int16 = lfp_chunk_np.astype(dtype)
            f_out.write(lfp_data_int16.tobytes())

            # --- Cleanup and Progress ---
            # Explicitly delete intermediate arrays to potentially help memory management
            del data_chunk_np, data_chunk_float, data_chunk_xp, lfp_chunk_xp, lfp_chunk_np, lfp_data_int16

            chunk_count += 1
            if chunk_count % 10 == 0: # Report progress every 10 chunks
                progress = (bytes_processed / file_size) * 100
                print(f"Processed {chunk_count} chunks... ({progress:.1f}% complete)")
                # Free GPU memory periodically if using GPU
                if GPU_AVAILABLE and chunk_count % 50 == 0: # Free every 50 chunks
                    free_gpu_memory()

    # Final GPU memory cleanup
    if GPU_AVAILABLE:
        free_gpu_memory()

    print(f"\nLFP extraction complete. Output saved to {output_file}")
