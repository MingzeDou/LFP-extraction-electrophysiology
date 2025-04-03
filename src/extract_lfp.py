import os
import numpy as np
from typing import Optional, Tuple, Union, Any
import logging

# Import necessary modules from filters.py and config
# Setup logger *before* potential import errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.utils.filters import low_pass_filter, downsample, cp, using_cupy as GPU_AVAILABLE, Array, ArrayModule
    from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY
except ImportError as e:
    logger.error(f"Error importing dependencies: {e}. Ensure src is in the Python path.")
    # Define fallbacks if imports fail, though this indicates an environment issue
    GPU_AVAILABLE = False # type: ignore
    cp = np
    Array = np.ndarray
    ArrayModule = Any # type: ignore
    SAMPLE_RATE_ORIGINAL = 30000
    TARGET_SAMPLING_RATE = 1250
    CUTOFF_FREQUENCY = 450
    def low_pass_filter(data, cutoff, fs, initial_zi=None): return data, None
    def downsample(data, target_fs, original_fs): return data[::int(original_fs/target_fs)]


# Get the correct array module (NumPy or CuPy)
xp: ArrayModule = cp # xp is now the potentially imported cupy module, or numpy if cupy failed

def free_gpu_memory() -> None:
    """Attempt to free GPU memory pools if CuPy is available and initialized.

    Checks for the default memory pool and pinned memory pool and calls
    `free_all_blocks()` on them. Catches potential exceptions during cleanup.
    """
    if GPU_AVAILABLE and hasattr(cp, 'get_default_memory_pool'):
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks() # type: ignore
        except Exception as e:
            logger.warning(f"Error freeing GPU memory: {e}")


def process_chunk(data_chunk: Array, filter_state: Optional[Array] = None, overlap_samples_to_trim: int = 0) -> Tuple[np.ndarray, Optional[Array]]:
    """Process a single chunk of data: filter, trim overlap, and downsample.

    Handles both NumPy and CuPy arrays for input `data_chunk` and `filter_state`
    by leveraging the `low_pass_filter` function's backend detection.

    Parameters
    ----------
    data_chunk : Array
        Input data chunk, expected shape (n_samples, n_channels).
        Should be float32 type. Can be NumPy or CuPy array.
    filter_state : Optional[Array], optional
        Filter state from the previous chunk. Must match the array type of
        `data_chunk`. Defaults to None.
    overlap_samples_to_trim : int, optional
        Number of samples (at the original sampling rate) corresponding to
        the overlap prepended to this chunk, which should be trimmed *after*
        filtering but *before* downsampling. Defaults to 0.

    Returns
    -------
    lfp_chunk_np : np.ndarray
        The processed LFP data for this chunk as a NumPy float32 array.
    next_filter_state : Optional[Array]
        The final filter state from this chunk, ready to be passed to the
        next chunk. Type matches the input `data_chunk`. Returns None if
        filtering fails.

    Raises
    ------
    ValueError
        If the calculated downsampling factor is not an integer.
    """
    # Determine parameters
    if SAMPLE_RATE_ORIGINAL % TARGET_SAMPLING_RATE != 0:
         raise ValueError(f"Downsampling factor ({SAMPLE_RATE_ORIGINAL}/{TARGET_SAMPLING_RATE}) must be an integer.")
    downsampling_factor: int = int(SAMPLE_RATE_ORIGINAL / TARGET_SAMPLING_RATE)
    # Ensure cutoff is safe for the target rate (Nyquist)
    cutoff: float = min(CUTOFF_FREQUENCY, TARGET_SAMPLING_RATE * 0.45)

    # 1. Filter first (handles NumPy/CuPy automatically based on data_chunk type)
    # The low_pass_filter function now processes all channels in parallel.
    filtered_data: Array
    next_filter_state: Optional[Array]
    filtered_data, next_filter_state = low_pass_filter(
        data_chunk, cutoff, SAMPLE_RATE_ORIGINAL, initial_zi=filter_state
    )

    # 2. Trim overlap *from the filtered data* before downsampling
    # Note: Overlap was added before filtering, so we trim it after filtering.
    if overlap_samples_to_trim > 0:
        # Ensure trimming doesn't exceed array bounds
        if overlap_samples_to_trim >= filtered_data.shape[0]:
             logger.warning(f"Overlap trim ({overlap_samples_to_trim}) >= chunk size ({filtered_data.shape[0]}). Returning empty array for this chunk.")
             # Return empty array of correct type and shape
             xp_mod = cp.get_array_module(filtered_data) # type: ignore
             empty_shape = (0,) + filtered_data.shape[1:]
             return xp_mod.zeros(empty_shape, dtype=np.float32), next_filter_state
        filtered_data = filtered_data[overlap_samples_to_trim:, :]

    # 3. Downsample
    # Downsample function currently requires NumPy input.
    # TODO: Consider making downsample handle CuPy arrays if performance critical.
    if GPU_AVAILABLE and isinstance(filtered_data, cp.ndarray):
        filtered_data_np: np.ndarray = cp.asnumpy(filtered_data)
    else:
        # Already NumPy or filtering was done on CPU
        filtered_data_np = filtered_data # type: ignore

    lfp_chunk_np: np.ndarray = downsample(filtered_data_np, TARGET_SAMPLING_RATE, SAMPLE_RATE_ORIGINAL)

    # Ensure output is float32 numpy array (downsample should return np)
    if lfp_chunk_np.dtype != np.float32:
         lfp_chunk_np = lfp_chunk_np.astype(np.float32)

    return lfp_chunk_np, next_filter_state


def extract_lfp(input_file: str, output_file: str, chunk_size: int, num_channels: int = 384) -> None:
    """Extract LFP data from a binary file with filtering and downsampling.

    Reads a raw binary data file (int16 channels multiplexed), applies a
    low-pass filter, downsamples the data, and saves the result as a binary
    int16 file. Processes the file in chunks to manage memory usage, optionally
    using GPU acceleration via CuPy if available.

    Parameters
    ----------
    input_file : str
        Path to the input raw data file (.dat). Data is expected to be int16,
        multiplexed channel-wise (sample1_ch1, sample1_ch2, ..., sample2_ch1,...).
    output_file : str
        Path to save the output LFP file (.lfp). Data will be saved as int16.
    chunk_size : int
        Approximate size of data chunks to process in bytes. Adjust based on
        available CPU/GPU memory. Will be aligned to frame boundaries.
    num_channels : int, optional
        Number of channels in the recording. Defaults to 384.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the calculated chunk size is too small for a single frame, or if
        the downsampling factor is not an integer.
    Exception
        Propagates exceptions occurring during file I/O or processing.
    """
    # Validate inputs
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Starting LFP extraction:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Num channels: {num_channels}")
    logger.info(f"  Chunk size: {chunk_size} bytes")

    # --- Validate Configuration ---
    if SAMPLE_RATE_ORIGINAL % TARGET_SAMPLING_RATE != 0:
        raise ValueError(f"TARGET_SAMPLING_RATE ({TARGET_SAMPLING_RATE} Hz) must evenly divide SAMPLE_RATE_ORIGINAL ({SAMPLE_RATE_ORIGINAL} Hz).")
    if CUTOFF_FREQUENCY >= TARGET_SAMPLING_RATE / 2.0:
        raise ValueError(f"CUTOFF_FREQUENCY ({CUTOFF_FREQUENCY} Hz) must be less than the Nyquist frequency of the target rate ({TARGET_SAMPLING_RATE / 2.0} Hz).")

    # --- Configuration and Setup ---
    BYTES_PER_SAMPLE: int = 2 # int16
    bytes_per_frame: int = num_channels * BYTES_PER_SAMPLE

    # Ensure chunk_size is a multiple of bytes_per_frame for clean reads
    if chunk_size < bytes_per_frame:
         raise ValueError(f"Chunk size ({chunk_size} bytes) must be at least the size of one frame ({bytes_per_frame} bytes).")
    chunk_size = (chunk_size // bytes_per_frame) * bytes_per_frame

    # Overlap: Use samples corresponding to ~0.5 seconds at original rate
    # This helps avoid edge artifacts from filtering chunks independently.
    overlap_samples: int = int(0.5 * SAMPLE_RATE_ORIGINAL)
    # overlap_bytes: int = overlap_samples * bytes_per_frame # Not directly used

    # Scaling factor for int16 to float32 conversion (usually 1 if no voltage scaling needed)
    # If your recording system saves data scaled by a factor (e.g., microvolts per bit),
    # you might adjust this. For now, assume direct int16 values.
    scaling_factor: float = 1.0

    # Get file size for progress tracking
    try:
        file_size: int = os.path.getsize(input_file)
    except OSError as e:
        raise FileNotFoundError(f"Could not get size of input file {input_file}: {e}")

    bytes_processed: int = 0
    filter_state: Optional[Array] = None # Unified filter state for all channels

    # Memory mapping can be efficient but adds complexity. Sticking to read() for now.
    logger.info(f"Processing parameters:")
    logger.info(f"  Chunk size: {chunk_size / (1024*1024):.2f} MB ({chunk_size} bytes)")
    logger.info(f"  Overlap samples: {overlap_samples}")
    logger.info(f"  GPU Available: {GPU_AVAILABLE}")
    logger.info(f"  Original FS: {SAMPLE_RATE_ORIGINAL} Hz, Target FS: {TARGET_SAMPLING_RATE} Hz, Cutoff: {CUTOFF_FREQUENCY} Hz")

    # --- Main Processing Loop ---
    gpu_processing_failed = False # Flag to track if GPU failed once
    try:
        with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            chunk_count: int = 0
            # Store the overlapping part of the *previous* raw chunk (as numpy float32)
            previous_overlap_np: Optional[np.ndarray] = None

            while True:
                # --- Read Data ---
                current_chunk_bytes: bytes = f_in.read(chunk_size)
                if not current_chunk_bytes:
                    break # End of file reached

                bytes_processed += len(current_chunk_bytes)

                # Convert raw bytes to NumPy int16 array
                current_chunk_np_int16: np.ndarray = np.frombuffer(current_chunk_bytes, dtype=np.int16)

                # Handle incomplete frames at the very end of the file
                remainder: int = len(current_chunk_np_int16) % num_channels
                if remainder != 0:
                    # This should only happen on the last partial read if file size is not multiple of frame size
                    logger.warning(f"Trimming {remainder} trailing samples from the end of the file (incomplete frame).")
                    current_chunk_np_int16 = current_chunk_np_int16[:-remainder]

                if current_chunk_np_int16.size == 0:
                    continue # Skip if trimming resulted in empty chunk

                # Reshape and convert to float32 for processing
                current_chunk_np: np.ndarray = current_chunk_np_int16.reshape(-1, num_channels).astype(np.float32) * scaling_factor

                # --- Overlap Handling ---
                processing_chunk_np: np.ndarray
                overlap_to_trim_in_samples: int
                if previous_overlap_np is not None:
                    # Prepend the overlap from the previous chunk
                    processing_chunk_np = np.vstack((previous_overlap_np, current_chunk_np))
                    overlap_to_trim_in_samples = previous_overlap_np.shape[0]
                else:
                    # First chunk, no overlap to prepend
                    processing_chunk_np = current_chunk_np
                    overlap_to_trim_in_samples = 0

                # Store the *last* part of the current *raw* chunk for the *next* iteration's overlap
                # Ensure we don't try to slice more than available
                current_overlap_samples = min(overlap_samples, processing_chunk_np.shape[0])
                if current_overlap_samples > 0:
                    previous_overlap_np = processing_chunk_np[-current_overlap_samples:, :].copy()
                else:
                    # Chunk was smaller than overlap, store the whole chunk
                    previous_overlap_np = processing_chunk_np.copy()


                # --- Process Chunk ---
                lfp_chunk_np: np.ndarray
                processing_chunk: Array # Can be np or cp array

                # Decide whether to attempt GPU processing
                attempt_gpu = GPU_AVAILABLE and not gpu_processing_failed

                if attempt_gpu:
                    try:
                        # Transfer data and state (if exists) to GPU
                        processing_chunk = xp.asarray(processing_chunk_np) # type: ignore
                        filter_state_gpu = xp.asarray(filter_state) if filter_state is not None else None # type: ignore

                        lfp_chunk_np, filter_state = process_chunk(
                            processing_chunk, filter_state_gpu, overlap_to_trim_in_samples
                        )
                        # Ensure filter state is back on CPU for the next iteration's check/transfer
                        if filter_state is not None:
                              filter_state = cp.asnumpy(filter_state) # type: ignore

                    except Exception as e:
                         # Catch potential GPU errors (e.g., OOM, state shape error from low_pass_filter)
                         logger.error(f"Error during GPU processing: {e}. Falling back to CPU for this chunk and subsequent chunks.", exc_info=False) # Log less verbosely now
                         gpu_processing_failed = True # Set flag to prevent future GPU attempts
                         free_gpu_memory() # Attempt to clear GPU memory

                         # Fallback to CPU processing for this chunk
                         logger.info("Retrying chunk processing on CPU...")
                         processing_chunk = processing_chunk_np # Use the NumPy version
                         # Reset filter state for the CPU fallback as the GPU state was invalid/caused error
                         filter_state = None
                         lfp_chunk_np, filter_state = process_chunk(
                             processing_chunk, filter_state, overlap_to_trim_in_samples
                         )
                         # Note: filter_state will now be a NumPy array if successful

                # If GPU wasn't attempted or failed *initially*, process on CPU
                elif not attempt_gpu: # This 'elif' handles the case where GPU was never tried for the chunk
                    # Ensure filter state is NumPy if coming from a previous (successful) GPU chunk's state
                    if filter_state is not None and not isinstance(filter_state, np.ndarray):
                         logger.debug("Converting previous filter state from CuPy to NumPy for CPU processing.")
                         try:
                              filter_state = cp.asnumpy(filter_state) # type: ignore
                         except Exception as conv_e:
                              logger.error(f"Could not convert filter state to NumPy for CPU processing: {conv_e}. Resetting state.")
                              filter_state = None

                    processing_chunk = processing_chunk_np
                    lfp_chunk_np, filter_state = process_chunk(
                        processing_chunk, filter_state, overlap_to_trim_in_samples
                    )
                    # Note: filter_state will be NumPy array here


                # --- Write Output ---
                if lfp_chunk_np.size > 0:
                    # Convert processed chunk (float32) back to int16 for storage
                    # Clamp data to int16 range before converting to avoid overflow/wrap-around
                    np.clip(lfp_chunk_np, -32768, 32767, out=lfp_chunk_np)
                    lfp_chunk_int16: np.ndarray = lfp_chunk_np.astype(np.int16)

                    f_out.write(lfp_chunk_int16.tobytes())

                # --- Progress Reporting ---
                chunk_count += 1
                if chunk_count % 10 == 0: # Report progress every 10 chunks
                    progress = (bytes_processed / file_size) * 100 if file_size > 0 else 0
                    # Use standard logging for progress, avoiding \r for better compatibility
                    logger.info(f"Processed chunk {chunk_count} ({bytes_processed / (1024*1024):.2f} / {file_size / (1024*1024):.2f} MB) [{progress:.1f}%]...")

                # Free GPU memory periodically (optional, depends on memory pressure)
                if GPU_AVAILABLE and chunk_count % 50 == 0: # Every 50 chunks
                    free_gpu_memory()

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {input_file}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise e
    finally:
        # Final GPU memory cleanup
        if GPU_AVAILABLE:
            logger.debug("Performing final GPU memory cleanup.")
            free_gpu_memory()
        logger.info(f"LFP extraction finished. Output saved to {output_file}")
