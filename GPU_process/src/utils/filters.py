import numpy as np
import scipy.signal # Keep for filter design
# Import cupy and cupyx - Raise error if unavailable
try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
    # Import lfilter_zi utility, handle potential changes in cupy versions
    try:
        from cupyx.scipy.signal._filtering import lfilter_zi
    except ImportError:
        try:
            # Older location? Check if this exists
            from cupyx.scipy.signal._signaltools import lfilter_zi
        except ImportError:
            # If cupyx version not found, it's an error for GPU-only path
            raise ImportError("cupyx.scipy.signal lfilter_zi helper not found. Cannot proceed with GPU filtering.")
except ImportError as e:
    raise ImportError(f"CuPy or CuPyX components failed to import: {e}. Please install CuPy or use the CPU_process scripts.") from e

def low_pass_filter(data_gpu, cutoff, fs, initial_zi=None):
    """
    Apply a low-pass filter optimized for LFP extraction using CuPy.
    Uses a lower order filter with better stability and processes multi-channel data efficiently.
    This function requires CuPy and operates solely on the GPU.

    Args:
        data_gpu: Input signal (cupy array, samples x channels)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        initial_zi: Initial filter state (optional). List of 1D CuPy zi arrays (one per channel).

    Returns:
        tuple: (filtered_data (cupy array), final_filter_state (list of cupy arrays))
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    filter_order = 4 # Using 4th order filter

    # Design filter using SciPy (coefficients are small, CPU calculation is fine)
    sos_np = scipy.signal.butter(filter_order, normal_cutoff, btype='low', output='sos')

    # --- CuPy Path Only ---
    # Convert SOS to BA form (on CPU)
    b_np, a_np = scipy.signal.sos2tf(sos_np)
    b_gpu = cp.asarray(b_np) # Transfer BA coeffs to GPU
    a_gpu = cp.asarray(a_np)

    # data_gpu is assumed to be a CuPy array already
    if data_gpu.ndim == 1:
        num_channels = 1
        data_gpu = data_gpu[:, cp.newaxis] # Reshape to (samples, 1)
    elif data_gpu.ndim == 2:
        num_channels = data_gpu.shape[1]
    else:
        raise ValueError(f"Input data must be 1D or 2D CuPy array, got {data_gpu.ndim}D")

    # Allocate output array on GPU
    filtered_data = cp.zeros_like(data_gpu)
    final_zi_list = [] # Store list of final states per channel

    # Calculate default single-channel lfilter zi *on GPU* using GPU coefficients
    # Note: lfilter_zi requires 'a' and 'b' coefficients
    zi_single_gpu = lfilter_zi(b_gpu, a_gpu) # Use the imported lfilter_zi with GPU arrays

    for ch in range(num_channels):
        channel_data = data_gpu[:, ch] # Get single channel (1D GPU array)
        ch_initial_zi = None

        # Check if channel_data is empty before proceeding
        if channel_data.size == 0:
             print(f"Warning: Skipping empty channel {ch} in GPU filter path.")
             # Need to append a valid state placeholder if skipping
             # Create default zi on GPU for the next iteration
             final_zi_list.append(zi_single_gpu * 0.0) # Use 0 state
             continue

        if initial_zi is None:
            # First chunk for this channel, use default zi calculated on GPU, scaled by first sample
            ch_initial_zi = zi_single_gpu * channel_data[0]
        elif isinstance(initial_zi, list) and ch < len(initial_zi):
             # Subsequent chunk, use state from previous chunk (already on GPU)
             ch_initial_zi = initial_zi[ch]
        else:
             # Should not happen if called correctly from extract_lfp
             print(f"Warning: Unexpected initial_zi format for channel {ch} in GPU path. Using default.")
             ch_initial_zi = zi_single_gpu * channel_data[0]

        # Filter single channel on GPU using lfilter
        try:
            filtered_ch, next_zi_ch = cusignal.lfilter(b_gpu, a_gpu, channel_data, zi=ch_initial_zi)
            filtered_data[:, ch] = filtered_ch
            final_zi_list.append(next_zi_ch) # Append GPU array state
        except Exception as e:
                # Raise error instead of fallback
                raise RuntimeError(f"Error during cusignal.lfilter for channel {ch}: {e}. "
                                   f"Input shape: {channel_data.shape}, zi shape: {ch_initial_zi.shape if ch_initial_zi is not None else 'None'}. "
                                   "Consider using the CPU_process scripts.") from e

        # Return multi-channel GPU array and list of GPU states
    return filtered_data, final_zi_list


def downsample(data_gpu, target_fs, original_fs):
    """
    Downsample data by integer factor using CuPy, applying an anti-aliasing filter.
    This function requires CuPy and operates solely on the GPU.

    Args:
        data_gpu: Input signal (cupy array, samples x channels)
        target_fs: Target sampling frequency in Hz
        original_fs: Original sampling frequency in Hz

    Returns:
        Downsampled data (cupy array, samples x channels)
    """
    factor = int(original_fs / target_fs)
    if factor <= 0:
        raise ValueError(f"Downsampling factor must be positive integer, calculated {factor} from {original_fs}/{target_fs}")
    if factor == 1:
        return data_gpu # No downsampling needed

    # data_gpu is assumed to be a CuPy array already
    # Check for empty input
    if data_gpu.shape[0] == 0:
        # Return an empty array with the correct number of channels and type
        return cp.zeros((0, data_gpu.shape[1]) if data_gpu.ndim == 2 else (0,), dtype=data_gpu.dtype)

    # --- GPU Path Only ---
    try:
        # Use CuPy's decimate function (cupyx.scipy.signal.decimate)
        # axis=0 assumes time dimension is the first axis (samples, channels)
        # zero_phase=True provides linear phase filtering but is slower
        downsampled_data = cusignal.decimate(data_gpu, factor, axis=0, zero_phase=True)
        return downsampled_data
    except Exception as e:
        # Raise error instead of fallback
        raise RuntimeError(f"Error during cusignal.decimate: {e}. "
                           "Consider using the CPU_process scripts.") from e
