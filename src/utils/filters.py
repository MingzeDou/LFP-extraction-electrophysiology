import numpy as np
import scipy.signal
# Conditionally import cupy and cupyx
try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
    _cupy_available = True
except ImportError:
    _cupy_available = False
    cp = None # Define cp as None if import fails
    cusignal = None

def low_pass_filter(data, cutoff, fs, xp, initial_zi=None):
    """
    Apply a low-pass filter optimized for LFP extraction using the provided array module (xp)
    and leveraging GPU filtering (cupyx.scipy.signal) if xp is cupy.
    Uses a lower order filter with better stability and processes multi-channel data efficiently.
    
    Args:
        data: Input signal (numpy or cupy array, samples x channels)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        xp: Array module (numpy or cupy)
        initial_zi: Initial filter state (optional)

    Returns:
        tuple: (filtered_data (xp array), final_filter_state)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Increase filter order slightly - might avoid edge case in cupyx.scipy.signal with n_sections=1
    filter_order = 4

    # Design filter using SciPy (coefficients are small, CPU calculation is fine)
    sos_np = scipy.signal.butter(filter_order, normal_cutoff, btype='low', output='sos')
    # For order 4, sos_np will have shape (2, 6) -> n_sections = 2

    # --- Filtering Logic ---
    if _cupy_available and xp == cp:
        # --- CuPy Path: Process channel by channel to avoid axis=0 issue ---
        sos_gpu = xp.asarray(sos_np)    # Transfer coefficients to GPU once
        data_xp = xp.asarray(data)      # Ensure data is on GPU

        if data_xp.ndim == 1:
            num_channels = 1
            data_xp = data_xp[:, xp.newaxis] # Reshape to (samples, 1)
        elif data_xp.ndim == 2:
            num_channels = data_xp.shape[1]
        else:
            raise ValueError(f"Input data must be 1D or 2D, got {data_xp.ndim}D")

        # Allocate output array on GPU
        filtered_data = xp.zeros_like(data_xp)
        final_zi_list = [] # Store list of final states per channel

        # Calculate default single-channel zi on CPU and transfer if needed later
        zi_single_np = scipy.signal.sosfilt_zi(sos_np)

        for ch in range(num_channels):
            channel_data = data_xp[:, ch]
            ch_initial_zi = None

            if initial_zi is None:
                # First chunk for this channel, create default zi on GPU
                ch_initial_zi = xp.asarray(np.ascontiguousarray(zi_single_np))
            elif isinstance(initial_zi, list) and ch < len(initial_zi):
                 # Subsequent chunk, use state from previous chunk (already on GPU)
                 ch_initial_zi = initial_zi[ch]
            else:
                 # Fallback if initial_zi format is unexpected
                 print(f"Warning: Unexpected initial_zi format for channel {ch}. Using default.")
                 ch_initial_zi = xp.asarray(np.ascontiguousarray(zi_single_np))

            # Filter single channel on GPU (no axis needed)
            filtered_ch, next_zi_ch = cusignal.sosfilt(sos_gpu, channel_data, zi=ch_initial_zi)
            filtered_data[:, ch] = filtered_ch
            final_zi_list.append(next_zi_ch) # Append GPU array state

        # Return multi-channel GPU array and list of GPU states
        return filtered_data, final_zi_list

    else:
        # --- SciPy Path: Use axis=0 for efficiency ---
        _sosfilt = scipy.signal.sosfilt # Use SciPy's sosfilt
        sos = sos_np                # Use NumPy coefficients

        # Ensure data is NumPy array
        if hasattr(data, 'get'): # Check if it's a CuPy array (e.g., fallback)
             data_np = data.get()
        else:
             data_np = np.asarray(data)

        # Ensure initial_zi is NumPy if provided
        initial_zi_np = None
        if initial_zi is not None:
             if isinstance(initial_zi, list): # Handle case where previous chunk was GPU
                 print("Warning: Switching from GPU state list to CPU state array.")
                 # Attempt to convert list of CuPy arrays back (might be slow)
                 try:
                     initial_zi_np = np.stack([s.get() for s in initial_zi], axis=-1)
                 except Exception as e:
                     print(f"Error converting GPU state list to CPU array: {e}. Using default state.")
                     initial_zi_np = None # Fallback to default
             elif hasattr(initial_zi, 'get'): # Single CuPy array (shouldn't happen with current logic)
                 initial_zi_np = initial_zi.get()
             else: # Assume it's already a NumPy array
                 initial_zi_np = np.asarray(initial_zi)

        # Determine number of channels
        if data_np.ndim == 1:
            num_channels = 1
            data_np = data_np[:, np.newaxis] # Reshape to (samples, 1)
        elif data_np.ndim == 2:
            num_channels = data_np.shape[1]
        else:
            raise ValueError(f"Input data must be 1D or 2D, got {data_np.ndim}D")

        # Initialize filter state if not provided or if conversion failed
        if initial_zi_np is None:
            zi_single_np = scipy.signal.sosfilt_zi(sos_np) # Shape (n_sections, 2)
            zi_np = np.repeat(zi_single_np[:, :, np.newaxis], num_channels, axis=2) # Shape (n_sections, 2, num_channels)
            initial_zi_np = np.ascontiguousarray(zi_np)

        # Apply filter along the time axis (axis=0)
        filtered_data, final_zi = _sosfilt(sos, data_np, axis=0, zi=initial_zi_np)

        # If input was 1D originally, return 1D array
        if num_channels == 1:
           filtered_data = filtered_data.flatten()

        # Return NumPy array and NumPy state array
        return filtered_data, final_zi


def downsample(data, target_fs, original_fs, xp):
    """Downsample data by integer factor using the provided array module (xp)."""
    factor = int(original_fs / target_fs)
    if factor <= 0:
        raise ValueError("Downsampling factor must be positive.")
    if factor == 1:
        return data # No downsampling needed

    data_xp = xp.asarray(data)
    # Slice along the time axis (axis 0)
    return data_xp[::factor, ...] # Use ellipsis for multi-channel compatibility
