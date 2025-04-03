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

    # --- Select filtering function and prepare inputs based on xp ---
    if _cupy_available and xp == cp:
        _sosfilt = cusignal.sosfilt # Use CuPy's sosfilt
        sos = xp.asarray(sos_np)    # Transfer coefficients to GPU
        data_xp = xp.asarray(data)  # Ensure data is on GPU
        # Ensure initial_zi is on GPU if provided
        if initial_zi is not None:
            initial_zi = xp.asarray(initial_zi)
    else:
        _sosfilt = scipy.signal.sosfilt # Use SciPy's sosfilt
        sos = sos_np                # Use NumPy coefficients
        # Ensure data is NumPy array (might be CuPy if fallback occurred)
        if hasattr(data, 'get'): # Check if it's a CuPy array
             data_xp = data.get()
        else:
             data_xp = np.asarray(data)
        # Ensure initial_zi is NumPy if provided
        if initial_zi is not None and hasattr(initial_zi, 'get'):
            initial_zi = initial_zi.get()
        elif initial_zi is not None:
            initial_zi = np.asarray(initial_zi)

    # Determine number of channels (use shape of data_xp)
    if data_xp.ndim == 1:
        num_channels = 1
        # Reshape to 2D for consistent processing (samples, 1)
        data_xp = data_xp[:, xp.newaxis]
    elif data_xp.ndim == 2:
        num_channels = data_xp.shape[1]
    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data_xp.ndim}D")

    # Initialize filter state if not provided, ensuring it's on the correct device (CPU/GPU)
    if initial_zi is None:
        # Calculate zi using SciPy on CPU first
        zi_single_np = scipy.signal.sosfilt_zi(sos_np) # Shape (n_sections, 2)
        # Repeat state for each channel
        zi_np = np.repeat(zi_single_np[:, :, np.newaxis], num_channels, axis=2) # Shape (n_sections, 2, num_channels)
        # Ensure C-contiguity before transferring to GPU, as CuPy might be sensitive to memory layout
        initial_zi = xp.asarray(np.ascontiguousarray(zi_np))

    # Apply filter along the time axis (axis=0)
    # Both scipy.signal.sosfilt and cupyx.scipy.signal.sosfilt expect
    # zi shape (n_sections, 2, n_channels) when axis=0 and data is (samples, n_channels)
    filtered_data, final_zi = _sosfilt(sos, data_xp, axis=0, zi=initial_zi)

    # If input was 1D originally, return 1D array
    # Check original data ndim or num_channels, not filtered_data.ndim
    if num_channels == 1:
       filtered_data = filtered_data.flatten()

    return filtered_data, final_zi # final_zi will be on the same device as filtered_data


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
