import numpy as np
from scipy import signal
# CuPy is imported and managed in the main script (extract_lfp.py)

def low_pass_filter(data, cutoff, fs, xp, initial_zi=None):
    """
    Apply a low-pass filter optimized for LFP extraction using the provided array module (xp).
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
    filter_order = 2 # Lower order for stability with chunking

    # Design filter using SciPy (works for both NumPy and CuPy inputs later)
    sos = signal.butter(filter_order, normal_cutoff, btype='low', output='sos')

    # Ensure data is xp array
    data_xp = xp.asarray(data)

    # Determine number of channels
    if data_xp.ndim == 1:
        num_channels = 1
        # Reshape to 2D for consistent processing
        data_xp = data_xp[:, xp.newaxis]
    else:
        num_channels = data_xp.shape[1]

    # Initialize filter state if not provided
    if initial_zi is None:
        # Create initial state compatible with sosfilt shape requirements
        zi_single = signal.sosfilt_zi(sos) # Shape (n_sections, 2)
        # Repeat state for each channel: shape becomes (n_sections, 2, num_channels)
        initial_zi = xp.asarray(np.repeat(zi_single[:, :, np.newaxis], num_channels, axis=2))

    # Apply filter along the time axis (axis=0)
    # sosfilt expects zi shape (n_sections, 2, n_channels) when axis=0 and data is (samples, n_channels)
    filtered_data, final_zi = signal.sosfilt(sos, data_xp, axis=0, zi=initial_zi)

    # If input was 1D, return 1D array
    if num_channels == 1 and filtered_data.ndim > 1:
       filtered_data = filtered_data.flatten()

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
