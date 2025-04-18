import numpy as np
from scipy import signal

# Using NumPy for CPU processing
print("Using NumPy for CPU processing")

def low_pass_filter(data, cutoff, fs, initial_zi=None):
    """
    Apply a low-pass filter optimized for LFP extraction using NumPy/SciPy.
    Uses a lower order filter with better stability.
    
    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        initial_zi: Initial filter state (optional)
        
    Returns:
        tuple: (filtered_data, final_filter_state)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    filter_order = 2
    
    # Create a Butterworth filter
    sos = signal.butter(filter_order, normal_cutoff, btype='low', output='sos')
    
    if len(data.shape) > 1 and data.shape[1] > 1:
        # Handle multi-channel data
        channels = data.shape[1]
        result = np.zeros_like(data)
        final_zi = []
        
        for ch in range(channels):
            channel_data = data[:, ch]
            
            # Get initial state for this channel
            if initial_zi is None:
                zi = signal.sosfilt_zi(sos)
                zi_ch = np.tile(zi, (1, 1)).copy()
            else:
                zi_ch = initial_zi[ch]
            
            # No padding needed when using filter states properly
            filtered, next_zi = signal.sosfilt(sos, channel_data, zi=zi_ch)
            result[:, ch] = filtered
            final_zi.append(next_zi)
        
        return result, final_zi
    else:
        # Single channel data
        # Get initial state
        if initial_zi is None:
            zi = signal.sosfilt_zi(sos)
            zi_expanded = np.tile(zi, (1, 1)).copy()
        else:
            zi_expanded = initial_zi
        
        filtered, next_zi = signal.sosfilt(sos, data, zi=zi_expanded)
        return filtered, next_zi

def multi_stage_filter(data, cutoff, fs, initial_zi=None):
    """Cascade of filters for better stability"""
    # First stage - higher cutoff
    first_cutoff = min(cutoff * 2, fs * 0.45)  
    filtered, zi1 = low_pass_filter(data, first_cutoff, fs, initial_zi)
    
    # Second stage - target cutoff
    filtered, zi2 = low_pass_filter(filtered, cutoff, fs, None)
    
    return filtered, zi1  # Return first stage filter state

def downsample(data, target_fs, original_fs):
    """Downsample data using scipy.signal.decimate for anti-aliasing."""
    factor = int(original_fs / target_fs)
    if factor <= 1:
        return data  # No downsampling needed or invalid factor
    
    # Ensure data is NumPy array for decimate
    data_np = np.asarray(data) # Ensure it's a NumPy array

    # Decimate along the time axis (axis=0)
    downsampled_data = signal.decimate(data_np, factor, axis=0, ftype='fir', zero_phase=True)
    
    return downsampled_data

def save_lfp_data(filename, lfp_data):
    """Save LFP data to binary file"""
    lfp_data.tofile(filename)
