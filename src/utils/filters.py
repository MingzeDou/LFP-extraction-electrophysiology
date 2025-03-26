import numpy as np
from scipy import signal

# Check for CuPy availability
using_cupy = False
try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    test_array.sum()
    using_cupy = True
    print("Using CuPy for GPU acceleration")
except (ImportError, RuntimeError):
    print("CuPy not available or CUDA error, using NumPy instead")

def low_pass_filter(data, cutoff, fs, initial_zi=None):
    """
    Apply a low-pass filter optimized for LFP extraction.
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
        if using_cupy:
            data_cpu = cp.asnumpy(data)
            
            # Get initial state
            if initial_zi is None:
                zi = signal.sosfilt_zi(sos)
                zi_expanded = np.tile(zi, (1, 1)).copy()
            else:
                zi_expanded = initial_zi
            
            filtered, next_zi = signal.sosfilt(sos, data_cpu, zi=zi_expanded)
            return cp.array(filtered), next_zi
        else:
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
    """Downsample data by integer factor"""
    factor = int(original_fs / target_fs)
    return data[::factor]

def save_lfp_data(filename, lfp_data):
    """Save LFP data to binary file"""
    lfp_data.tofile(filename)