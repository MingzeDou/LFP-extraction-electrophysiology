import numpy as np
import scipy.signal
# Conditionally import cupy and cupyx
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
            # Fallback to scipy's version if cupyx version not found
            print("Warning: cupyx.scipy.signal.lfilter_zi not found, using scipy.signal.lfilter_zi.")
            from scipy.signal import lfilter_zi
    _cupy_available = True
except ImportError:
    _cupy_available = False
    cp = None # Define cp as None if import fails
    cusignal = None
    lfilter_zi = scipy.signal.lfilter_zi # Use scipy's version if cupy not available

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
        initial_zi: Initial filter state (optional). For CuPy lfilter path, this is a list of 1D zi arrays. For SciPy sosfilt path, this is a 3D zi array.

    Returns:
        tuple: (filtered_data (xp array), final_filter_state)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Using 4th order filter
    filter_order = 4

    # Design filter using SciPy (coefficients are small, CPU calculation is fine)
    sos_np = scipy.signal.butter(filter_order, normal_cutoff, btype='low', output='sos')

    # --- Filtering Logic ---
    if _cupy_available and xp == cp:
        # --- CuPy Path: Use lfilter channel-by-channel as sosfilt seems buggy ---
        # Convert SOS to BA form (on CPU)
        b_np, a_np = scipy.signal.sos2tf(sos_np)
        b_gpu = xp.asarray(b_np) # Transfer BA coeffs to GPU
        a_gpu = xp.asarray(a_np)

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
        final_zi_list = [] # Store list of final states per channel (lfilter state shape is different)

        # Calculate default single-channel lfilter zi on CPU
        # Note: lfilter_zi requires 'a' and 'b' coefficients
        zi_single_np = lfilter_zi(b_np, a_np) # Use the imported lfilter_zi

        for ch in range(num_channels):
            channel_data = data_xp[:, ch] # Get single channel (1D GPU array)
            ch_initial_zi = None

            # Check if channel_data is empty before proceeding
            if channel_data.size == 0:
                 print(f"Warning: Skipping empty channel {ch} in CuPy filter path.")
                 # Need to append a valid state placeholder if skipping
                 # Create default zi on GPU for the next iteration
                 default_zi_gpu = xp.asarray(np.ascontiguousarray(zi_single_np)) * 0.0 # Use 0 if first sample not available
                 final_zi_list.append(default_zi_gpu)
                 continue

            if initial_zi is None:
                # First chunk for this channel, create default zi on GPU, scaled by first sample
                ch_initial_zi = xp.asarray(np.ascontiguousarray(zi_single_np)) * channel_data[0]
            elif isinstance(initial_zi, list) and ch < len(initial_zi):
                 # Subsequent chunk, use state from previous chunk (already on GPU)
                 ch_initial_zi = initial_zi[ch]
            else:
                 # Fallback if initial_zi format is unexpected
                 print(f"Warning: Unexpected initial_zi format for channel {ch}. Using default.")
                 ch_initial_zi = xp.asarray(np.ascontiguousarray(zi_single_np)) * channel_data[0]

            # Filter single channel on GPU using lfilter
            try:
                filtered_ch, next_zi_ch = cusignal.lfilter(b_gpu, a_gpu, channel_data, zi=ch_initial_zi)
                filtered_data[:, ch] = filtered_ch
                final_zi_list.append(next_zi_ch) # Append GPU array state
            except ValueError as e:
                print(f"Error during cusignal.lfilter for channel {ch}: {e}")
                print(f"  Input shape: {channel_data.shape}, zi shape: {ch_initial_zi.shape if ch_initial_zi is not None else 'None'}")
                # Fallback: write zeros and default state? Or raise error?
                # For now, let's write zeros and append default state to avoid crashing
                filtered_data[:, ch] = 0
                default_zi_gpu = xp.asarray(np.ascontiguousarray(zi_single_np)) * 0.0
                final_zi_list.append(default_zi_gpu)


        # Return multi-channel GPU array and list of GPU states
        return filtered_data, final_zi_list

    else:
        # --- SciPy Path: Use sosfilt with axis=0 for efficiency and stability ---
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
                     # Ensure states are NumPy before stacking
                     np_states = [s.get() if hasattr(s, 'get') else np.asarray(s) for s in initial_zi]
                     # lfilter state is 1D per channel, stack along a new axis
                     initial_zi_np = np.stack(np_states, axis=-1) # Shape (order, n_channels)
                 except Exception as e:
                     print(f"Error converting GPU state list to CPU array: {e}. Using default state.")
                     initial_zi_np = None # Fallback to default
             elif hasattr(initial_zi, 'get'): # Single CuPy array (sosfilt state)
                 initial_zi_np = initial_zi.get()
             else: # Assume it's already a NumPy array (sosfilt state)
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
        # This needs to be sosfilt state for the SciPy path
        if initial_zi_np is None or initial_zi_np.ndim != 3: # Check if state is not suitable for sosfilt
            if initial_zi_np is not None:
                 print("Warning: Provided initial state not suitable for sosfilt. Using default.")
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
