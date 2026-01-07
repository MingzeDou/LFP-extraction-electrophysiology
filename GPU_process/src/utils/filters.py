import numpy as np
import scipy.signal  # Keep for filter design

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
            raise ImportError(
                "cupyx.scipy.signal lfilter_zi helper not found. Cannot proceed with GPU filtering."
            )
except ImportError as e:
    raise ImportError(
        f"CuPy or CuPyX components failed to import: {e}. Please install CuPy or use the CPU_process scripts."
    ) from e

import numpy as np
import scipy.signal

# Import cupy and cupyx - Raise error if unavailable
try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
except ImportError as e:
    raise ImportError(
        f"CuPy or CuPyX components failed to import: {e}. Please install CuPy or use the CPU_process scripts."
    ) from e


def apply_filter_and_downsample(data_gpu, cutoff, fs, target_fs):
    """
    Apply a zero-phase low-pass filter and downsample data on GPU.
    Performs forward-backward filtering to ensure zero phase shift.

    Args:
        data_gpu: Input signal (cupy array, samples x channels)
        cutoff: Cutoff frequency in Hz
        fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz

    Returns:
        cupy array: Filtered and downsampled data
    """
    # 1. Design Butterworth filter (BA is faster than SOS, albeit less stable at high orders - 4th order is safe)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    filter_order = 4

    # Design on CPU (BA form)
    b_cpu, a_cpu = scipy.signal.butter(
        filter_order, normal_cutoff, btype="low", output="ba"
    )

    # Transfer coefficients to GPU
    b_gpu = cp.asarray(b_cpu)
    a_gpu = cp.asarray(a_cpu)

    # 2. Apply Zero-Phase Filter (Forward-Backward)
    # Forward pass
    filtered = cusignal.lfilter(b_gpu, a_gpu, data_gpu, axis=0)

    # Backward pass (flip -> filter -> flip)
    filtered = cp.flip(filtered, axis=0)
    filtered = cusignal.lfilter(b_gpu, a_gpu, filtered, axis=0)
    filtered = cp.flip(filtered, axis=0)

    # 3. Downsample (Simple Slicing)
    # We can use simple slicing because the low-pass filter above acts as anti-aliasing
    downsample_factor = int(fs / target_fs)

    if downsample_factor > 1:
        # Slice: start=0, stop=None, step=factor
        return filtered[::downsample_factor]

    return filtered
