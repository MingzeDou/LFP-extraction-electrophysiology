import numpy as np
import scipy.signal

try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
except ImportError as e:
    raise ImportError(
        f"CuPy or CuPyX components failed to import: {e}. "
        "Please install CuPy or use the CPU_process scripts."
    ) from e


DEFAULT_FILTER_TAPS = 2001


def compute_downsample_factor(raw_fs, lfp_fs):
    """Return integer downsampling factor derived from sample rates."""
    raw_fs = int(raw_fs)
    lfp_fs = int(lfp_fs)
    if raw_fs <= 0 or lfp_fs <= 0:
        raise ValueError("raw_fs and lfp_fs must be positive")
    if raw_fs % lfp_fs != 0:
        raise ValueError(
            f"raw_fs ({raw_fs}) must be an integer multiple of lfp_fs ({lfp_fs}) "
            "for exact sample alignment"
        )
    return raw_fs // lfp_fs


def design_lfp_filter(cutoff, raw_fs, num_taps=DEFAULT_FILTER_TAPS):
    """
    Design an odd-length, linear-phase FIR low-pass filter.

    Coefficients are designed with SciPy on CPU, then transferred to GPU by the
    caller. This keeps CPU and GPU filter definitions identical.
    """
    num_taps = int(num_taps)
    if num_taps < 3:
        raise ValueError("num_taps must be >= 3")
    if num_taps % 2 == 0:
        num_taps += 1

    nyquist = raw_fs / 2.0
    if not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff must be between 0 and Nyquist ({nyquist} Hz)")

    taps = scipy.signal.firwin(num_taps, cutoff, fs=raw_fs, window="hamming")
    return taps.astype(np.float32)


def filter_chunk_centered(data_gpu, taps_gpu):
    """Apply centered FIR convolution along time axis to samples x channels GPU data."""
    data_gpu = data_gpu.astype(cp.float32, copy=False)
    taps_gpu = taps_gpu.astype(cp.float32, copy=False)
    kernel = taps_gpu[:, None]

    if hasattr(cusignal, "oaconvolve"):
        return cusignal.oaconvolve(data_gpu, kernel, mode="same", axes=0).astype(cp.float32)
    return cusignal.convolve(data_gpu, kernel, mode="same", method="fft").astype(cp.float32)


def aligned_lfp_indices(block_start, block_end, raw_fs, lfp_fs):
    """Raw sample indices in [block_start, block_end) that land exactly on LFP ticks."""
    factor = compute_downsample_factor(raw_fs, lfp_fs)
    first = ((int(block_start) + factor - 1) // factor) * factor
    if first >= block_end:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, int(block_end), factor, dtype=np.int64)


def to_int16_gpu(data_gpu):
    """Round, clip, and convert filtered LFP data back to int16 on GPU."""
    return cp.rint(cp.clip(data_gpu, np.iinfo(np.int16).min, np.iinfo(np.int16).max)).astype(
        cp.int16
    )
