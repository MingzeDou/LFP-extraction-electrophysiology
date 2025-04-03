import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union, Any, Type
import logging # Import logging

# Define type aliases for clarity
Array = Union[np.ndarray, Any] # Use Any for cp.ndarray if cupy is optional
SignalModule = Any # Type for signal or cusignal module
ArrayModule = Any # Type for np or cp module

# Setup logger for this module
logger = logging.getLogger(__name__)

# Check for CuPy availability and necessary signal processing functions
using_cupy: bool = False
cp: ArrayModule = np # Default cp to numpy
cusignal: SignalModule = signal # Default cusignal to scipy.signal
try:
    import cupy as cp_module # type: ignore
    import cupyx.scipy.signal as cusignal_module # type: ignore
    # Basic check
    test_array = cp_module.array([1, 2, 3])
    test_array.sum()
    # Check if sosfilt is available (might depend on CuPy/CUDA version)
    if hasattr(cusignal_module, 'sosfilt') and hasattr(cusignal_module, 'sosfilt_zi'):
        cp = cp_module # type: ignore
        cusignal = cusignal_module # type: ignore
        using_cupy = True
        Array = Union[np.ndarray, cp.ndarray] # Redefine Array now that cp is available
        logger.info("CuPy/CuPyX detected and sosfilt available. Using GPU acceleration for filtering.")
    else:
        logger.info("CuPy found, but cupyx.scipy.signal.sosfilt not available. Using NumPy/SciPy for filtering.")
except ImportError:
    logger.info("CuPy or CuPyX not installed. Using NumPy/SciPy for filtering.")
except Exception as e: # Catch other potential errors during CuPy init (e.g., CUDA errors)
    logger.warning(f"CuPy/CuPyX import or test failed: {e}. Using NumPy/SciPy for filtering.")


def low_pass_filter(data: Array, cutoff: float, fs: float, initial_zi: Optional[Array] = None) -> Tuple[Array, Array]:
    """Apply a low-pass filter optimized for LFP extraction.

    Uses a 4th-order Butterworth filter with second-order sections (SOS)
    for stability. Handles both NumPy and CuPy arrays efficiently, processing
    all channels in parallel if applicable.

    Parameters
    ----------
    data : Array
        Input signal array of shape (n_samples,) or (n_samples, n_channels).
        Can be a NumPy or CuPy array.
    cutoff : float
        Cutoff frequency in Hz. Must be less than fs/2.
    fs : float
        Sampling frequency in Hz.
    initial_zi : Optional[Array], optional
        Initial filter state. Must match the array type (NumPy/CuPy) of `data`.
        Shape should be (n_sections, 2) for single channel or
        (n_sections, 2, n_channels) for multi-channel.
        Defaults to None, which calculates the steady-state initial state.

    Returns
    -------
    filtered_data : Array
        Filtered data array, same type and shape as input `data`.
    final_filter_state : Array
        Final filter state array, same type as input `data`. Suitable for
        passing as `initial_zi` to the next chunk.

    Raises
    ------
    ValueError
        If cutoff frequency is >= fs/2.
    TypeError
        If `data` and `initial_zi` are of different array types (NumPy vs CuPy).
    ValueError
        If `initial_zi` has an incorrect shape.
    """
    # Determine the array module (NumPy or CuPy) based on input data type
    if using_cupy and isinstance(data, cp.ndarray):
        xp = cp
        sig = cusignal
    elif isinstance(data, np.ndarray):
        xp = np
        sig = signal
    else:
        # Fallback or raise error if data is not a recognized array type
        try:
            # Attempt to use get_array_module if maybe type checking failed (less likely)
            xp = cp.get_array_module(data)
            sig = cusignal if xp == cp else signal
            logger.warning(f"Input data type {type(data)} not explicitly handled, using get_array_module.")
        except AttributeError: # Handle case where cp is numpy and lacks get_array_module
             if isinstance(data, np.ndarray): # Double check if it's numpy after all
                 xp = np
                 sig = signal
             else:
                 raise TypeError(f"Unsupported data type: {type(data)}. Expected NumPy or CuPy array.")

    if cutoff >= fs / 2.0:
        raise ValueError("Cutoff frequency must be less than half the sampling frequency (fs/2)")

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Increase filter order for sharper cutoff / better anti-aliasing before downsampling
    filter_order: int = 4 # Increased from 2 to 4

    # Design filter using SciPy (coefficients are small, CPU is fine)
    # Output is 'sos' (second-order sections) for numerical stability
    sos: np.ndarray = signal.butter(filter_order, normal_cutoff, btype='low', output='sos')

    # Transfer sos coefficients to GPU if using cupy
    sos_filt: Array = sos # Use sos_filt for filtering, potentially on GPU
    if xp == cp:
        sos_filt = cp.asarray(sos)

    # Determine processing axis (time axis is typically 0 for (samples, channels))
    axis: int = 0
    n_sections: int = sos.shape[0]

    # Handle initial state zi
    zi_filt: Optional[Array] = None # State used for filtering
    if initial_zi is None:
        # Calculate initial state using the appropriate backend
        zi_calc = sig.sosfilt_zi(sos_filt) # Shape (n_sections, 2)
        # Reshape zi for multi-channel if necessary
        if data.ndim > 1:
            n_channels: int = data.shape[axis * -1 + 1] # Get channel dimension size
            # Expected shape for sosfilt zi is (n_sections, 2, n_channels)
            # We stack along the last axis.
            zi_filt = xp.stack([zi_calc] * n_channels, axis=-1)
        else:
            zi_filt = zi_calc
    else:
        # Ensure initial_zi is the correct type (np or cp) and shape
        if (xp == np and not isinstance(initial_zi, np.ndarray)) or \
           (using_cupy and xp == cp and not isinstance(initial_zi, cp.ndarray)):
             raise TypeError(f"Data (module: {xp.__name__}) and initial_zi (type: {type(initial_zi)}) must be of the same array type (NumPy or CuPy)")

        # Check shape after confirming type
        expected_zi_shape_base: Tuple[int, int] = (n_sections, 2)
        if data.ndim == 1 and initial_zi.shape != expected_zi_shape_base:
             raise ValueError(f"Expected initial_zi shape {expected_zi_shape_base} for single channel, got {initial_zi.shape}")
        if data.ndim > 1:
             n_channels = data.shape[axis * -1 + 1]
             expected_zi_shape_multi: Tuple[int, int, int] = (n_sections, 2, n_channels)
             if initial_zi.shape != expected_zi_shape_multi:
                 raise ValueError(f"Expected initial_zi shape {expected_zi_shape_multi} for multi-channel, got {initial_zi.shape}")
        # Correct indentation: This line belongs to the outer else block
        zi_filt = initial_zi # Use the provided initial state

    # --- Apply filter ---
    original_dtype = data.dtype # Store original dtype
    if xp == np:
        # --- NumPy path: Explicitly handle zi state ---
        logger.debug("Using NumPy sosfilt with explicit zi handling.")
        try:
            # Calculate initial state if not provided
            if initial_zi is None:
                 zi_np = signal.sosfilt_zi(sos) # Calculate default zi state
                 if data.ndim > 1:
                     n_channels: int = data.shape[axis * -1 + 1]
                     zi_np = np.stack([zi_np] * n_channels, axis=-1)
            else:
                 zi_np = initial_zi # Use provided state

            # --- Debugging Checks ---
            if not np.any(sos):
                 logger.error("SOS coefficients are all zero!")
                 return data, None
            if zi_np is None:
                 logger.error("Calculated/Provided zi_np is None before sosfilt call!")
                 return data, None
            logger.debug(f"Before sosfilt - Data shape: {data.shape}, Data type: {data.dtype}")
            logger.debug(f"Before sosfilt - zi_np shape: {zi_np.shape}, zi_np type: {zi_np.dtype}")
            # --- End Debugging Checks ---

            # Apply filter using the original sos calculated by scipy.signal and the prepared state
            filtered_data_raw, final_zi = signal.sosfilt(sos, data, axis=axis, zi=zi_np)
            # Ensure output dtype matches input dtype
            filtered_data = filtered_data_raw.astype(original_dtype, copy=False)


            # --- Debugging Checks ---
            logger.debug(f"After sosfilt - Returned state type: {type(final_zi)}")
            if final_zi is not None:
                 logger.debug(f"After sosfilt - Returned final_zi shape: {final_zi.shape}, type: {final_zi.dtype}")
            # Check if data was actually modified
            if np.allclose(data, filtered_data, atol=1e-6): # Use a slightly smaller tolerance
                 logger.warning("NumPy sosfilt did not appear to modify data significantly.")
            else:
                 logger.debug("NumPy sosfilt appears to have modified data.")
            # --- End Debugging Checks ---


            if not isinstance(final_zi, np.ndarray):
                 logger.error(f"NumPy sosfilt did not return expected state array. Got: {type(final_zi)}")
                 # Fallback: return None for state if it's not an array
                 final_zi = None
            # Check if data was actually modified
            if np.allclose(data, filtered_data, atol=1e-7):
                 logger.warning("NumPy sosfilt did not appear to modify data significantly.")

        except Exception as e:
            logger.error(f"Error during simplified NumPy sosfilt: {e}", exc_info=True)
            return data, None # Return original data and None state on error
        # --- End Simplified NumPy path ---
    elif xp == cp:
         # --- CuPy path (keep original logic with state handling) ---
         try:
            filtered_data_raw, final_zi = sig.sosfilt(sos_filt, data, axis=axis, zi=zi_filt)
            # Ensure output dtype matches input dtype
            filtered_data = filtered_data_raw.astype(original_dtype, copy=False)
         except Exception as e:
             logger.error(f"Error during CuPy sosfilt: {e}", exc_info=True)
             return data, None # Return original data and None state on error
         # --- End CuPy path ---
    else:
         logger.error("Unknown array module.")
         return data, None


    # --- Validate final_zi state (common validation after filtering) ---
    if final_zi is not None: # Only validate if we expect a state
        if not isinstance(final_zi, (np.ndarray, cp.ndarray)):
             logger.error(f"Filter function returned invalid final state type: {type(final_zi)}")
             final_zi = None # Set to None if type is wrong
        # Check if the returned state type matches the expected module (xp)
        elif (xp == np and not isinstance(final_zi, np.ndarray)) or \
             (using_cupy and xp == cp and not isinstance(final_zi, cp.ndarray)): # Check cp type only if using_cupy
             logger.warning(f"Filter final state type ({type(final_zi)}) mismatch with input data type module ({xp.__name__}). Attempting conversion.")
             try: # Correct indentation
                 if xp == cp and using_cupy: # Check using_cupy again before using cp
                     final_zi = cp.asarray(final_zi)
                 else:
                     final_zi = np.asarray(final_zi)
             except Exception as e: # Correct indentation
                 logger.error(f"Could not convert final filter state type: {e}")
                 final_zi = None # Indicate failure


    return filtered_data, final_zi

# Removed unused multi_stage_filter function

def downsample(data: Array, target_fs: float, original_fs: float) -> Array:
    """Downsample data by an integer factor using simple slicing.

    Parameters
    ----------
    data : Array
        Input data array (NumPy or CuPy). Assumes time is axis 0.
    target_fs : float
        Target sampling frequency.
    original_fs : float
        Original sampling frequency.

    Returns
    -------
    Array
        Downsampled data array.

    Raises
    ------
    ValueError
        If the downsampling factor is not an integer.
    """
    if original_fs % target_fs != 0:
        raise ValueError(f"Downsampling factor ({original_fs}/{target_fs}) must be an integer.")
    factor = int(original_fs / target_fs)
    # Slicing works for both NumPy and CuPy
    return data[::factor]

# Removed unused save_lfp_data function
