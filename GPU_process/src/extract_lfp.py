import os
import time

import numpy as np

from src.config import CUTOFF_FREQUENCY, SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE
from src.utils.filters import (
    aligned_lfp_indices,
    compute_downsample_factor,
    design_lfp_filter,
    filter_chunk_centered,
    to_int16_gpu,
)

try:
    import cupy as cp
except ImportError as e:
    raise ImportError(
        f"CuPy failed to import: {e}. Please install CuPy or use the CPU_process scripts."
    ) from e

try:
    _test = cp.array([1, 2, 3])
    _test.sum()
    print(f"Using CuPy version: {cp.__version__}")
    print(f"GPU device: {cp.cuda.get_device_id()}")
except Exception as e:
    raise RuntimeError(
        f"CuPy initialization failed ({type(e).__name__}: {e}). Ensure CUDA drivers "
        "and toolkit are compatible. Otherwise, use the CPU_process scripts."
    ) from e


DEFAULT_FILTER_TAPS = 2001


def free_gpu_memory():
    """Free cached CuPy memory blocks."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"Error freeing GPU memory: {e}")


def extract_lfp(input_file, output_file, chunk_size, num_channels=384, num_taps=DEFAULT_FILTER_TAPS):
    """
    Extract aligned LFP from an OpenEphys int16 .dat file using GPU processing.

    The GPU path uses the same FIR coefficients and global sample-index alignment
    as the CPU path. LFP sample k aligns to raw sample k * raw_fs / lfp_fs.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_fs = int(SAMPLE_RATE_ORIGINAL)
    lfp_fs = int(TARGET_SAMPLING_RATE)
    downsample_factor = compute_downsample_factor(raw_fs, lfp_fs)

    dtype_input = np.int16
    bytes_per_sample = np.dtype(dtype_input).itemsize
    bytes_per_frame = int(num_channels) * bytes_per_sample

    file_size_bytes = os.path.getsize(input_file)
    total_samples = file_size_bytes // bytes_per_frame
    trailing_bytes = file_size_bytes % bytes_per_frame
    expected_lfp_samples = (total_samples + downsample_factor - 1) // downsample_factor

    samples_per_chunk = int(chunk_size) // bytes_per_frame
    if samples_per_chunk <= 0:
        raise ValueError("chunk_size is too small for one full sample across all channels")

    taps = design_lfp_filter(CUTOFF_FREQUENCY, raw_fs, num_taps=num_taps)
    taps_gpu = cp.asarray(taps)
    filter_delay = (len(taps) - 1) // 2

    print(f"Processing OpenEphys file with GPU: {input_file}")
    print(f"Output LFP file: {output_file}")
    print(f"Channels: {num_channels}")
    print(f"Raw fs: {raw_fs} Hz; LFP fs: {lfp_fs} Hz")
    print(f"Derived downsampling factor: {downsample_factor}")
    print(f"Cutoff: {CUTOFF_FREQUENCY} Hz; FIR taps: {len(taps)}")
    print(f"Total raw samples: {total_samples}")
    print(f"Expected LFP samples: {expected_lfp_samples}")
    if trailing_bytes:
        print(f"Warning: ignoring {trailing_bytes} trailing byte(s) not forming a full frame")

    data_map = np.memmap(
        input_file,
        dtype=dtype_input,
        mode="r",
        shape=(total_samples, int(num_channels)),
    )

    start_time = time.time()
    samples_written = 0
    chunk_idx = 0

    with open(output_file, "wb") as out_f:
        for block_start in range(0, total_samples, samples_per_chunk):
            block_end = min(block_start + samples_per_chunk, total_samples)

            load_start = max(0, block_start - filter_delay)
            load_end = min(total_samples, block_end + filter_delay)

            loaded_cpu = np.asarray(data_map[load_start:load_end, :], dtype=np.float32)
            try:
                loaded_gpu = cp.asarray(loaded_cpu)
                filtered_gpu = filter_chunk_centered(loaded_gpu, taps_gpu)
            except Exception as e:
                free_gpu_memory()
                raise RuntimeError(f"GPU processing failed on chunk {chunk_idx + 1}: {e}") from e

            output_raw_indices = aligned_lfp_indices(
                block_start, block_end, raw_fs=raw_fs, lfp_fs=lfp_fs
            )
            if output_raw_indices.size:
                local_indices = cp.asarray(output_raw_indices - load_start)
                lfp_gpu = filtered_gpu[local_indices, :]
                lfp_cpu = cp.asnumpy(to_int16_gpu(lfp_gpu))
                out_f.write(lfp_cpu.tobytes())
                samples_written += int(output_raw_indices.size)

            del loaded_cpu, loaded_gpu, filtered_gpu
            if output_raw_indices.size:
                del local_indices, lfp_gpu, lfp_cpu

            chunk_idx += 1
            progress = (block_end / total_samples) * 100 if total_samples else 100.0
            print(f"Processed chunk {chunk_idx}: {progress:.1f}%")
            free_gpu_memory()

    free_gpu_memory()
    duration = time.time() - start_time
    if samples_written != expected_lfp_samples:
        raise RuntimeError(
            f"LFP sample count mismatch: wrote {samples_written}, "
            f"expected {expected_lfp_samples}"
        )

    print(f"\nTotal processing time: {duration:.2f} seconds")
    print(f"OpenEphys LFP extraction complete using GPU. Saved to {output_file}")
