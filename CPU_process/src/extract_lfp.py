import os
import time

import numpy as np

from src.config import CUTOFF_FREQUENCY, SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE
from src.utils.filters import (
    aligned_lfp_indices,
    compute_downsample_factor,
    design_lfp_filter,
    filter_chunk_centered,
    to_int16,
)


DEFAULT_FILTER_TAPS = 2001


def extract_lfp(input_file, output_file, chunk_size, num_channels=384, num_taps=DEFAULT_FILTER_TAPS):
    """
    Extract aligned LFP from an OpenEphys int16 .dat file using CPU processing.

    LFP sample k is aligned to raw sample k * raw_fs / lfp_fs. The downsampling
    factor is derived from SAMPLE_RATE_ORIGINAL and TARGET_SAMPLING_RATE.
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
    filter_delay = (len(taps) - 1) // 2

    print(f"Processing OpenEphys file with CPU: {input_file}")
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

            loaded = np.asarray(data_map[load_start:load_end, :], dtype=np.float32)
            filtered = filter_chunk_centered(loaded, taps)

            output_raw_indices = aligned_lfp_indices(
                block_start, block_end, raw_fs=raw_fs, lfp_fs=lfp_fs
            )
            if output_raw_indices.size:
                local_indices = output_raw_indices - load_start
                lfp_chunk = filtered[local_indices, :]
                out_f.write(to_int16(lfp_chunk).tobytes())
                samples_written += int(output_raw_indices.size)

            chunk_idx += 1
            progress = (block_end / total_samples) * 100 if total_samples else 100.0
            print(f"Processed chunk {chunk_idx}: {progress:.1f}%")

    duration = time.time() - start_time
    if samples_written != expected_lfp_samples:
        raise RuntimeError(
            f"LFP sample count mismatch: wrote {samples_written}, "
            f"expected {expected_lfp_samples}"
        )

    print(f"\nTotal processing time: {duration:.2f} seconds")
    print(f"OpenEphys LFP extraction complete using CPU. Saved to {output_file}")
