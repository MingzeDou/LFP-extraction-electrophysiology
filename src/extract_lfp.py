import os
import numpy as np
from src.utils.filters import low_pass_filter, downsample
from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY

# Use NumPy as default processor
xp = np
GPU_AVAILABLE = False

# Try to initialize CuPy
try:
    import cupy as cp
    # Test CUDA functionality
    a = cp.array([1, 2, 3])
    a.sum()
    cp.RawKernel('extern "C" __global__ void example() {}', 'example')
    xp = cp
    GPU_AVAILABLE = True
    print(f"Using CuPy version: {cp.__version__}")
    print(f"GPU device: {cp.cuda.get_device_id()}")
except (ImportError, RuntimeError) as e:
    print(f"CuPy initialization failed, using NumPy instead")

def free_gpu_memory():
    """Free GPU memory if CuPy is available"""
    if GPU_AVAILABLE and hasattr(cp, 'get_default_memory_pool'):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

def process_chunk(data_chunk, filter_states=None, overlap=0, num_channels=1):
    """Process a chunk of data: filter then downsample"""
    is_multichannel = num_channels > 1 and len(data_chunk.shape) > 1
    downsampling_factor = int(SAMPLE_RATE_ORIGINAL / TARGET_SAMPLING_RATE)
    cutoff = min(TARGET_SAMPLING_RATE * 0.45, CUTOFF_FREQUENCY)
    
    if is_multichannel:
        # Calculate output size
        output_samples = data_chunk.shape[0] // downsampling_factor
        if overlap > 0:
            output_samples -= (overlap // downsampling_factor)
        
        processed_data = np.zeros((output_samples, num_channels), dtype=np.float32)
        new_filter_states = []
        
        # Process each channel individually
        for ch_idx in range(num_channels):
            channel_data = data_chunk[:, ch_idx]
            
            # 1. Filter first at original sample rate
            ch_filter_state = None if filter_states is None else filter_states[ch_idx]
            filtered, next_filter_state = low_pass_filter(channel_data, cutoff, SAMPLE_RATE_ORIGINAL, ch_filter_state)
            new_filter_states.append(next_filter_state)
            
            # 2. Then downsample
            downsampled = downsample(filtered, TARGET_SAMPLING_RATE, SAMPLE_RATE_ORIGINAL)
            
            # 3. Trim overlap if needed
            if overlap > 0:
                overlap_ds = overlap // downsampling_factor
                downsampled = downsampled[overlap_ds:]
                
            # Ensure consistent length
            if len(downsampled) > output_samples:
                downsampled = downsampled[:output_samples]
            elif len(downsampled) < output_samples:
                padding = np.zeros(output_samples - len(downsampled))
                downsampled = np.concatenate([downsampled, padding])
            
            # Convert CuPy array to NumPy if needed
            if GPU_AVAILABLE and hasattr(downsampled, 'get'):
                downsampled = downsampled.get()
                
            processed_data[:, ch_idx] = downsampled
        
        return processed_data, new_filter_states
    else:
        # Single-channel mode
        # 1. Filter first at original sample rate
        if GPU_AVAILABLE:
            try:
                gpu_data = xp.array(data_chunk)
                filtered_data = low_pass_filter(gpu_data, cutoff, SAMPLE_RATE_ORIGINAL)
                filtered_data = cp.asnumpy(filtered_data)
            except Exception as e:
                filtered_data = low_pass_filter(data_chunk, cutoff, SAMPLE_RATE_ORIGINAL)
        else:
            filtered_data = low_pass_filter(data_chunk, cutoff, SAMPLE_RATE_ORIGINAL)
            
        # 2. Then downsample
        downsampled_data = downsample(filtered_data, TARGET_SAMPLING_RATE, SAMPLE_RATE_ORIGINAL)
        
        # Trim overlap if needed
        if overlap > 0:
            overlap_ds = overlap // downsampling_factor
            downsampled_data = downsampled_data[overlap_ds:]
        
        # Return as 2D array for consistency
        if len(downsampled_data.shape) == 1:
            downsampled_data = downsampled_data.reshape(-1, 1)
        
        return downsampled_data

def extract_lfp(input_file, output_file, chunk_size, num_channels=384):
    """
    Extract LFP from OpenEphys data file.
    
    Args:
        input_file: Path to input raw data file (.dat)
        output_file: Path to output LFP file
        chunk_size: Size of chunks to process in bytes
        num_channels: Number of channels in the data
    """
    # Validate inputs
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing OpenEphys file: {input_file} to {output_file}")
    
    # Adjust chunk size to be divisible by bytes_per_frame
    bytes_per_frame = num_channels * 2  # int16 = 2 bytes
    chunk_size = chunk_size - (chunk_size % bytes_per_frame)
    
    # Calculate overlap size for smooth transitions
    overlap_samples = int(0.5 * SAMPLE_RATE_ORIGINAL)  # 0.5s of overlap
    
    # Set scaling factor - keep original scaling since we're staying in int16
    scaling_factor = 1  # No need to scale when keeping as int16
    
    # Get file size for progress tracking
    file_size = os.path.getsize(input_file)
    bytes_processed = 0
    
    filter_states = None

    with open(input_file, 'rb') as f:
        with open(output_file, 'wb') as out_f:
            chunk_count = 0
            previous_chunk_end = None
            
            while bytes_processed < file_size:
                # Read a chunk of data
                data_bytes = f.read(chunk_size)
                if not data_bytes:
                    break
                
                bytes_processed += len(data_bytes)
                
                # Convert to numpy array as int16
                data_chunk = np.frombuffer(data_bytes, dtype=np.int16)
                
                # Handle incomplete chunks at end of file
                if len(data_chunk) % num_channels != 0:
                    complete_samples = len(data_chunk) // num_channels
                    data_chunk = data_chunk[:complete_samples * num_channels]
                
                # Reshape and convert to float32 for processing
                data_chunk = data_chunk.reshape(-1, num_channels)
                data_chunk = data_chunk.astype(np.float32) * scaling_factor
                
                # Handle overlap between chunks
                if previous_chunk_end is not None and chunk_count > 0:
                    data_chunk = np.vstack([previous_chunk_end, data_chunk])
                
                # Store end of chunk for overlap
                if data_chunk.shape[0] >= overlap_samples:
                    previous_chunk_end = data_chunk[-overlap_samples:, :].copy()
                
                # Process chunk (filtering and downsampling)
                overlap_to_trim = overlap_samples if chunk_count > 0 else 0
                lfp_chunk, filter_states = process_chunk(
                    data_chunk, filter_states, overlap_to_trim, num_channels)
                
                # Write to file
                # if len(lfp_chunk.shape) > 1:
                #     lfp_chunk = lfp_chunk.flatten('C')
                    
                # Convert back to int16 for storage
                lfp_data = lfp_chunk.astype(np.int16)  # Changed to int16
                out_f.write(lfp_data.tobytes())
                                
                # Progress reporting
                chunk_count += 1
                if chunk_count % 10 == 0:
                    progress = (bytes_processed / file_size) * 100
                    print(f"Processed {chunk_count} chunks... ({progress:.1f}% complete)")
                    # Free GPU memory periodically
                    if GPU_AVAILABLE and chunk_count % 50 == 0:
                        free_gpu_memory()

    if GPU_AVAILABLE:
        free_gpu_memory()

    print(f"OpenEphys LFP extraction complete. Saved to {output_file}")