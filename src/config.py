# Configuration settings for LFP extraction tool
INPUT_FILE = r"path/to/your/your/datafile"  # Path to the input file
OUTPUT_FILE = r"path/to/your/output/file"  # Path to the output file
SAMPLE_RATE_ORIGINAL = 30000  # Original sample rate in Hz
TARGET_SAMPLING_RATE = 1250      # Target sample rate in Hz
CUTOFF_FREQUENCY = 450   # Cutoff frequency for low-pass filter in Hz (Hz)
CHUNK_SIZE = 499998720       # Default size of data chunks to process (in bytes)
N_CHANNELS = 385  # Default number of channels in the recordings
