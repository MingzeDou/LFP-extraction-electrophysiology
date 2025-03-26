import argparse
from src.extract_lfp import extract_lfp
from src.config import INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, N_CHANNELS

def main():
    parser = argparse.ArgumentParser(description='LFP Extraction Tool')
    parser.add_argument('--input_file', type=str, default=INPUT_FILE, help='Path to the raw .dat file')
    parser.add_argument('--output_file', type=str, default=OUTPUT_FILE, help='Path to save the output .lfp file')
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE, help='Size of data chunks to process (in bytes)')
    parser.add_argument('--num_channels', type=int, default=N_CHANNELS, help='Number of channels in the data')
    
    args = parser.parse_args()
    
    extract_lfp(args.input_file, args.output_file, args.chunk_size, args.num_channels)

if __name__ == '__main__':
    main()