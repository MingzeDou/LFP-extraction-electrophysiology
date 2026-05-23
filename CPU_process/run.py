from src.extract_lfp import extract_lfp
from src.config import INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, N_CHANNELS


def main():
    # Define variables directly here (or edit src/config.py)
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    chunk_size = CHUNK_SIZE
    num_channels = N_CHANNELS

    extract_lfp(input_file, output_file, chunk_size, num_channels)


if __name__ == "__main__":
    main()
