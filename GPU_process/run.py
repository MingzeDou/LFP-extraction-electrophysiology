from src.extract_lfp import extract_lfp


def main():
    # Define variables directly here (or edit src/config.py)
    # Using defaults from config.py for now, but these can be overridden
    input_file = r"path\to\your\input_file.dat"
    output_file = r"path\to\your\output_file.lfp"
    chunk_size = 1500000
    num_channels = 384

    extract_lfp(input_file, output_file, chunk_size, num_channels)


if __name__ == "__main__":
    main()
