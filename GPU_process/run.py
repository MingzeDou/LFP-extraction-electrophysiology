from src.extract_lfp import extract_lfp


def main():
    # Define variables directly here (or edit src/config.py)
    # Using defaults from config.py for now, but these can be overridden
    input_file = r"/Volumes/Peter/ephys_data_mingze/MV03_2025-12-21_12-00-27/MV03_2025-12-21_12-00-27_ProbeB.dat"
    output_file = r"/Volumes/Peter/ephys_data_mingze/MV03_2025-12-21_12-00-27/MV03_2025-12-21_12-00-27_ProbeB.lfp"
    chunk_size = 550000000
    num_channels = 384

    extract_lfp(input_file, output_file, chunk_size, num_channels)


if __name__ == "__main__":
    main()
