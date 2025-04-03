import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy import signal
from matplotlib.gridspec import GridSpec
import time
import pandas as pd
from tabulate import tabulate  # You may need to install this: pip install tabulate

# Import necessary components from the LFP extraction tool source
# Assuming compare_files.py is run from the project root directory
try:
    from src.utils.filters import low_pass_filter
    from src.config import SAMPLE_RATE_ORIGINAL, TARGET_SAMPLING_RATE, CUTOFF_FREQUENCY
except ImportError:
    print("Warning: Could not import from src. Ensure this script is run from the project root.")
    # Define fallbacks if run standalone (less accurate comparison)
    SAMPLE_RATE_ORIGINAL = 30000
    TARGET_SAMPLING_RATE = 1250
    CUTOFF_FREQUENCY = 450
    # Define a dummy low_pass_filter if needed
    def low_pass_filter(data, cutoff, fs, initial_zi=None):
        print("Warning: Using dummy low_pass_filter. Filter comparison plot will be inaccurate.")
        # Simple placeholder filter (not accurate)
        b, a = signal.butter(4, cutoff / (0.5 * fs), btype='low')
        filtered_data = signal.lfilter(b, a, data)
        # Dummy state matching expected output format
        dummy_state = np.zeros((len(a)-1,)) if initial_zi is None else initial_zi
        return filtered_data, dummy_state


def compute_psd(signal_data, fs, nperseg=None):
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        signal_data: Input time series
        fs: Sampling frequency
        nperseg: Length of each segment (default: fs)
        
    Returns:
        frequencies: Array of sample frequencies
        psd: Power spectral density
    """
    if nperseg is None:
        nperseg = min(len(signal_data), fs)
        
    frequencies, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
    return frequencies, psd

def compute_coherence(x, y, fs, nperseg=None):
    """
    Compute magnitude squared coherence between two signals.
    
    Args:
        x: First input time series
        y: Second input time series
        fs: Sampling frequency
        nperseg: Length of each segment
        
    Returns:
        frequencies: Array of frequencies
        coherence: Magnitude squared coherence
    """
    if nperseg is None:
        nperseg = min(len(x), fs)
        
    frequencies, coherence = signal.coherence(x, y, fs=fs, nperseg=nperseg)
    return frequencies, coherence

def compute_snr(signal_data, noise_data=None):
    """
    Compute signal-to-noise ratio.
    If noise_data is not provided, use the signal's own variance as a proxy.
    
    Args:
        signal_data: Input signal
        noise_data: Noise estimate (optional)
        
    Returns:
        snr: Signal-to-noise ratio in dB
    """
    if noise_data is None:
        # Estimate SNR using variance of the signal as a proxy
        signal_var = np.var(signal_data)
        # Estimate noise as the high-frequency components
        noise_var = np.var(signal_data - signal.savgol_filter(signal_data, 15, 3))
        if noise_var == 0:
            return float('inf')
        return 10 * np.log10(signal_var / noise_var)
    else:
        signal_power = np.mean(signal_data**2)
        noise_power = np.mean(noise_data**2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

def filter_data(data, fs, lowcut=0.5, highcut=300, order=4):
    """
    Apply bandpass filter to data.
    
    Args:
        data: Input time series
        fs: Sampling frequency
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        order: Filter order
        
    Returns:
        filtered_data: Filtered time series
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def compare_signals_raw(dat_file, lfp_file, num_channels=384, num_samples=None):
    """
    Compare raw DAT file with processed LFP file.
    Simply downsample the raw data to match LFP sampling rate.
    
    Args:
        dat_file: Path to original .dat file (raw data)
        lfp_file: Path to generated .lfp file (filtered and downsampled)
        num_channels: Number of channels in recordings
        num_samples: Number of samples to analyze
    """
    print(f"Comparing DAT: {dat_file} with LFP: {lfp_file}")
    
    # Define parameters
    raw_fs = 30000         # Raw data sampling rate (Hz)
    lfp_fs = 1250          # LFP sampling rate (Hz)
    downsampling_factor = raw_fs / lfp_fs
    scaling_factor = 0.195  # Typical Neuropixels scaling factor
    
    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 60000)  # Limit to ~2 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)  # Changed to int16
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)
    
    # Select channels to analyze
    test_channels = [0, 10, 100, 128, 255, 383]
    test_channels = [ch for ch in test_channels if ch < num_channels]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(test_channels), 1, figsize=(12, 3*len(test_channels)))
    
    # Time vectors for plotting
    lfp_time = np.arange(len(lfp_data)) / lfp_fs
    
    # Downsample raw data and compare
    for i, ch in enumerate(test_channels):
        # Get the raw signal and downsample
        raw_signal = raw_data[:, ch]
        indices = np.arange(0, len(raw_signal), downsampling_factor).astype(int)
        downsampled_raw = raw_signal[indices]
        
        # Trim to match sizes
        min_len = min(len(downsampled_raw), len(lfp_data[:, ch]))
        downsampled_raw = downsampled_raw[:min_len]
        lfp_ch = lfp_data[:min_len, ch]
        
        # Calculate correlation
        correlation = np.corrcoef(downsampled_raw, lfp_ch)[0, 1] if min_len > 0 else 0
        
        # Plot comparison
        ax = axes[i] if len(test_channels) > 1 else axes
        ax.plot(lfp_time[:min_len], downsampled_raw, label='Raw downsampled', alpha=0.7)
        ax.plot(lfp_time[:min_len], lfp_ch, label='LFP file', alpha=0.7)
        ax.set_title(f'Channel {ch}: Correlation = {correlation:.3f}', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_signals_detailed(dat_file, lfp_file, num_channels=384, num_samples=None, channel=0):
    """
    Perform detailed comparison between raw and processed LFP data for a single channel,
    including time-domain, frequency-domain and coherence analysis.
    
    Args:
        dat_file: Path to original .dat file (raw data)
        lfp_file: Path to generated .lfp file (filtered and downsampled)
        num_channels: Number of channels in recordings
        num_samples: Number of samples to analyze
        channel: Channel to analyze in detail
    """
    print(f"Detailed analysis for channel {channel}")

    # Use parameters imported from src.config or fallbacks
    raw_fs = SAMPLE_RATE_ORIGINAL
    lfp_fs = TARGET_SAMPLING_RATE
    filter_cutoff = CUTOFF_FREQUENCY # The cutoff used during LFP extraction
    downsampling_factor = raw_fs / lfp_fs
    if downsampling_factor != int(downsampling_factor):
        print(f"Warning: Downsampling factor ({downsampling_factor}) is not an integer. Comparison might be less accurate.")
    downsampling_factor = int(downsampling_factor)

    # Scaling factor (adjust if your system uses a different value)
    scaling_factor = 0.195 # Typical Neuropixels scaling factor

    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 120000)  # Limit to ~4 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)

    # Get signal for the specified channel
    raw_signal_orig = raw_data[:, channel] # Raw signal at original sampling rate
    lfp_signal = lfp_data[:, channel]      # LFP signal from file (already processed)

    # --- Apply the same filtering used in LFP extraction to the raw segment ---
    # Use the imported low_pass_filter function
    # Note: This filters the segment in one go, ignoring chunking/overlap effects
    # which might exist in the full LFP file generation. It's an approximation.
    cutoff_freq_for_filter = min(filter_cutoff, raw_fs * 0.45) # Safety check
    filtered_raw_signal, _ = low_pass_filter(raw_signal_orig.astype(np.float32), cutoff_freq_for_filter, raw_fs)

    # --- Downsample the filtered raw signal using simple slicing (mimicking extract_lfp) ---
    downsampled_filtered_raw = filtered_raw_signal[::downsampling_factor]

    # --- Trim signals to the minimum length for comparison ---
    min_len = min(len(downsampled_filtered_raw), len(lfp_signal))
    downsampled_filtered_raw = downsampled_filtered_raw[:min_len]
    lfp_signal = lfp_signal[:min_len]
    raw_signal_orig = raw_signal_orig[:min_len * downsampling_factor] # Keep corresponding raw segment
    filtered_raw_signal = filtered_raw_signal[:min_len * downsampling_factor] # Keep corresponding filtered raw segment

    # --- Time vectors for plotting ---
    raw_time = np.arange(len(raw_signal_orig)) / raw_fs
    lfp_time = np.arange(min_len) / lfp_fs

    # --- Calculate Metrics (using downsampled filtered raw vs LFP file) ---
    correlation = np.corrcoef(downsampled_filtered_raw, lfp_signal)[0, 1] if min_len > 1 else np.nan
    snr_filt_raw = compute_snr(downsampled_filtered_raw)
    snr_lfp = compute_snr(lfp_signal)

    # --- Calculate PSDs ---
    nperseg_raw = min(len(raw_signal_orig), raw_fs * 2) # Use ~2s window for raw PSD
    nperseg_lfp = min(len(lfp_signal), lfp_fs * 2)     # Use ~2s window for LFP PSD

    freqs_orig_raw, psd_orig_raw = compute_psd(raw_signal_orig, raw_fs, nperseg=nperseg_raw)
    freqs_filt_raw, psd_filt_raw = compute_psd(filtered_raw_signal, raw_fs, nperseg=nperseg_raw)
    freqs_lfp, psd_lfp = compute_psd(lfp_signal, lfp_fs, nperseg=nperseg_lfp)

    # --- Calculate Coherence (between downsampled filtered raw and LFP file) ---
    freqs_coh, coherence = compute_coherence(downsampled_filtered_raw, lfp_signal, lfp_fs, nperseg=nperseg_lfp)

    # --- Create Figure ---
    fig = plt.figure(figsize=(15, 16)) # Increased height
    gs = GridSpec(4, 2, figure=fig) # Changed to 4 rows

    # --- Plotting ---
    # 1. Time domain plot (Downsampled Filtered Raw vs LFP File)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(lfp_time, downsampled_filtered_raw, label='Raw (Filtered & Downsampled)', alpha=0.7)
    ax1.plot(lfp_time, lfp_signal, label='LFP File Signal', alpha=0.7)
    ax1.set_title(f'Channel {channel} - Time Domain Comparison', fontsize=14)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal (scaled)')
    ax1.legend()
    ax1.grid(True)

    # 2. PSD of Raw vs Filtered Raw (at original sample rate)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(freqs_orig_raw, psd_orig_raw, label='Original Raw', alpha=0.8)
    ax2.semilogy(freqs_filt_raw, psd_filt_raw, label=f'Raw Filtered ({cutoff_freq_for_filter:.0f} Hz LP)', alpha=0.8)
    ax2.set_title('Filter Effect on Raw Signal PSD', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD [V**2/Hz]')
    ax2.set_xlim([0, raw_fs / 2])
    ax2.axvline(filter_cutoff, color='r', linestyle='--', label=f'Target Cutoff ({filter_cutoff} Hz)')
    ax2.axvline(lfp_fs / 2, color='g', linestyle=':', label=f'LFP Nyquist ({lfp_fs/2} Hz)')
    ax2.legend()
    ax2.grid(True)

    # 3. PSD of Final LFP Signal (Zoomed near Nyquist for Aliasing Check)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(freqs_lfp, psd_lfp, label='LFP File Signal')
    ax3.set_title('LFP File PSD (Aliasing Check)', fontsize=14)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('PSD [V**2/Hz]')
    # Zoom near LFP Nyquist frequency
    nyquist_lfp = lfp_fs / 2
    zoom_range = nyquist_lfp * 0.1 # Look at the top 10% of the LFP band
    ax3.set_xlim([nyquist_lfp - zoom_range, nyquist_lfp])
    # Adjust y-limits to see potential peaks near Nyquist
    if len(psd_lfp[freqs_lfp > nyquist_lfp - zoom_range]) > 0:
         min_psd_zoom = np.min(psd_lfp[freqs_lfp > nyquist_lfp - zoom_range])
         max_psd_zoom = np.max(psd_lfp[freqs_lfp > nyquist_lfp - zoom_range])
         ax3.set_ylim([min_psd_zoom * 0.1, max_psd_zoom * 10]) # Log scale adjustment
    ax3.axvline(nyquist_lfp, color='g', linestyle=':', label=f'LFP Nyquist ({nyquist_lfp} Hz)')
    ax3.legend()
    ax3.grid(True)


    # 4. Coherence plot (Downsampled Filtered Raw vs LFP File)
    ax4 = fig.add_subplot(gs[2, 0]) # Moved to row 3
    ax4.plot(freqs_coh, coherence)
    ax4.set_title('Coherence (Filtered Raw vs LFP File)', fontsize=14)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Coherence')
    ax4.set_ylim([0, 1.1])
    ax4.set_xlim([0, lfp_fs / 2])
    ax4.grid(True)

    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[2, 1]) # Moved to row 3
    ax5.axis('off')
    # Calculate mean coherence below filter cutoff for a more relevant metric
    mean_coherence_passband = np.mean(coherence[freqs_coh < filter_cutoff]) if any(freqs_coh < filter_cutoff) else np.nan

    summary_text = (
        f"Summary Statistics (Ch {channel}):\n\n"
        f"Comparison: Filtered Raw vs LFP File\n"
        f"Correlation: {correlation:.4f}\n"
        f"SNR (Filt Raw): {snr_filt_raw:.2f} dB\n"
        f"SNR (LFP File): {snr_lfp:.2f} dB\n\n"
        f"Coherence Metrics:\n"
        f" Mean (0-{filter_cutoff:.0f}Hz): {mean_coherence_passband:.4f}\n"
        f" Mean (Overall): {np.mean(coherence):.4f}\n"
        f" Median (Overall): {np.median(coherence):.4f}\n"
        f" Max: {np.max(coherence):.4f} at {freqs_coh[np.argmax(coherence)]:.1f} Hz"
    )
    ax5.text(0.05, 0.5, summary_text, fontsize=11, va='center', family='monospace')

    # Add an extra empty plot area in the last row if needed, or use for more info
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    info_text = (
        f"Analysis Parameters:\n"
        f" Raw FS: {raw_fs} Hz | LFP FS: {lfp_fs} Hz | Downsampling Factor: {downsampling_factor}\n"
        f" LFP Filter: {filter_cutoff:.0f} Hz Low-pass (Order used in low_pass_filter)\n"
        f" Samples Analyzed (Raw): {len(raw_signal_orig)} | Samples Analyzed (LFP): {min_len}\n"
        f" PSD Window: {nperseg_lfp} samples ({nperseg_lfp/lfp_fs:.2f} s) | Coherence Window: {nperseg_lfp} samples"
    )
    ax6.text(0.05, 0.5, info_text, fontsize=10, va='center')


    plt.tight_layout(pad=2.0) # Add padding
    plt.show()

def compare_frequency_bands(dat_file, lfp_file, num_channels=384, num_samples=None, channel=0):
    """
    Compare power in specific frequency bands between raw and processed LFP data.
    
    Args:
        dat_file: Path to original .dat file (raw data)
        lfp_file: Path to generated .lfp file (filtered and downsampled)
        num_channels: Number of channels in recordings
        num_samples: Number of samples to analyze
        channel: Channel to analyze
    """
    print(f"Frequency band analysis for channel {channel}")
    
    # Define parameters
    raw_fs = 30000         # Raw data sampling rate (Hz)
    lfp_fs = 1250          # LFP sampling rate (Hz)
    downsampling_factor = raw_fs / lfp_fs
    scaling_factor = 0.195  # Typical Neuropixels scaling factor
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 80),
        'High Gamma': (80, 200)
    }
    
    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 300000)  # Limit to ~10 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)

    # Get signal for the specified channel
    raw_signal = raw_data[:, channel]
    lfp_signal = lfp_data[:, channel]
    
    # Downsample raw data
    indices = np.arange(0, len(raw_signal), downsampling_factor).astype(int)
    downsampled_raw = raw_signal[indices]
    
    # Trim to match sizes
    min_len = min(len(downsampled_raw), len(lfp_signal))
    downsampled_raw = downsampled_raw[:min_len]
    lfp_signal = lfp_signal[:min_len]
    
    # Calculate PSDs
    nperseg = min(4096, min_len)
    freqs_raw, psd_raw = compute_psd(downsampled_raw, lfp_fs, nperseg=nperseg)
    freqs_lfp, psd_lfp = compute_psd(lfp_signal, lfp_fs, nperseg=nperseg)
    
    # Calculate power in each frequency band
    band_powers_raw = {}
    band_powers_lfp = {}
    
    for band_name, (fmin, fmax) in bands.items():
        # Find indices corresponding to the frequency band
        idx_raw = np.logical_and(freqs_raw >= fmin, freqs_raw <= fmax)
        idx_lfp = np.logical_and(freqs_lfp >= fmin, freqs_lfp <= fmax)
        
        # Calculate power in band (area under PSD curve)
        power_raw = np.trapz(psd_raw[idx_raw], freqs_raw[idx_raw])
        power_lfp = np.trapz(psd_lfp[idx_lfp], freqs_lfp[idx_lfp])
        
        band_powers_raw[band_name] = power_raw
        band_powers_lfp[band_name] = power_lfp
    
    # Create figure with bar plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot PSDs
    ax1.semilogy(freqs_raw, psd_raw, label='Raw downsampled', alpha=0.8)
    ax1.semilogy(freqs_lfp, psd_lfp, label='LFP file', alpha=0.8)
    ax1.set_title(f'Channel {channel} - Power Spectral Density', fontsize=14)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD [V**2/Hz]')
    ax1.set_xlim([0, min(lfp_fs/2, 200)])  # Limit to 200 Hz for better visualization
    ax1.legend()
    ax1.grid(True)
    
    # Add colored patches for frequency bands
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    for (band_name, (fmin, fmax)), color in zip(bands.items(), colors):
        if fmax <= ax1.get_xlim()[1]:  # Only plot bands within the visible range
            ymin, ymax = ax1.get_ylim()
            rect = plt.Rectangle((fmin, ymin), fmax-fmin, ymax-ymin, 
                                facecolor=color, alpha=0.1)
            ax1.add_patch(rect)
            # Add band name at the top of each colored area
            ax1.text(fmin + (fmax-fmin)/2, ymax*0.9, band_name, 
                    ha='center', va='top', fontsize=8)
    
    # Bar chart of powers by frequency band
    band_names = list(bands.keys())
    powers_raw = [band_powers_raw[band] for band in band_names]
    powers_lfp = [band_powers_lfp[band] for band in band_names]
    
    x = np.arange(len(band_names))
    width = 0.35
    
    ax2.bar(x - width/2, powers_raw, width, label='Raw downsampled')
    ax2.bar(x + width/2, powers_lfp, width, label='LFP file')
    ax2.set_title('Power by Frequency Band', fontsize=14)
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Power')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names)
    ax2.legend()
    ax2.grid(True, axis='y')
    
    # Add power difference percentages
    for i, (p_raw, p_lfp) in enumerate(zip(powers_raw, powers_lfp)):
        if p_raw > 0:  # Avoid division by zero
            diff_pct = 100 * (p_lfp - p_raw) / p_raw
            ax2.text(i, max(p_raw, p_lfp) + 0.05 * max(powers_raw + powers_lfp),
                    f"{diff_pct:.1f}%", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def collect_correlations(dat_file, lfp_file, num_channels=384, num_samples=None, test_channels=None):
    """
    Collect correlations between raw and processed data for specific channels.
    
    Returns:
        dict: Dictionary mapping channel numbers to correlation values
    """
    # Define parameters
    raw_fs = 30000         # Raw data sampling rate (Hz)
    lfp_fs = 1250          # LFP sampling rate (Hz)
    downsampling_factor = raw_fs / lfp_fs
    scaling_factor = 0.195  # Typical Neuropixels scaling factor
    
    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 60000)  # Limit to ~2 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)
    
    # Set default test channels if not provided
    if test_channels is None:
        test_channels = [0, 10, 100, 128, 255, 383]
        test_channels = [ch for ch in test_channels if ch < num_channels]
    
    # Calculate correlations for each channel
    correlations = {}
    for ch in test_channels:
        # Get the raw signal and downsample
        raw_signal = raw_data[:, ch]
        indices = np.arange(0, len(raw_signal), downsampling_factor).astype(int)
        downsampled_raw = raw_signal[indices]
        
        # Trim to match sizes
        min_len = min(len(downsampled_raw), len(lfp_data[:, ch]))
        if min_len > 0:
            downsampled_raw = downsampled_raw[:min_len]
            lfp_ch = lfp_data[:min_len, ch]
            
            # Calculate correlation
            correlation = np.corrcoef(downsampled_raw, lfp_ch)[0, 1]
            correlations[ch] = correlation
        else:
            correlations[ch] = np.nan
            
    return correlations

def collect_detailed_metrics(dat_file, lfp_file, num_channels=384, num_samples=None, channel=0):
    """
    Collect detailed metrics for a specific channel.
    
    Returns:
        dict: Dictionary of detailed metrics
    """
    # Define parameters
    raw_fs = 30000         # Raw data sampling rate (Hz)
    lfp_fs = 1250          # LFP sampling rate (Hz)
    downsampling_factor = raw_fs / lfp_fs
    scaling_factor = 0.195  # Typical Neuropixels scaling factor
    
    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 120000)  # Limit to ~4 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)
    
    # Get signal for the specified channel
    raw_signal = raw_data[:, channel]
    lfp_signal = lfp_data[:, channel]
    
    # Downsample raw data
    indices = np.arange(0, len(raw_signal), downsampling_factor).astype(int)
    downsampled_raw = raw_signal[indices]
    
    # Trim to match sizes
    min_len = min(len(downsampled_raw), len(lfp_signal))
    downsampled_raw = downsampled_raw[:min_len]
    lfp_signal = lfp_signal[:min_len]
    
    # Calculate metrics
    correlation = np.corrcoef(downsampled_raw, lfp_signal)[0, 1]
    snr_raw = compute_snr(downsampled_raw)
    snr_lfp = compute_snr(lfp_signal)
    
    # Calculate coherence
    freqs_coh, coherence = compute_coherence(downsampled_raw, lfp_signal, lfp_fs)
    
    # Collect metrics
    metrics = {
        'correlation': correlation,
        'snr_raw': snr_raw,
        'snr_lfp': snr_lfp,
        'mean_coherence': np.mean(coherence),
        'median_coherence': np.median(coherence),
        'max_coherence': np.max(coherence),
        'max_coherence_freq': freqs_coh[np.argmax(coherence)]
    }
    
    return metrics

def collect_frequency_bands(dat_file, lfp_file, num_channels=384, num_samples=None, channel=0):
    """
    Collect frequency band power metrics for a specific channel.
    
    Returns:
        dict: Dictionary of band powers
    """
    # Define parameters
    raw_fs = 30000         # Raw data sampling rate (Hz)
    lfp_fs = 1250          # LFP sampling rate (Hz)
    downsampling_factor = raw_fs / lfp_fs
    scaling_factor = 0.195  # Typical Neuropixels scaling factor
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 80),
        'High Gamma': (80, 200)
    }
    
    # Determine number of samples to read
    if num_samples is None:
        file_size = os.path.getsize(dat_file)
        total_samples = file_size // (num_channels * 2)  # 2 bytes per sample
        num_samples = min(total_samples, 300000)  # Limit to ~10 seconds
    
    # Read binary data
    raw_data = np.fromfile(dat_file, dtype=np.int16, count=num_samples*num_channels)
    lfp_samples = int(num_samples / downsampling_factor)
    lfp_data = np.fromfile(lfp_file, dtype=np.int16, count=lfp_samples*num_channels)
    
    # Apply scaling
    raw_data = raw_data.astype(np.float32) * scaling_factor
    lfp_data = lfp_data.astype(np.float32) * scaling_factor
    
    # Reshape the data
    raw_data = raw_data.reshape(-1, num_channels)
    lfp_data = lfp_data.reshape(-1, num_channels)
    
    # Get signal for the specified channel
    raw_signal = raw_data[:, channel]
    lfp_signal = lfp_data[:, channel]
    
    # Downsample raw data
    indices = np.arange(0, len(raw_signal), downsampling_factor).astype(int)
    downsampled_raw = raw_signal[indices]
    
    # Trim to match sizes
    min_len = min(len(downsampled_raw), len(lfp_signal))
    downsampled_raw = downsampled_raw[:min_len]
    lfp_signal = lfp_signal[:min_len]
    
    # Calculate PSDs
    nperseg = min(4096, min_len)
    freqs_raw, psd_raw = compute_psd(downsampled_raw, lfp_fs, nperseg=nperseg)
    freqs_lfp, psd_lfp = compute_psd(lfp_signal, lfp_fs, nperseg=nperseg)
    
    # Calculate power in each frequency band
    band_results = {}
    
    for band_name, (fmin, fmax) in bands.items():
        # Find indices corresponding to the frequency band
        idx_raw = np.logical_and(freqs_raw >= fmin, freqs_raw <= fmax)
        idx_lfp = np.logical_and(freqs_lfp >= fmin, freqs_lfp <= fmax)
        
        # Calculate power in band (area under PSD curve)
        power_raw = np.trapz(psd_raw[idx_raw], freqs_raw[idx_raw])
        power_lfp = np.trapz(psd_lfp[idx_lfp], freqs_lfp[idx_lfp])
        
        band_results[band_name] = {
            'raw_power': power_raw,
            'lfp_power': power_lfp,
            'freq_range': (fmin, fmax)
        }
    
    return band_results

def main():
    parser = argparse.ArgumentParser(description='Compare raw DAT and processed LFP files')
    parser.add_argument('dat_file', type=str, help='Path to the original .dat file')
    parser.add_argument('lfp_file', type=str, help='Path to the generated .lfp file')
    parser.add_argument('--num-channels', type=int, default=384, help='Number of channels (default: 384)')
    parser.add_argument('--num-samples', type=int, default=30000, help='Number of samples to compare (default: 30000)')
    parser.add_argument('--analysis', choices=['basic', 'detailed', 'frequency', 'all'], 
                      default='all', help='Type of analysis to perform (default: basic)')
    parser.add_argument('--channel', type=int, default=0, help='Channel to analyze in detail (default: 0)')
    parser.add_argument('--output', type=str, help='Path to save summary results (optional)')
    
    args = parser.parse_args()
    
    # Dictionary to store results
    results = {}
    
    # Print header for analysis
    print("\n" + "="*80)
    print(f"LFP Analysis: {os.path.basename(args.dat_file)} vs {os.path.basename(args.lfp_file)}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    if args.analysis == 'basic' or args.analysis == 'all':
        # Collect correlation results
        test_channels = [0, 10, 100, 128, 255, 383]
        test_channels = [ch for ch in test_channels if ch < args.num_channels]
        correlations = collect_correlations(args.dat_file, args.lfp_file, 
                                           args.num_channels, args.num_samples,
                                           test_channels)
        
        # Print correlation table
        print("\nChannel Correlations:")
        print("-"*40)
        corr_table = []
        for ch, corr in correlations.items():
            corr_table.append([ch, f"{corr:.4f}"])
        print(tabulate(corr_table, headers=["Channel", "Correlation"], tablefmt="grid"))
        print("\n")
        
        # Store results
        results['correlations'] = correlations
        
        # Only show plots after printing metrics
        compare_signals_raw(args.dat_file, args.lfp_file, args.num_channels, args.num_samples)
        
    if args.analysis == 'detailed' or args.analysis == 'all':
        # Run detailed analysis and collect metrics
        detailed_results = collect_detailed_metrics(args.dat_file, args.lfp_file,
                                                   args.num_channels, args.num_samples,
                                                   args.channel)
        
        # Print detailed metrics
        print(f"\nDetailed Analysis for Channel {args.channel}:")
        print("-"*60)
        print(f"Correlation:         {detailed_results['correlation']:.4f}")
        print(f"SNR (Raw):           {detailed_results['snr_raw']:.2f} dB")
        print(f"SNR (LFP):           {detailed_results['snr_lfp']:.2f} dB")
        print(f"Mean Coherence:      {detailed_results['mean_coherence']:.4f}")
        print(f"Median Coherence:    {detailed_results['median_coherence']:.4f}")
        print(f"Max Coherence:       {detailed_results['max_coherence']:.4f} at {detailed_results['max_coherence_freq']:.1f} Hz")
        print("\n")
        
        # Store results
        results['detailed'] = detailed_results
        
        # Show plots
        compare_signals_detailed(args.dat_file, args.lfp_file, args.num_channels, args.num_samples, args.channel)
        
    if args.analysis == 'frequency' or args.analysis == 'all':
        # Run frequency band analysis and collect metrics
        band_results = collect_frequency_bands(args.dat_file, args.lfp_file,
                                              args.num_channels, args.num_samples,
                                              args.channel)
        
        # Print frequency band table
        print(f"\nFrequency Band Analysis for Channel {args.channel}:")
        print("-"*80)
        band_table = []
        for band, data in band_results.items():
            if data['raw_power'] > 0:
                diff_pct = 100 * (data['lfp_power'] - data['raw_power']) / data['raw_power']
                band_table.append([
                    band, 
                    f"{data['raw_power']:.4e}",
                    f"{data['lfp_power']:.4e}", 
                    f"{diff_pct:.2f}%"
                ])
            else:
                band_table.append([
                    band, 
                    f"{data['raw_power']:.4e}",
                    f"{data['lfp_power']:.4e}", 
                    "N/A"
                ])
                
        print(tabulate(band_table, 
                     headers=["Band", "Raw Power", "LFP Power", "Difference"],
                     tablefmt="grid"))
        print("\n")
        
        # Store results
        results['frequency_bands'] = band_results
        
        # Show plots
        compare_frequency_bands(args.dat_file, args.lfp_file, args.num_channels, args.num_samples, args.channel)
    
    # Save results to file if output path provided
    if args.output:
        save_path = args.output
        if not save_path.endswith('.txt'):
            save_path += '.txt'
            
        with open(save_path, 'w') as f:
            f.write(f"LFP Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw data file: {args.dat_file}\n")
            f.write(f"LFP data file: {args.lfp_file}\n")
            f.write(f"Number of channels: {args.num_channels}\n\n")
            
            if 'correlations' in results:
                f.write(f"Channel Correlations\n")
                f.write(f"-------------------\n")
                for ch, corr in results['correlations'].items():
                    f.write(f"Channel {ch}: {corr:.4f}\n")
                f.write("\n")
            
            if 'detailed' in results:
                f.write(f"Detailed Analysis (Channel {args.channel})\n")
                f.write(f"--------------------------------\n")
                f.write(f"Correlation: {results['detailed']['correlation']:.4f}\n")
                f.write(f"SNR (Raw): {results['detailed']['snr_raw']:.2f} dB\n")
                f.write(f"SNR (LFP): {results['detailed']['snr_lfp']:.2f} dB\n")
                f.write(f"Mean Coherence: {results['detailed']['mean_coherence']:.4f}\n")
                f.write(f"Median Coherence: {results['detailed']['median_coherence']:.4f}\n")
                f.write(f"Max Coherence: {results['detailed']['max_coherence']:.4f} at {results['detailed']['max_coherence_freq']:.1f} Hz\n\n")
            
            if 'frequency_bands' in results:
                f.write(f"Frequency Band Analysis (Channel {args.channel})\n")
                f.write(f"-------------------------------------\n")
                f.write(f"{'Band':<12} {'Raw Power':>15} {'LFP Power':>15} {'Difference %':>12}\n")
                f.write(f"{'-'*55}\n")
                
                for band, data in results['frequency_bands'].items():
                    raw_power = data['raw_power']
                    lfp_power = data['lfp_power']
                    if raw_power > 0:
                        diff_pct = 100 * (lfp_power - raw_power) / raw_power
                        f.write(f"{band:<12} {raw_power:>15.4e} {lfp_power:>15.4e} {diff_pct:>12.2f}%\n")
                    else:
                        f.write(f"{band:<12} {raw_power:>15.4e} {lfp_power:>15.4e} {'N/A':>12}\n")
                        
        print(f"Analysis results saved to: {save_path}")

if __name__ == "__main__":
    main()
