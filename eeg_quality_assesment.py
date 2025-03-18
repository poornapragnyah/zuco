import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from scipy import io
import h5py


def assess_eeg_quality(file_path, output_dir=None):
    """
    Assess the quality of preprocessed EEG data.
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file
    output_dir : str or None
        Directory to save quality assessment results
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    raw = mne.io.read_raw(file_path, preload=True)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(file_path).split('.')[0]

    # # Create output directory if needed
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)
    #     if original_file_path:
    #         basename = os.path.basename(original_file_path).split('.')[0]
    #     else:
    #         # Generate a timestamp-based name if original path not provided
    #         basename = f"eeg_quality_assessment_{int(time.time())}"
    
    # Extract data
    data = raw.get_data()
    n_channels, n_samples = data.shape
    sfreq = raw.info['sfreq']
    
    # Initialize quality metrics
    metrics = {}
    
    # 1. Signal-to-Noise Ratio (SNR)
    # Estimate signal power in alpha band (8-12 Hz)
    spectrum = raw.compute_psd(method='welch', fmin=1, fmax=50, n_fft=int(sfreq * 2))
    psd, freqs = spectrum.get_data(return_freqs=True)
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
    signal_power = np.mean(psd[:, alpha_idx], axis=1)
    
    # Estimate noise power in 48-52 Hz (near line noise)
    noise_idx = np.logical_and(freqs >= 48, freqs <= 52)
    noise_power = np.mean(psd[:, noise_idx], axis=1)
    
    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    # 2. Stationarity (variance over time)
    window_size = int(sfreq * 10)  # 10-second windows
    n_windows = n_samples // window_size
    
    if n_windows > 1:
        window_vars = np.zeros((n_channels, n_windows))
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            window_vars[:, i] = np.var(data[:, start:end], axis=1)
            
        # Calculate coefficient of variation of variance across windows
        cv_var = np.std(window_vars, axis=1) / np.mean(window_vars, axis=1)
        metrics['stationarity'] = np.mean(cv_var)
    else:
        metrics['stationarity'] = np.nan
    
    # 3. Channel correlation (average correlation between channels)
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
    metrics['channel_correlation'] = np.mean(np.abs(corr_matrix))
    
    # 4. Artifact metrics
    # Count voltage jumps/spikes
    diff_data = np.diff(data, axis=1)
    threshold = 5 * np.std(diff_data, axis=1, keepdims=True)
    spikes = np.sum(np.abs(diff_data) > threshold, axis=1)
    metrics['spike_rate'] = np.sum(spikes) / (n_samples / sfreq)  # Spikes per second
    
    # 5. Spectral power distribution
    # Calculate power in standard frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    for band_name, (fmin, fmax) in bands.items():
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psd[:, band_idx], axis=(0, 1))
        metrics[f'{band_name}_power'] = band_power
    
    # 6. Calculate alpha/theta ratio (indicator of alertness)
    metrics['alpha_theta_ratio'] = metrics['alpha_power'] / metrics['theta_power']
    
    # Visualize results if output directory is provided
    if output_dir:
        # Plot PSD
        plt.figure(figsize=(10, 6))
        plt.semilogy(freqs, np.mean(psd, axis=0))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (µV²/Hz)')
        plt.title('Power Spectral Density')
        for band_name, (fmin, fmax) in bands.items():
            plt.axvspan(fmin, fmax, alpha=0.3, label=band_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{basename}_psd.png'))
        
        # Plot data quality metrics
        plt.figure(figsize=(10, 6))
        metrics_to_plot = {
            'SNR (dB)': metrics['snr'],
            'Non-stationarity': metrics['stationarity'],
            'Channel correlation': metrics['channel_correlation'],
            'Spike rate (per sec)': metrics['spike_rate'],
            'Alpha/Theta ratio': metrics['alpha_theta_ratio']
        }
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.ylabel('Value')
        plt.title('EEG Quality Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{basename}_metrics.png'))
        
        # Plot channel SNR as topographic map
        try:
            plt.figure(figsize=(8, 8))
            mne.viz.plot_topomap(snr, raw.info, cmap='viridis')
            plt.colorbar(label='SNR (dB)')
            plt.title('Signal-to-Noise Ratio by Channel')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{basename}_snr_topo.png'))
        except Exception as e:
            print(f"Could not generate topographic map: {e}")
    
    return metrics

def main():
    """
    Main function for EEG quality assessment.
    """
    parser = argparse.ArgumentParser(description='EEG Quality Assessment')
    parser.add_argument('--input', '-i', required=True, help='Input EEG file path or directory')
    parser.add_argument('--output_dir', '-o', default='quality_assessment', 
                        help='Output directory for quality assessment results')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Process all .fif files in directory
        files = [f for f in os.listdir(args.input) if f.endswith('.fif')]
        for file in files:
            file_path = os.path.join(args.input, file)
            metrics = assess_eeg_quality(file_path, args.output_dir)
            print(f"Metrics for {file}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    else:
        metrics = assess_eeg_quality(args.input, args.output_dir)
        print("Quality metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
