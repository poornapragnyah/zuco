import os
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, quantile_transform
from autoreject import AutoReject
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
import warnings


# Create visualization directories
os.makedirs("visualizations/before", exist_ok=True)
os.makedirs("visualizations/after", exist_ok=True)

class EEGPreprocessor:
    """
    A comprehensive EEG preprocessing pipeline for the EEG-to-video generation project.
    Handles importing, filtering, artifact removal, and standardization of EEG data.
    """
    
    def __init__(self, sfreq=None, montage='standard_1020', 
                 l_freq=1, h_freq=30, notch_freq=50, n_ica_components=20):
        """
        Initialize the EEG preprocessor with default parameters.
        
        Parameters:
        -----------
        sfreq : float or None
            Sampling frequency. If None, will be determined from the data.
        montage : str
            EEG electrode montage to use
        l_freq : float
            Low-frequency cutoff for bandpass filter
        h_freq : float
            High-frequency cutoff for bandpass filter
        notch_freq : float
            Frequency to use for notch filtering (typically power line frequency)
        n_ica_components : int
            Number of ICA components to compute
        """
        self.sfreq = sfreq
        self.montage = montage
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.n_ica_components = n_ica_components
        self.raw = None
        self.ica = None
        self.bad_channels = []

    def print_data_info(self, stage):
        """
        Print shape and data type of the raw EEG data.
        
        Parameters:
        -----------
        stage : str
            Description of the current preprocessing stage.
        """
        data = self.raw.get_data()
        print(f"[INFO] {stage} - Data shape: {data.shape}, Data type: {data.dtype}")

    def load_data(self, file_path):
        """
        Load EEG data from various file formats using MNE.
        
        Parameters:
        -----------
        file_path : str
            Path to the EEG data file
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.edf':
                self.raw = mne.io.read_raw_edf(file_path, preload=True)
            elif file_ext == '.fif':
                self.raw = mne.io.read_raw_fif(file_path, preload=True)
            elif file_ext == '.set':
                self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
            elif file_ext == '.cnt':
                self.raw = mne.io.read_raw_cnt(file_path, preload=True)
            elif file_ext == '.bdf':
                self.raw = mne.io.read_raw_bdf(file_path, preload=True)
            elif file_ext == '.npy':
                # Load the .npy file
                data = np.load(file_path)
                
                # Ensure the data is in the correct shape (n_channels, n_samples)
                if data.ndim == 1:
                    data = data[np.newaxis, :]  # Convert to (1, n_samples) if single channel
                elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                    data = data.T  # Transpose to (n_channels, n_samples)
                
                # Create an MNE Info object
                if self.sfreq is None:
                    raise ValueError("Sampling frequency (sfreq) must be provided for .npy files")
                
                info = mne.create_info(
                    ch_names=[f'EEG {i+1}' for i in range(data.shape[0])],  # Default channel names
                    sfreq=self.sfreq,
                    ch_types='eeg'  # Assuming all channels are EEG
                )
                
                # Create the RawArray object
                self.raw = mne.io.RawArray(data, info)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            # Set sampling frequency if provided
            if self.sfreq is not None and self.sfreq != self.raw.info['sfreq']:
                self.raw = self.raw.resample(self.sfreq)
                
            # Set montage
            try:
                self.raw.set_montage(self.montage)
                print(f"Successfully set montage to {self.montage}")
            except Exception as e:
                print(f"Warning: Could not set montage: {e}")
                
            print(f"Loaded EEG data with {len(self.raw.ch_names)} channels at {self.raw.info['sfreq']} Hz")
            self.print_data_info("After loading data")
            return self
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise

    def channel_alignment(self):
        """
        Ensure all channels are correctly positioned and referenced.
        
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Re-reference to average reference
        self.raw = self.raw.set_eeg_reference('average', projection=True)
        self.raw.apply_proj()
        print(f"Projections after applying: {self.raw.info['projs']}")
        print("Re-referenced to average reference")
        self.print_data_info("After channel alignment")
        return self
        
    def apply_filters(self):
        """
        Apply bandpass filter and notch filter to remove frequency artifacts.
        
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Apply bandpass filter
        self.raw = self.raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                                  method='fir', phase='zero-double',fir_window='hamming')
        print(f"Applied bandpass filter {self.l_freq}-{self.h_freq} Hz")
        
        # Apply notch filter to remove power line noise
        self.raw = self.raw.notch_filter(freqs=[self.notch_freq, 2 * self.notch_freq, 3 * self.notch_freq], method='fir',
                                        phase='zero-double',fir_window='hamming')
        print(f"Applied notch filter at {self.notch_freq} Hz")
        self.print_data_info("After filtering")
        return self
        
    def detect_bad_channels(self, z_threshold=2.5):
        """
        Detect bad channels using statistical methods.
        
        Parameters:
        -----------
        z_threshold : float
            Z-score threshold for marking channels as bad
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Get channel variances
        data = self.raw.get_data()
        ch_vars = np.var(data, axis=1)
        
        # Z-score normalization
        z_scores = (ch_vars - np.mean(ch_vars)) / np.std(ch_vars)
        
        # Identify bad channels (too high or too low variance)
        self.bad_channels = [self.raw.ch_names[i] for i in range(len(z_scores)) 
                             if abs(z_scores[i]) > z_threshold]
        
        print(f"Detected {len(self.bad_channels)} bad channels: {self.bad_channels}")
        return self
        
    def interpolate_bad_channels(self):
        """
        Interpolate bad channels using neighboring channels.
        
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if not self.bad_channels:
            print("No bad channels to interpolate")
            return self
            
        # Mark bad channels
        self.raw.info['bads'] = self.bad_channels
        
        # Interpolate bad channels
        self.raw = self.raw.interpolate_bads(reset_bads=True)
        print(f"Interpolated {len(self.bad_channels)} bad channels")
        
        return self

    def apply_ica(self, n_components=None, method='fastica', random_state=42):
        """
        Apply Independent Component Analysis (ICA) for artifact removal.

        Parameters:
        -----------
        n_components : int or None
            Number of ICA components to compute
        method : str
            ICA method to use
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")

        n_components = n_components or self.n_ica_components

        # Create and fit ICA
        self.ica = ICA(n_components=n_components, method=method, random_state=random_state)
        self.ica.fit(self.raw)

        # Auto-detect and plot EOG artifacts (eye blinks and movements)
        eog_channels = ['E8', 'E14', 'E17', 'E21', 'E25', 'E125', 'E126', 'E127', 'E128' ] # Initialize eog_indices to an empty list
        eog_indices = []
        try:
            eog_indices, eog_scores = self.ica.find_bads_eog(self.raw,ch_name=eog_channels)
            print(f"Detected {len(eog_indices)} EOG-related components: {eog_indices}")
        except RuntimeError:
            print("No EOG channels found. Skipping EOG artifact removal.")

        # Combine artifacts to exclude
        exclude_idx = eog_indices

        # Apply ICA to remove artifacts
        self.ica.exclude = exclude_idx
        self.raw = self.ica.apply(self.raw)

        print(f"Applied ICA and removed {len(exclude_idx)} artifact components. Removed artifacts: {exclude_idx}")
        return self
            
    def apply_adaptive_filtering(self, window_size=10, step=1):
        """
        Apply adaptive filtering to handle varying signal quality.
        
        Parameters:
        -----------
        window_size : float
            Window size in seconds
        step : float
            Step size in seconds
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get data and times
        data = self.raw.get_data()
        sfreq = self.raw.info['sfreq']
        window_samples = int(window_size * sfreq)
        step_samples = int(step * sfreq)
        n_channels, n_samples = data.shape
        
        # Process data in sliding windows
        for start_idx in range(0, n_samples - window_samples, step_samples):
            end_idx = start_idx + window_samples
            window_data = data[:, start_idx:end_idx]
            
            # Calculate SNR for this window
            signal_power = np.mean(np.square(window_data))
            noise_est = np.std(np.diff(window_data, axis=1))
            noise_power = np.mean(np.square(noise_est))
            
            # Apply more aggressive filtering to windows with poor SNR
            if noise_power > 0 and signal_power / noise_power < 10:  # Low SNR
                # Design a more restrictive filter for this segment
                narrower_band = [max(self.l_freq + 1, 4), min(self.h_freq - 1, 20)]
                
                for ch_idx in range(n_channels):
                    # Apply a more restrictive filter to this window
                    b, a = signal.butter(4, [f/(sfreq/2) for f in narrower_band], btype='band')
                    data[ch_idx, start_idx:end_idx] = signal.filtfilt(b, a, window_data[ch_idx, :])
        
        # Update raw data with adaptively filtered data
        self.raw._data = data
        print("Applied adaptive filtering based on signal quality")
        self.print_data_info("After adaptive filtering")
        return self

    def standardize_data(self, method='zscore'):
        """
        Standardize data across subjects using different normalization methods.
        
        Parameters:
        -----------
        method : str
            Normalization method ('zscore', 'robust', or 'quantile')
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        data = self.raw.get_data()
        n_channels, n_samples = data.shape
        
        if method == 'zscore':
            scaler = StandardScaler()
            for ch_idx in range(n_channels):
                data[ch_idx, :] = scaler.fit_transform(data[ch_idx, :].reshape(-1, 1)).ravel()
                
        elif method == 'robust':
            scaler = RobustScaler()
            for ch_idx in range(n_channels):
                data[ch_idx, :] = scaler.fit_transform(data[ch_idx, :].reshape(-1, 1)).ravel()
                
        elif method == 'quantile':
            for ch_idx in range(n_channels):
                data[ch_idx, :] = quantile_transform(
                    data[ch_idx, :].reshape(-1, 1), output_distribution='normal'
                ).ravel()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
            
        # Update raw data with standardized data
        self.raw._data = data
        print(f"Applied {method} standardization to data")
        self.print_data_info(f"After standardization ({method})")
        return self
            
    def visualize_data(self, duration=10, start=0):
        """
        Visualize the processed EEG data.
        
        Parameters:
        -----------
        duration : float
            Duration of data to plot in seconds
        start : float
            Start time for plotting in seconds
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Plot raw data
        self.raw.plot(duration=duration, start=start, 
                    scalings='auto', title='Processed EEG Data')
        
        # Plot PSD
        self.raw.plot_psd(fmax=self.h_freq * 2)
        
        return self
        
    def save_processed_data(self, output_path):
        """
        Save the processed EEG data to a file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the processed data
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        self.raw.save(output_path, overwrite=True)
        print(f"Saved processed data to {output_path}")
        return self

    def save_visualizations(self, prefix, run_num, desc):
        try:
            print(f"Saving {prefix} visualizations...")
            os.makedirs(f'visualizations/{prefix}/{run_num}', exist_ok=True)

            # Save description to a text file if provided
            if desc:
                try:
                    with open(f'visualizations/{prefix}/{run_num}/description.txt', 'w') as file:
                        file.write(desc)
                except Exception as e:
                    print(f"Error saving description: {e}")

            # Power Spectral Density (PSD) visualization using .compute_psd().plot()
            try:
                psd = self.raw.compute_psd(fmax=50)
                fig_psd = psd.plot(average=True, show=False)
                fig_psd.savefig(f"visualizations/{prefix}/{run_num}/psd_{prefix}.png")
                plt.close(fig_psd)
            except Exception as e:
                print(f"Error generating or saving PSD plot: {e}")

            # Check and create ICA if not provided
            try:
                if self.ica is None:
                    print("ICA not provided. Creating ICA...")
                    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter='auto')
                    ica.fit(self.raw)
                    self.ica = ica
                else:
                    ica = self.ica
            except Exception as e:
                print(f"Error creating or fitting ICA: {e}")
                return

            # ICA components and sources visualization
            try:
                print("Generating ICA components and sources visualizations...")
                os.makedirs(f"visualizations/{prefix}/{run_num}/ica_components", exist_ok=True)

                # Plot ICA components
                fig_ica_components = ica.plot_components(show=False)
                fig_ica_components.savefig(f"visualizations/{prefix}/{run_num}/ica_components/ica_components_{prefix}.png")
                plt.close(fig_ica_components)

                # Plot ICA sources
                try:
                    fig_ica_sources = ica.plot_sources(self.raw, show=False)
                    fig_ica_sources.savefig(f"visualizations/{prefix}/{run_num}/ica_components/ica_sources_{prefix}.png")
                    plt.close(fig_ica_sources)
                except Exception as e:
                    print(f"Error plotting ICA sources: {e}")

            except Exception as e:
                print(f"Error plotting ICA components or sources: {e}")

        except Exception as e:
            print(f"Unexpected error in save_visualizations: {e}")

    def run_pipeline(self, file_path, output_path=None):
        """
        Execute the full preprocessing pipeline on a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the EEG data file
        output_path : str or None
            Path to save the processed data
            
        Returns:
        --------
        self.raw : instance of mne.io.Raw
            The processed EEG data
        """
        self.load_data(file_path)
        self.channel_alignment()
        self.apply_filters()
        self.detect_bad_channels()
        self.interpolate_bad_channels()
        self.apply_ica()
        self.apply_adaptive_filtering()
        self.standardize_data()
        
        if output_path:
            self.save_processed_data(output_path)
            
        return self.raw