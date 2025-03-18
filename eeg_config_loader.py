import os
import yaml
import argparse
from eeg_preprocessor import EEGPreprocessor
from eeg_quality_assesment import assess_eeg_quality

class ConfigLoader:
    """
    Load and apply configuration settings for the EEG preprocessing pipeline.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration loader.
        
        Parameters:
        -----------
        config_path : str or None
            Path to the configuration file
        """
        self.config = None
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path):
        """
        Load configuration from a YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file
            
        Returns:
        --------
        dict
            The loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return self.config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
            
    def save_config(self, config_path):
        """
        Save the current configuration to a YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to save the configuration
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
            
    def create_preprocessor(self):
        """
        Create an EEG preprocessor based on the loaded configuration.
        
        Returns:
        --------
        EEGPreprocessor
            Configured EEG preprocessor instance
        """
        if not self.config:
            raise ValueError("No configuration loaded")
            
        # Extract configuration parameters
        params = {
            'sfreq': self.config['data'].get('sfreq'),
            'montage': self.config['data'].get('montage', 'standard_1020'),
            'l_freq': self.config['filters'].get('l_freq', 1.0),
            'h_freq': self.config['filters'].get('h_freq', 30.0),
            'notch_freq': self.config['filters'].get('notch_freq', 50.0),
            'n_ica_components': self.config['ica'].get('n_components', 20)
        }
        
        # Create and return preprocessor
        return EEGPreprocessor(**params)
        
    def update_config(self, section, key, value):
        """
        Update a specific configuration parameter.
        
        Parameters:
        -----------
        section : str
            Configuration section
        key : str
            Parameter key
        value : any
            New parameter value
            
        Returns:
        --------
        self : instance
            The instance itself for chaining
        """
        if not self.config:
            self.config = {}
            
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
        return self
        
    def run_pipeline_from_config(self, input_path, output_path=None):
        """
        Run the full preprocessing pipeline using the loaded configuration.
        
        Parameters:
        -----------
        input_path : str
            Path to input EEG file
        output_path : str or None
            Path to save processed data
            
        Returns:
        --------
        mne.io.Raw
            Processed EEG data
        """
        preprocessor = self.create_preprocessor()
        
        # Load data
        preprocessor.load_data(input_path)
        
        # Channel alignment
        preprocessor.channel_alignment()
        
        # Apply filters with parameters from config
        filter_method = self.config['filters'].get('filter_method', 'fir')
        if filter_method == 'fir':
            phase = self.config['filters'].get('fir_params', {}).get('phase', 'zero-double')
            preprocessor.raw = preprocessor.raw.filter(
                l_freq=preprocessor.l_freq, 
                h_freq=preprocessor.h_freq,
                method='fir',
                phase=phase
            )
        else:
            order = self.config['filters'].get('iir_params', {}).get('order', 4)
            ftype = self.config['filters'].get('iir_params', {}).get('ftype', 'butter')
            preprocessor.raw = preprocessor.raw.filter(
                l_freq=preprocessor.l_freq, 
                h_freq=preprocessor.h_freq,
                method='iir',
                iir_params={'order': order, 'ftype': ftype}
            )
        
        # Apply notch filter
        preprocessor.raw = preprocessor.raw.notch_filter(
            freqs=preprocessor.notch_freq, 
            method=filter_method
        )
        
        # Detect bad channels
        if self.config['bad_channels'].get('auto_detect', True):
            z_threshold = self.config['bad_channels'].get('z_threshold', 3.0)
            preprocessor.detect_bad_channels(z_threshold=z_threshold)
            
            # Add known bad channels
            known_bad = self.config['bad_channels'].get('known_bad_channels', [])
            preprocessor.bad_channels.extend(known_bad)
            
            # Remove duplicates
            preprocessor.bad_channels = list(set(preprocessor.bad_channels))
            
            # Interpolate bad channels
            preprocessor.interpolate_bad_channels()
        
        # Apply ICA
        if self.config['ica'].get('auto_detect_artifacts', True):
            method = self.config['ica'].get('method', 'fastica')
            n_components = self.config['ica'].get('n_components', 20)
            preprocessor.apply_ica(n_components=n_components, method=method)
        
        # Apply adaptive filtering
        if self.config['adaptive_filtering'].get('enabled', True):
            window_size = self.config['adaptive_filtering'].get('window_size', 10)
            step = self.config['adaptive_filtering'].get('step', 1)
            preprocessor.apply_adaptive_filtering(window_size=window_size, step=step)
        
        # Standardize data
        method = self.config['standardization'].get('method', 'zscore')
        preprocessor.standardize_data(method=method)
        
        # Save processed data if output path provided
        if output_path:
            preprocessor.save_processed_data(output_path)
        
        # Run quality assessment if configured
        if self.config['quality'].get('run_assessment', True):
            metrics = assess_eeg_quality(preprocessor.raw)
            
            # Check against thresholds
            thresholds = self.config['quality'].get('thresholds', {})
            passed_qa = True
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metric == 'snr':
                        if metrics[metric] < threshold:
                            print(f"Warning: SNR ({metrics[metric]:.2f} dB) below threshold ({threshold:.2f} dB)")
                            passed_qa = False
                    elif metric == 'spike_rate':
                        if metrics[metric] > threshold:
                            print(f"Warning: Spike rate ({metrics[metric]:.2f}/s) above threshold ({threshold:.2f}/s)")
                            passed_qa = False
                    elif metric == 'stationarity':
                        if metrics[metric] > threshold:
                            print(f"Warning: Non-stationarity ({metrics[metric]:.2f}) above threshold ({threshold:.2f})")
                            passed_qa = False
            
            if passed_qa:
                print("Quality assessment passed all thresholds")
            else:
                print("Quality assessment failed some thresholds")
        
        return preprocessor.raw

def main():
    """
    Main function for configuration-based EEG preprocessing.
    """
    parser = argparse.ArgumentParser(description='EEG Preprocessing with Configuration')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--input', '-i', required=True, help='Input EEG file path')
    parser.add_argument('--output', '-o', help='Output file path for processed data')
    parser.add_argument('--update', '-u', nargs=3, action='append',
                       metavar=('SECTION', 'KEY', 'VALUE'),
                       help='Update configuration parameter (can be used multiple times)')
    parser.add_argument('--save-config', '-s', help='Save updated configuration to file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load configuration
    loader = ConfigLoader(args.config)
    
    # Update configuration if requested
    if args.update:
        for section, key, value in args.update:
            # Try to convert value to appropriate type
            try:
                # Try to convert to number if possible
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if conversion fails
                pass
                
            loader.update_config(section, key, value)
    
    # Save updated configuration if requested
    if args.save_config:
        loader.save_config(args.save_config)
    
    # Run preprocessing pipeline
    try:
        raw = loader.run_pipeline_from_config(args.input, args.output)
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()