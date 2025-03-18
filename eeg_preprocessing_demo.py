import os
import argparse
from eeg_preprocessor_2 import EEGPreprocessor

def main():
    """
    Demo script for running the EEG preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description='EEG Preprocessing Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input EEG file path')
    parser.add_argument('--output', '-o', help='Output file path for processed data')
    parser.add_argument('--l_freq', type=float, default=1.0, help='Low cutoff frequency')
    parser.add_argument('--h_freq', type=float, default=30.0, help='High cutoff frequency')
    parser.add_argument('--notch_freq', type=float, default=50.0, help='Notch filter frequency')
    parser.add_argument('--montage', default='standard_1020', help='EEG electrode montage')
    parser.add_argument('--n_ica', type=int, default=20, help='Number of ICA components')
    parser.add_argument('--standardize', default='zscore', 
                       choices=['zscore', 'robust', 'quantile'], help='Standardization method')
    parser.add_argument('--visualize', action='store_true', help='Visualize data after processing')
    parser.add_argument('--run_num', type=int, required=True, help='Trial number')
    parser.add_argument('--desc', type=str, required=True, help='Description of the trial')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Initialize and run the preprocessing pipeline
    print(f"Processing file: {args.input}")
    preprocessor = EEGPreprocessor(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=args.notch_freq,
        montage=args.montage,
        n_ica_components=args.n_ica
    )
    
    try:
        preprocessor.load_data(args.input)
        preprocessor.save_visualizations("before",run_num=args.run_num,desc=args.desc)
        preprocessor.channel_alignment()
        print("Running filters...")
        preprocessor.apply_filters()
        print("Detecting bad channels...")
        preprocessor.detect_bad_channels()
        print("Interpolating bad channels...")
        preprocessor.interpolate_bad_channels()
        print("Running ICA for artifact removal...")
        preprocessor.apply_ica()
        print("Applying adaptive filtering...")
        preprocessor.apply_adaptive_filtering()
        print(f"Standardizing data using {args.standardize} method...")
        preprocessor.standardize_data(method=args.standardize)

        preprocessor.save_visualizations("after",run_num=args.run_num,desc=args.desc)
        
        if args.visualize:
            print("Visualizing processed data...")
            preprocessor.visualize_data()
        
        if args.output:
            print(f"Saving processed data to {args.output}")
            preprocessor.save_processed_data(args.output)
            
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()