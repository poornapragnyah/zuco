# EEG Preprocessing and Quality Assessment

This README provides instructions on how to preprocess EEG data and assess its quality using the provided Python scripts.

## Prerequisites
Make sure you have the required Python packages installed. You can typically install the necessary packages with:
```
pip install mne argparse numpy scipy scikit-learn matplotlib
```

## Preprocessing EEG Data
The preprocessing script is called `eeg_preprocessing_demo.py`. It applies various signal processing steps, including filtering and Independent Component Analysis (ICA), and optionally standardizes the data.

### Command Format
```
python eeg_preprocessing_demo.py --input <input_file> --output <output_file> --montage <montage_name> --run_num <trial_number> --desc <description> [options]
```

### Example Usage
```
python eeg_preprocessing_demo.py --input ../ZDM_SNR6_EEG_raw.fif --output ./out.fif --montage GSN-HydroCel-128 --run_num 2 --desc "seed" --l_freq 0.5
```

### Arguments
- `--input`, `-i`: Path to the input EEG file (required).
- `--output`, `-o`: Path to save the processed data.
- `--l_freq`: Low cutoff frequency for filtering (default: 1.0 Hz).
- `--h_freq`: High cutoff frequency for filtering (default: 30.0 Hz).
- `--notch_freq`: Frequency for notch filtering to remove line noise (default: 50.0 Hz).
- `--montage`: EEG electrode montage to use (default: standard_1020).
- `--n_ica`: Number of ICA components to apply (default: 20).
- `--standardize`: Method for data standardization (options: zscore, robust, quantile; default: zscore).
- `--visualize`: Visualize data after processing.
- `--run_num`: Trial number (required).
- `--desc`: Description of the trial (required).

## Quality Assessment
Once preprocessing is complete, you can assess the quality of the processed EEG data using the `eeg_quality_assesment.py` script.

### Command Format
```
python eeg_quality_assesment.py --input <output_file>
```

### Example Usage
```
python eeg_quality_assesment.py --input out.fif
```

This script evaluates the quality of the processed EEG data and provides metrics or visualizations as configured.

