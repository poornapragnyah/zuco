#!/usr/bin/env python3
"""
Main script for running the complete EEG preprocessing pipeline (Step 1)
for the EEG-to-video generation project.
"""

import os
import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Import our pipeline components
from eeg_preprocessor import EEGPreprocessor
from eeg_config_loader import ConfigLoader
from eeg_quality_assesment import assess_eeg_quality

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('eeg-pipeline')

def process_file(args):
    """
    Process a single EEG file with configuration.
    
    Parameters:
    -----------
    args : tuple
        (input_file, output_file, config_path, debug)
        
    Returns:
    --------
    dict
        Processing results and metrics
    """
    input_file, output_file, config_path, debug = args
    
    try:
        # Set up file-specific logging
        if debug:
            log_file = os.path.splitext(output_file)[0] + '.log'
            file_logger = logging.FileHandler(log_file)
            file_logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_logger)
        
        logger.info(f"Processing {input_file}")
        start_time = time.time()
        
        # Load configuration and run pipeline
        loader = ConfigLoader(config_path)
        raw = loader.run_pipeline_from_config(input_file, output_file)
        
        # Run quality assessment
        qa_dir = os.path.join(os.path.dirname(output_file), 'quality_assessment')
        os.makedirs(qa_dir, exist_ok=True)
        metrics = assess_eeg_quality(output_file, qa_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Successfully processed {input_file} in {processing_time:.2f} seconds")
        
        # Prepare result data
        result = {
            'input_file': input_file,
            'output_file': output_file,
            'processing_time': processing_time,
            'success': True,
            'metrics': metrics
        }
        
        # Clean up file logger if used
        if debug:
            logger.removeHandler(file_logger)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        if debug:
            logger.exception("Detailed error:")
        
        # Clean up file logger if used
        if debug:
            logger.removeHandler(file_logger)
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'success': False,
            'error': str(e)
        }

def main():
    """
    Main function to run the EEG preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description='EEG Preprocessing Pipeline (Step 1)')
    parser.add_argument('--input_dir', '-i', help='Input directory containing EEG files')
    parser.add_argument('--input_file', '-f', help='Single input EEG file')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory for processed files')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--file_ext', default='.edf', help='File extension to process')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    parser.add_argument('--report', '-r', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.input_dir and not args.input_file:
        parser.error("Either --input_dir or --input_file must be specified")
    
    if args.input_dir and args.input_file:
        parser.error("Only one of --input_dir or --input_file can be specified")
    
    # Setup debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        debug_log_path = os.path.join(args.output_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        debug_handler = logging.FileHandler(debug_log_path)
        debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(debug_handler)
        logger.debug("Debug logging enabled")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files to process
    files_to_process = []
    
    if args.input_file:
        # Process a single file
        input_file = os.path.abspath(args.input_file)
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} does not exist")
            return 1
        
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '_processed.fif')
        files_to_process.append((input_file, output_file, args.config, args.debug))
    else:
        # Process all files in directory
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.exists(input_dir):
            logger.error(f"Input directory {input_dir} does not exist")
            return 1
        
        for filename in os.listdir(input_dir):
            if filename.endswith(args.file_ext):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '_processed.fif')
                files_to_process.append((input_file, output_file, args.config, args.debug))
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process files
    results = []
    
    if args.n_jobs > 1:
        # Use multiprocessing for parallel processing
        n_jobs = min(args.n_jobs, cpu_count())
        logger.info(f"Processing files in parallel with {n_jobs} workers")
        
        with Pool(n_jobs) as pool:
            results = pool.map(process_file, files_to_process)
    else:
        # Process files sequentially
        logger.info("Processing files sequentially")
        for file_args in files_to_process:
            results.append(process_file(file_args))
    
    # Summarize results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Processing complete: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        logger.info("Failed files:")
        for f in failed:
            logger.info(f"  - {f['input_file']}: {f.get('error', 'Unknown error')}")
    
    # Generate HTML report if requested
    if args.report:
        logger.info("Generating HTML report")
        report_path = os.path.join(args.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        try:
            generate_html_report(results, report_path)
            logger.info(f"Report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    return 0 if not failed else 1

def generate_html_report(results, output_path):
    """
    Generate an HTML report of the processing results.
    
    Parameters:
    -----------
    results : list
        List of processing results
    output_path : str
        Path to save the HTML report
    """
    import pandas as pd
    from jinja2 import Template
    
    # Convert results to DataFrame for easier processing
    data = []
    for r in results:
        row = {
            'input_file': r['input_file'],
            'output_file': r['output_file'],
            'success': r['success'],
            'processing_time': r.get('processing_time', None),
        }
        
        # Add metrics if available
        if r.get('metrics'):
            for k, v in r['metrics'].items():
                row[f'metric_{k}'] = v
        
        # Add error if available
        if not r['success']:
            row['error'] = r.get('error', 'Unknown error')
            
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create basic HTML template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EEG Preprocessing Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            tr.success { background-color: #dff0d8; }
            tr.failure { background-color: #f2dede; }
            .summary { margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
            .timestamp { color: #666; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>EEG Preprocessing Report</h1>
        <div class="timestamp">Generated on {{ timestamp }}</div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total files: {{ total_files }}</p>
            <p>Successful: {{ successful_files }}</p>
            <p>Failed: {{ failed_files }}</p>
            <p>Average processing time: {{ avg_time }} seconds</p>
        </div>
        
        <h2>File Details</h2>
        <table>
            <tr>
                <th>Status</th>
                <th>Input File</th>
                <th>Processing Time (s)</th>
                <th>Details</th>
            </tr>
            {% for file in files %}
            <tr class="{{ 'success' if file.success else 'failure' }}">
                <td>{{ 'Success' if file.success else 'Failed' }}</td>
                <td>{{ file.input_file }}</td>
                <td>{{ "%.2f"|format(file.processing_time) if file.processing_time else 'N/A' }}</td>
                <td>
                    {% if file.success %}
                        {% for key, value in file.metrics.items() %}
                            {{ key }}: {{ value }}<br>
                        {% endfor %}
                    {% else %}
                        Error: {{ file.error }}
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    
    # Prepare template data
    template_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': len(results),
        'successful_files': sum(1 for r in results if r['success']),
        'failed_files': sum(1 for r in results if not r['success']),
        'avg_time': f"{sum(r.get('processing_time', 0) for r in results if r.get('processing_time')) / max(1, sum(1 for r in results if r.get('processing_time'))):.2f}",
        'files': results
    }
    
    # Generate HTML report
    template = Template(template_str)
    html = template.render(**template_data)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    sys.exit(main())