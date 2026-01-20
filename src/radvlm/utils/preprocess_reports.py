from PIL import Image
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')
import pydicom
import numpy as np
import subprocess
from pathlib import Path
from multiprocessing import Pool
from src.radvlm.utils.config import DATA_RAW_DIR_FULL_DATASET, DATA_PROCESSED_DIR_FULL_DATASET, DATA_PROCESSED_DIR

def extract_sections_simple(report):
    sections = {}
    lines = report.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if line is a section header
        if line.strip() and line.strip().isupper() and ':' in line:
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            parts = line.split(':', 1)
            current_section = parts[0].strip().lower()
            current_content = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def preprocess_single_report(report_path: str) -> str:
    """
    Preprocess radiology report text by removing unnecessary sections and formatting.
    
    Args:
        report_text: Original report text.
    Returns:
        Preprocessed report text.
    """
    
    try:
        with open(report_path, 'r') as f:
            report_text = f.read()
    except Exception as e:
        return f"✗ Error reading {report_path}: {e}"

    sections = extract_sections_simple(report_text)
    
    # Keep only relevant sections
    relevant_sections = ['findings', 'impression']
    processed_sections = []
    
    for sec in relevant_sections:
        if sec in sections:
            processed_sections.append(f"{sec.upper()}:\n\n{sections[sec]}")
    
    processed_report = 'FINAL REPORT\n\n'
    processed_report += '\n\n'.join(processed_sections).strip()
    
    try:
        with open(report_path, 'w') as f:
            f.write(processed_report)
        return f"✓ Successfully processed {report_path}"
    except Exception as e:
        return f"✗ Error writing {report_path}: {e}"


def preprocess_reports(data_dir):
    """
    Preprocess all reports in the given directory.
    """
    print("Preprocessing reports...", flush=True)
    if not os.path.exists(data_dir):
        print("Dataset directory does not exist.")
        raise FileNotFoundError("Dataset directory does not exist.")
    
    # Collect all text files
    txt_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        print("No text files found!")
        return
    
    total = len(txt_files)
    print(f"Found {total} text files. Transforming with 2 parallel workers in batches...", flush=True)
    
    # Process in chunks to limit memory usage
    chunk_size = 50  # Process 50 files at a time
    completed = 0
    success_count = 0
    error_count = 0
    
    for i in range(0, len(txt_files), chunk_size):
        chunk = txt_files[i:i+chunk_size]
        
        print(f"\nProcessing batch {i//chunk_size + 1} ({completed}/{total} completed)...", flush=True)
        
        with Pool(processes=2) as pool:
            for result in pool.imap_unordered(preprocess_single_report, chunk):
                # print(result, flush=True)
                completed += 1
                if result.startswith("✓"):
                    success_count += 1
                elif result.startswith("✗"):
                    error_count += 1
        
        # Pool is closed and joined here, freeing memory
        print(f"Batch complete. Progress: {completed}/{total}", flush=True)
    
    print("\nPreprocessing complete!", flush=True)
    print(f"  Successfully preprocessed: {success_count}")
    print(f"  Errors: {error_count}")
    return
    

if __name__ == "__main__":
    data_dir = DATA_PROCESSED_DIR
    preprocess_reports(data_dir)