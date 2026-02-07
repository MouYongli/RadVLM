from PIL import Image
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')
import pydicom
import numpy as np
import subprocess
from pathlib import Path
from multiprocessing import Pool
from src.radvlm.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def copy_subdir(args):
    """Copy a single subdirectory."""
    subdir, new_base = args
    # Extract the relative path structure (e.g., "p10/p10000032")
    parent_name = os.path.basename(os.path.dirname(subdir))
    subdir_name = os.path.basename(subdir)
    if subdir.split("/")[-2].startswith("p") and len(subdir.split("/")[-2]) == 3:
        # Just get the subfolder name, don't recreate parent structure      
        dest = os.path.join(new_base, subdir_name)
    else:
        # Create parent directory in destination if needed
        dest_parent = os.path.join(new_base, parent_name)
        os.makedirs(dest_parent, exist_ok=True)
        
        dest = os.path.join(dest_parent, subdir_name)
    # print(f"Copying {parent_name}/{subdir_name} to {dest}...", flush=True)
    try:
        subprocess.run(
            ["rsync", "-ah", "--ignore-existing", f"{subdir}/", f"{dest}/"],
            check=True,
            capture_output=True
        )
        return f"✓ {parent_name}/{subdir_name}"
    except subprocess.CalledProcessError as e:
        print(f"Error copying {parent_name}/{subdir_name}: {e}", flush=True)
        return f"✗ {parent_name}/{subdir_name}: {e}"

def copy_data_to_new_dir(old_data_dir: str, new_data_dir: str):   
    """
    Copy all files from the old data directory to the new data directory.

    Args:
        old_data_dir (str): Path to the old data directory.
        new_data_dir (str): Path to the new data directory.
    Returns:
        None
    """
    try:        
        if not os.path.exists(old_data_dir):
            raise FileNotFoundError(f"Source directory does not exist: {old_data_dir}")
        
        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir, exist_ok=True)
        
        print(f"Copying from {old_data_dir} → {new_data_dir}", flush=True)
        
        # Get all nested subdirectories
        subdirs = []
        for top_level_dir in Path(old_data_dir).iterdir():
            if top_level_dir.is_dir():
                # print(f"Scanning top-level directory: {top_level_dir}", flush=True)
                # print(old_data_dir.split("/")[-1], flush=True)
                if old_data_dir.split("/")[-1].startswith("p") and len(old_data_dir.split("/")[-1]) == 3:
                    subdirs.append(str(top_level_dir))
                else:
                    nested_dirs = [str(d) for d in top_level_dir.iterdir() if d.is_dir()]
                    subdirs.extend(nested_dirs)
        
        if not subdirs:
            print("No subdirectories found!")
            return
        
        total = len(subdirs)
        print(f"Found {total} patient directories. Copying with 4 parallel workers in batches...", flush=True)
        
        # Process in chunks to limit memory usage
        chunk_size = 100  # Process 100 directories at a time
        completed = 0
        
        for i in range(0, len(subdirs), chunk_size):
            chunk = subdirs[i:i+chunk_size]
            args = [(subdir, new_data_dir) for subdir in chunk]
            
            print(f"\nProcessing batch {i//chunk_size + 1} ({completed}/{total} completed)...", flush=True)
            
            with Pool(processes=4) as pool:
                for result in pool.imap_unordered(copy_subdir, args):
                    # print(result, flush=True)
                    completed += 1
            
            # Pool is closed and joined here, freeing memory
            print(f"Batch complete. Progress: {completed}/{total}", flush=True)
        
        print("\nData copying completed.")
        return
    except Exception as e:
        print(f"Error copying data: {e}")
        raise ValueError(f"Error copying data: {e}")


def transform_single_dcm(dcm_path):
    """Transform a single DICOM file to JPEG."""
    try:
        # Check if the JPEG file already exists
        jpeg_path = dcm_path.replace('.dcm', '.jpg')
        if os.path.exists(jpeg_path):
            return f"⊙ {os.path.basename(dcm_path)} (skipped)"
        
        # Read the DICOM file
        dcm_image = pydicom.dcmread(dcm_path)
        pixel_array = dcm_image.pixel_array.astype(np.float32)

        # Apply rescale slope and intercept
        # Rescale slope and intercept are used to convert pixel values to Hounsfield units (HU) in CT images
        # They are not always present, so we use default values of 1 and 0 if they are not found
        slope = getattr(dcm_image, 'RescaleSlope', 1)
        intercept = getattr(dcm_image, 'RescaleIntercept', 0)
        pixel_array = pixel_array * slope + intercept

        # Apply windowing if available
        # Windowing is used to enhance the contrast of the image
        window_center = getattr(dcm_image, 'WindowCenter', None)
        window_width = getattr(dcm_image, 'WindowWidth', None)
        if window_center and window_width:
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]
            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, lower, upper)
        else:
            lower, upper = pixel_array.min(), pixel_array.max()

        # Normalize to 8-bit
        pixel_array = ((pixel_array - lower) / (upper - lower + 1e-8) * 255).astype(np.uint8)

        # Handle PhotometricInterpretation (invert if MONOCHROME1)
        photometric = getattr(dcm_image, 'PhotometricInterpretation', 'MONOCHROME2')
        if photometric == 'MONOCHROME1':
            pixel_array = 255 - pixel_array

        image = Image.fromarray(pixel_array, mode='L')
        image.save(jpeg_path, 'JPEG')

        # Remove the original DICOM file to save space
        os.remove(dcm_path)
        
        return f"✓ {os.path.basename(dcm_path)}"
    except Exception as e:
        print(f"Error transforming {dcm_path}: {e}", flush=True)
        return f"✗ {os.path.basename(dcm_path)}: {e}"


def transform_dcm_to_jpg(data_dir: str) -> str:
    """
    Transform all DICOM images in the given directory to JPEG format.

    Args:
        data_dir (str): Path to the directory containing DICOM images.

    Returns:
        None
    """
    try:
        print("Transforming images to .jpg")
        if not os.path.exists(data_dir):
            print("Dataset directory does not exist.")
            raise FileNotFoundError("Dataset directory does not exist.")
        
        # Collect all DICOM files
        dcm_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.dcm'):
                    dcm_files.append(os.path.join(root, file))
        
        if not dcm_files:
            print("No DICOM files found!")
            return
        
        total = len(dcm_files)
        print(f"Found {total} DICOM files. Transforming with 4   parallel workers in batches...", flush=True)
        
        # Process in chunks to limit memory usage
        chunk_size = 100  # Process 100 files at a time
        completed = 0
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for i in range(0, len(dcm_files), chunk_size):
            chunk = dcm_files[i:i+chunk_size]
            
            print(f"\nProcessing batch {i//chunk_size + 1} ({completed}/{total} completed)...", flush=True)
            
            with Pool(processes=4) as pool:
                for result in pool.imap_unordered(transform_single_dcm, chunk):
                    # print(result, flush=True)
                    completed += 1
                    if result.startswith("✓"):
                        success_count += 1
                    elif result.startswith("⊙"):
                        skipped_count += 1
                    elif result.startswith("✗"):
                        error_count += 1
            
            # Pool is closed and joined here, freeing memory
            print(f"Batch complete. Progress: {completed}/{total}", flush=True)
        
        print("\nTransformed files to .jpg")
        print(f"  Successfully transformed: {success_count}")
        print(f"  Skipped (already .jpg): {skipped_count}")
        print(f"  Errors: {error_count}")
        return
    except Exception as e:
        print(f"Error transforming images: {e}")
        raise ValueError(f"Error transforming images: {e}")


def delete_index_files(data_dir):
    """
    Delete all index.html files in the given directory and its subdirectories.
    
    Args:
        data_dir (str): Path to the directory to search for .idx files.

    Returns:
        None
    """

    try:  
        print("Deleting index files")
        # Check if data_dir exists
        if not os.path.exists(data_dir):
            print("Dataset directory does not exist.")
            raise FileNotFoundError("Dataset directory does not exist.")
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file == 'index.html':
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    # print(f"Deleted {file_path}")
        print("Deleted all index files")
        return
    except Exception as e:
        print(f"Error deleting index files: {e}")
        raise ValueError(f"Error deleting index files: {e}")

def resize_single_image(args):
    """Resize a single image file."""
    image_path, max_resolution = args
    
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            
            # Check if resizing is needed
            if w > max_resolution or h > max_resolution:
                scale = max_resolution / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                new_img = img.resize(new_size, Image.BICUBIC)
                new_img.save(image_path)
                return f"✓ Resized: {os.path.basename(image_path)} ({w}x{h} → {new_size[0]}x{new_size[1]})"
            else:
                # Image already within limits, no action needed
                return f"○ Skipped: {os.path.basename(image_path)} (already {w}x{h})"
                
    except Exception as e:
        print(f"Error resizing {image_path}: {e}", flush=True)
        return f"✗ Error: {os.path.basename(image_path)}: {e}"


def resize_images_batch(data_dir, max_resolution=768, num_workers=4):
    """
    Resize images to a maximum resolution while maintaining aspect ratio using batch processing.
    
    Args:
        data_dir (str): Path to the directory containing images.
        max_resolution (int): Maximum resolution for the longest side of the image.
        num_workers (int): Number of parallel workers for processing.
        
    Returns:
        None
    """
    try:
        print(f"Resizing images to max resolution {max_resolution}", flush=True)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        
        # Collect all image paths
        print("Scanning for images...", flush=True)
        image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print("No images found!")
            return
        
        total = len(image_paths)
        print(f"Found {total} images. Processing with {num_workers} parallel workers in batches...", flush=True)
        
        # Process in chunks to limit memory usage
        chunk_size = 100  # Process 100 images at a time
        completed = 0
        resized_count = 0
        skipped_count = 0
        error_count = 0
        
        for i in range(0, len(image_paths), chunk_size):
            chunk = image_paths[i:i+chunk_size]
            args = [(img_path, max_resolution) for img_path in chunk]
            
            print(f"\nProcessing batch {i//chunk_size + 1} ({completed}/{total} completed)...", flush=True)
            
            with Pool(processes=num_workers) as pool:
                for result in pool.imap_unordered(resize_single_image, args):
                    completed += 1
                    
                    # Count results
                    if result.startswith("✓"):
                        resized_count += 1
                    elif result.startswith("○"):
                        skipped_count += 1
                    elif result.startswith("✗"):
                        error_count += 1
                        print(result, flush=True)  # Print errors immediately
            
            # Pool is closed and joined here, freeing memory
            print(f"Batch complete. Progress: {completed}/{total}", flush=True)
        
        print(f"\nImage resizing completed:")
        print(f"  Total processed: {total}")
        print(f"  Resized: {resized_count}")
        print(f"  Skipped (already correct size): {skipped_count}")
        print(f"  Errors: {error_count}")
        
        return
        
    except Exception as e:
        print(f"Error resizing images: {e}")
        raise ValueError(f"Error resizing images: {e}")

if __name__ == "__main__":
    
    try:
        old_data_dir = DATA_RAW_DIR
        new_data_dir = DATA_PROCESSED_DIR

        # check if old_data_dir exists
        if os.path.exists(old_data_dir):
            print(f"Using old data directory: {old_data_dir}")
            copy_data_to_new_dir(old_data_dir, new_data_dir)
            delete_index_files(new_data_dir)
            transform_dcm_to_jpg(new_data_dir)
            resize_images_batch(new_data_dir, max_resolution=768)
        else:
            print(f"Old data directory {old_data_dir} does not exist")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
