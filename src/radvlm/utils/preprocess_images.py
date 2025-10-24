from PIL import Image
import os
import pydicom
import numpy as np
from src.radvlm.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

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
        print("Copying data to new directory")
        if not os.path.exists(old_data_dir):
            print("Old dataset directory does not exist.")
            raise FileNotFoundError("Old dataset directory does not exist.")
        
        for root, dirs, files in os.walk(old_data_dir):
            for file in files:
                old_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(old_file_path, old_data_dir)
                new_file_path = os.path.join(new_data_dir, relative_path)
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                with open(old_file_path, 'rb') as src_file:
                    with open(new_file_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
        print("Data copying completed.")
        return
    except Exception as e:
        print(f"Error copying data: {e}")
        return


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
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.dcm'):
                    input_image_path = os.path.join(root, file)
                    # Check if the JPEG file already exists
                    jpeg_path = input_image_path.replace('.dcm', '.jpg')
                    if os.path.exists(jpeg_path):
                        continue

                    # Read the DICOM file
                    dcm_image = pydicom.dcmread(input_image_path)
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
                    image = Image.fromarray(pixel_array, mode='L')
                    image.save(jpeg_path, 'JPEG')

                    # Remove the original DICOM file to save space
                    os.remove(input_image_path)
                    print(f"Converted {input_image_path} to {jpeg_path}")
        print("Transformed files to .jpg")
        return 
    except Exception as e:
        print(f"Error transforming images: {e}")
        return


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
                    print(f"Deleted {file_path}")
        print("Deleted all index files")
        return
    except Exception as e:
        print(f"Error deleting index files: {e}")
        return

def resize_images(data_dir, max_resolution=2048):
    """
    Resize images to a maximum resolution while maintaining aspect ratio.
    
    Args:
        max_resolution (int): Maximum resolution for the longest side of the image.
        
    Returns:
        None
    """

    try:
        print(f"Resizing images to max resolution {max_resolution}")
        # here = os.path.dirname(os.path.abspath(__file__))
        # data_dir = os.path.join(here, "..", "..", "..", "data", "raw", "MIMIC-CXR-JPG")
        # Load images and texts from the dataset
        if not os.path.exists(data_dir):
            print("Dataset directory does not exist.")
            raise FileNotFoundError("Dataset directory does not exist.")
        
        for root, dirname, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg'): # Radiology reports are stored as .txt files
                    image_path = os.path.join(root, file)
                    # Print the resolution of the image and resize it if necessary
                    with Image.open(image_path) as img:
                        # print(f"Image {file}, resolution: {img.size}")
                        w, h = img.size
                        if w > max_resolution or h > max_resolution:
                            scale = max_resolution / max(w, h)
                            new_size = (int(w * scale), int(h * scale))
                            new_img = img.resize(new_size, Image.BICUBIC)
                            # new_path = image_path.replace("MIMIC-CXR", f"MIMIC-CXR/processed/{max_resolution}")
                            # if new_path == image_path:
                            #     raise ValueError("New path is the same as the original path.")
                            # os.makedirs(os.path.dirname(new_path), exist_ok=True)
                            new_img.save(image_path)  # Save the resized image
                            print(f"Saved resized image to {image_path}")
                            # print(f"Resized image {file} to {new_size}, saved as {new_path}")
                        else:
                            # new_path = image_path.replace("MIMIC-CXR", f"MIMIC-CXR/processed/{max_resolution}")
                            # if new_path == image_path:
                            #     raise ValueError("New path is the same as the original path.")
                            # os.makedirs(os.path.dirname(new_path), exist_ok=True)
                            img.save(image_path)
                            print(f"Image is within the max resolution, copied to {image_path}")

        print("Image resizing completed.")
        return
    except Exception as e:
        print(f"Error resizing images: {e}")
        return

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
            resize_images(new_data_dir, max_resolution=2048)
        else:
            print(f"Old data directory {old_data_dir} does not exist")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
