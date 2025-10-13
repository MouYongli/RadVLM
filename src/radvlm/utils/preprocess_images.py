from PIL import Image
import os
import pydicom
import numpy as np

def transform_dcm_to_jpg(data_dir: str) -> str:
    """
    Transform all DICOM images in the given directory to JPEG format.

    Args:
        data_dir (str): Path to the directory containing DICOM images.

    Returns:
        None
    """
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
    return 


def delete_index_files(data_dir):
    """
    Delete all index.html files in the given directory and its subdirectories.
    
    Args:
        data_dir (str): Path to the directory to search for .idx files.

    Returns:
        None
    """
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == 'index.html':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")
    return


def resize_images(max_resolution=2048):
    """
    Resize images to a maximum resolution while maintaining aspect ratio.
    
    Args:
        image_paths (list): List of paths to the images.
        max_resolution (int): Maximum resolution for the longest side of the image.
        
    Returns:
        None
    """
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "..", "..", "data", "raw", "MIMIC-CXR-JPG")
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
                        new_path = os.path.abspath(image_path).replace("raw", f"processed/{max_resolution}")
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        new_img.save(new_path)  # Save the resized image
                        # print(f"Resized image {file} to {new_size}, saved as {new_path}")  
                    else:
                        new_path = os.path.abspath(image_path).replace("raw", f"processed/{max_resolution}")
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        img.save(new_path)
    return

if __name__ == "__main__":
    # resize_images(max_resolution=2048)
    # resize_images(max_resolution=1024)
    # print("Image resizing completed.")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "raw", "MIMIC-CXR", "p15"))
    print(data_dir)
    delete_index_files(data_dir)
    transform_dcm_to_jpg(data_dir)