from PIL import Image
import os

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
    resize_images(max_resolution=2048)
    resize_images(max_resolution=1024)
    print("Image resizing completed.")