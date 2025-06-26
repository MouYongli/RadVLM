import os
import shutil

def filter_reports_with_conclusion():
    """
    Filters out reports that contain a conclusion and saves them in the folder RadVLM/data/processed.

    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "..", "..", "data", "raw", "MIMIC-CXR-JPG", "p10")
        # Load images and texts from the dataset
        if not os.path.exists(data_dir):
            print("Dataset directory does not exist.")
            raise FileNotFoundError("Dataset directory does not exist.")
        
        for root, dirname, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt'): # Radiology reports are stored as .txt files
                    with open(os.path.join(root, file), 'r') as f:
                        text_content = f.read().strip()
                        if not text_content:
                            print(f"Skipping empty report: {file}")
                            continue
                    if "conclusion" in text_content.lower():
                        
                        # Save the report in the processed folder
                        processed_dir = os.path.abspath(root).replace("raw", "processed")
                        if not os.path.exists(processed_dir):
                            os.makedirs(processed_dir) # Create the processed directory if it does not exist
                        with open(os.path.join(processed_dir, file), 'w') as f:
                            f.write(text_content)
                        print(f"Saved report: {file} to {processed_dir}")
                        
                        # Safe corresponding images
                        if not os.path.exists(os.path.join(root, file.replace('.txt', ''))):
                            print(f"Image directory for {file} does not exist.")
                            continue
                        else:
                            image_path = os.path.join(root, file.replace('.txt', ''))
                            image_files = [os.path.join(image_path, i) for i in os.listdir(image_path) if i.endswith('.jpg')]
                            print(f"Found {len(image_files)} images for report {file}.")
                            if image_files:
                                for image_file in image_files:
                                    new_image_path = image_path.replace("raw", "processed")
                                    if not os.path.exists(new_image_path):
                                        os.makedirs(new_image_path) # Create the new directory if it does not exist
                                    # Copy the image to the new directory
                                    shutil.copy2(image_file, new_image_path)
                            else:
                                print(f"No images found for report {file}.")
                       
                        print(f"Saved report with conclusion: {file}")
                    
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    if filter_reports_with_conclusion():
        print("Filtering completed successfully.")
    else:
        print("Filtering failed.")