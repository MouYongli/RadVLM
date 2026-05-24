import os
import json
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM')
from src.radvlm.utils.config import DATA_PROCESSED_DIR, DPO_DATA_PROCESSED_DIR

here = os.path.dirname(os.path.abspath(__file__))

def load_dataset(subpaths: list = []) -> list:
    """
    Load dataset with image and text pairs.

    Parameters:
        subpaths (list): List of subdirectories to load from the processed dataset directory.

    Returns:
        dataset (list): List of dictionaries with image and text pairs.
    """
    dataset = []

    try:
        # Load images and texts from the dataset
        data_dir = DATA_PROCESSED_DIR

        if subpaths:
            print(f"Loading dataset from: {data_dir} in {subpaths}", flush=True)
            subdirs = [os.path.join(data_dir, subpath) for subpath in subpaths]
        else:
            print(f"Loading dataset from: {data_dir}", flush=True)
            subdirs = [data_dir]
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        
        for subdir in subdirs:
            if not os.path.exists(subdir):
                print(f"Subdirectory does not exist: {subdir}. Skipping.", flush=True)
                continue
            for root, _, files in os.walk(subdir):
                for file in files:
                    if file.endswith('.txt'):
                        with open(os.path.join(root, file), 'r') as f:
                            # Read radiology report text
                            text_content = f.read().strip()

                        # Check if the corresponding image directory exists
                        if not os.path.exists(os.path.join(root, file.replace('.txt', ''))):
                            print(f"Image directory for {file} does not exist. Full path: {root}", flush=True)
                            continue # If no images exist for this report, skip this datapoint
                        else:
                            image_path = os.path.abspath(os.path.join(root, file.replace('.txt', ''))).replace("raw", "processed/2048")
                            if not os.path.exists(image_path):
                                raise FileNotFoundError(f"Image directory not found: {image_path}")
                            # Add the text and resized images to the dataset
                            dataset.append({
                                "file": os.path.join(root, file),
                                "content": text_content,
                                "images": [os.path.join(root, file.replace('.txt', ''), i) for i in os.listdir(image_path) if i.endswith('.jpg')]  
                            })

        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}", flush=True)
        return []
    
def load_preference_dataset(dataset_path) -> list:
    """
    Load preference dataset with image and text pairs.

    Returns:
        dataset (list): List of dictionaries with image and text pairs.
    """
    # Assume the dataset is stored in a json file with preference pairs
    
    here = os.path.dirname(os.path.abspath(__file__))
    # dataset_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/medgemma_radgraph_preferences.json"

    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        print(f"Error loading preference dataset: {e}", flush=True)
        return []