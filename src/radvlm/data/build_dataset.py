import os

here = os.path.dirname(os.path.abspath(__file__))

def load_dataset() -> list:
    """
    Load dataset with image and text pairs.

    Returns:
        dataset (list): List of dictionaries with image and text pairs.
    """
    dataset = []

    try:
        # Load images and texts from the dataset
        if not os.path.exists(os.path.join(here, "..", "..", "..", "data", "raw", "MIMIC-CXR-JPG", "p10")):
            print("Dataset directory does not exist.")
            return []
        
        data_dir = os.path.join(here, "..", "..", "..", "data", "raw", "MIMIC-CXR-JPG", "p10")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        text_content = f.read().strip()
                        if not text_content:
                            continue

                    # Check if the corresponding image directory exists
                    if not os.path.exists(os.path.join(root, file.replace('.txt', ''))):
                        print(f"Image directory for {file} does not exist.")
                        continue
                    else:
                        image_path = os.path.join(root, file.replace('.txt', ''))
                        dataset.append({
                            "content": text_content,
                            "images": [os.path.join(root, file.replace('.txt', ''), i) for i in os.listdir(image_path) if i.endswith('.jpg')]  
                        })

        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
# if __name__ == "__main__":
#     dataset = load_dataset()
#     if dataset:
#         print(f"Loaded {len(dataset)} items from the dataset.")
#     else:
#         print("No items loaded from the dataset.")