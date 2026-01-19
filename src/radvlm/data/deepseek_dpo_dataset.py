import torch
from PIL import Image
import pandas as pd
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))


class RadVLMDPODataset(torch.utils.data.Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, data, processor, tokenizer, max_seq_length=3072, split=None):
        """
        Args:
            data: List of dicts with 'image_paths', 'report_1', 'report_2', 'radiologist_preference', 'ground_truth'
            processor: DeepSeek VL2 processor
            tokenizer: Tokenizer
            max_seq_length: Maximum sequence length
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        
        # Filter data by split if needed
        if split:
            self.data = self._filter_data_by_split(data, split)
        else:
            self.data = data
        # self.data = data
            
        # Remove invalid samples
        self.data = self._validate_data(self.data)
        
        print(f"Initialized RadVLMDPODataset with {len(self.data)} samples", flush=True)
    
    def _filter_data_by_split(self, data, split):
        """Filter data by split using local split file"""
        split_file = os.path.join(os.path.dirname(__file__), "preference-data-split.csv")
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found at {split_file}. Using all data.", flush=True)
            return data
        
        
        df_split = pd.read_csv(split_file)
        split_dict = dict(zip(df_split['dicom_id'], df_split['split']))
        
        filtered_data = []
        for item in data:
            image_paths = item.get("image_paths", [])
            valid_images = []
            
            for img_path in image_paths:
                # Extract DICOM ID from path
                img_filename = os.path.basename(img_path).replace('.jpg', '')
                item_split = split_dict.get(img_filename)
                
                if item_split == split:
                    valid_images.append(img_path)
            
            # Keep item if at least one image matches the split
            if valid_images:
                item_copy = item.copy()
                item_copy["image_paths"] = valid_images
                filtered_data.append(item_copy)
        
        print(f"Filtered to {len(filtered_data)} items for split '{split}'", flush=True)
        return filtered_data
    
    def _validate_data(self, data):
        """Remove samples with missing or invalid data"""
        valid_data = []
        
        for idx, item in enumerate(data):
            # Check required fields
            if not item.get("image_paths"):
                print(f"Skipping item {idx}: No image paths", flush=True)
                continue
            
            if not item.get("report_1") or not item.get("report_2"):
                print(f"Skipping item {idx}: Missing report_1 or report_2", flush=True)
                continue
            
            if not item.get("radiologist_preference"):
                print(f"Skipping item {idx}: Missing radiologist_preference", flush=True)
                continue
            
            # Validate image files exist
            valid_images = [img for img in item["image_paths"] if os.path.exists(img)]
            if not valid_images:
                print(f"Skipping item {idx}: No valid image files found", flush=True)
                continue
            
            item["image_paths"] = valid_images
            valid_data.append(item)
        
        print(f"Validated {len(valid_data)} out of {len(data)} samples", flush=True)
        return valid_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with chosen and rejected responses
        """
        try:
            item = self.data[idx]
            image_paths = item["image_paths"]
            
            # Determine which is chosen vs rejected based on radiologist preference
            if item["radiologist_preference"] == "report_1":
                chosen_report = item["report_1"]
                rejected_report = item["report_2"]
            else:
                chosen_report = item["report_2"]
                rejected_report = item["report_1"]
            
            # Load PIL images (similar to deepseek's load_pil_images)
            pil_images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
            
            # Build prompt
            image_tokens = "<image>" * len(image_paths)
            prompt = f"{image_tokens}\nGenerate a radiology report for these X-rays."
            
            # Return dictionary with all needed info
            return {
                "prompt": prompt,
                "chosen": chosen_report,
                "rejected": rejected_report,
                "images": pil_images,
                "image_paths": image_paths,
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}", flush=True)
            return None


def dpo_collate_fn(batch, processor, tokenizer, max_seq_length=3072):
    """
    Custom collator for DPO training with DeepSeek-VL2
    
    Args:
        batch: List of samples from dataset
        processor: DeepSeek VL2 processor
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
    
    Returns:
        Dictionary with chosen and rejected inputs properly formatted for DPO
    """
    # Filter out None values from failed samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Process chosen responses
    chosen_conversations = []
    chosen_images_list = []
    
    for item in batch:
        conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt"],
                "images": item["image_paths"],
            },
            {
                "role": "<|Assistant|>",
                "content": item["chosen"]
            }
        ]
        chosen_conversations.append(conversation)
        chosen_images_list.append(item["images"])
    
    # Process rejected responses
    rejected_conversations = []
    rejected_images_list = []
    
    for item in batch:
        conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt"],
                "images": item["image_paths"],
            },
            {
                "role": "<|Assistant|>",
                "content": item["rejected"]
            }
        ]
        rejected_conversations.append(conversation)
        rejected_images_list.append(item["images"])
    
    # Tokenize chosen responses
    chosen_inputs = processor(
        conversations=chosen_conversations,
        images=chosen_images_list,
        force_batchify=True,
        system_prompt="",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )
    
    # Tokenize rejected responses
    rejected_inputs = processor(
        conversations=rejected_conversations,
        images=rejected_images_list,
        force_batchify=True,
        system_prompt="",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )
    
    # Convert to bfloat16 for pixel values
    if hasattr(chosen_inputs, 'pixel_values') and chosen_inputs.pixel_values is not None:
        chosen_inputs.pixel_values = chosen_inputs.pixel_values.to(dtype=torch.bfloat16)
    if hasattr(rejected_inputs, 'pixel_values') and rejected_inputs.pixel_values is not None:
        rejected_inputs.pixel_values = rejected_inputs.pixel_values.to(dtype=torch.bfloat16)
    
    # Return in format expected by DPOTrainer
    return {
        "input_ids_chosen": chosen_inputs["input_ids"],
        "attention_mask_chosen": chosen_inputs["attention_mask"],
        "pixel_values_chosen": chosen_inputs.get("pixel_values"),
        "input_ids_rejected": rejected_inputs["input_ids"],
        "attention_mask_rejected": rejected_inputs["attention_mask"],
        "pixel_values_rejected": rejected_inputs.get("pixel_values"),
    }