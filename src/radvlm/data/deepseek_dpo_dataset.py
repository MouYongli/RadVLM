import torch
from PIL import Image
import pandas as pd
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

from src.radvlm.utils.config import DATA_PROCESSED_DIR


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
            valid_images = [img for img in item["image_paths"] if os.path.exists(os.path.join(DATA_PROCESSED_DIR, img))]
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
            pil_images = [Image.open(os.path.join(DATA_PROCESSED_DIR, img_path)).convert('RGB') for img_path in image_paths]
            
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
    """
    # Filter out None values from failed samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Load images for each item (they come as paths from HF Dataset)
    for item in batch:
        if "images" in item and isinstance(item["images"][0], str):
            item["images"] = [Image.open(os.path.join(DATA_PROCESSED_DIR, img_path)).convert('RGB') 
                            for img_path in item["image_paths"]]
    
    # Process each item individually and collect results
    chosen_inputs_list = []
    rejected_inputs_list = []
    
    for item in batch:
        # Create conversation for chosen response
        chosen_conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt"],
                "images": item["images"],
            },
            {
                "role": "<|Assistant|>",
                "content": item["chosen"]
            }
        ]
        
        # Create conversation for rejected response
        rejected_conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt"],
                "images": item["images"],
            },
            {
                "role": "<|Assistant|>",
                "content": item["rejected"]
            }
        ]
        
        # Process individually (not as batch)
        chosen_input = processor(
            conversations=chosen_conversation,  # Single conversation, not list of conversations
            images=item["images"],
            force_batchify=True,
            system_prompt="",
            inference_mode=False,  # Keep EOS token for complete response
        )
        
        rejected_input = processor(
            conversations=rejected_conversation,  # Single conversation, not list of conversations
            images=item["images"],
            force_batchify=True,
            system_prompt="",
            inference_mode=False,  # Keep EOS token for complete response
        )
        
        chosen_inputs_list.append(chosen_input)
        rejected_inputs_list.append(rejected_input)
    
    # Manually batch the results
    def stack_inputs(inputs_list):
        batched = {}
        for key in inputs_list[0].keys():
            if isinstance(inputs_list[0][key], torch.Tensor):
                # Squeeze batch dimension from force_batchify, then stack
                tensors = [inp[key].squeeze(0) if inp[key].dim() > 1 and inp[key].shape[0] == 1 else inp[key] for inp in inputs_list]
                batched[key] = torch.stack(tensors)
            else:
                batched[key] = [inp[key] for inp in inputs_list]
        return batched
    
    chosen_inputs = stack_inputs(chosen_inputs_list)
    rejected_inputs = stack_inputs(rejected_inputs_list)
    
    # Convert to bfloat16 for pixel values
    if "pixel_values" in chosen_inputs and chosen_inputs["pixel_values"] is not None:
        chosen_inputs["pixel_values"] = chosen_inputs["pixel_values"].to(dtype=torch.bfloat16)
    if "pixel_values" in rejected_inputs and rejected_inputs["pixel_values"] is not None:
        rejected_inputs["pixel_values"] = rejected_inputs["pixel_values"].to(dtype=torch.bfloat16)
    
    # Create prompt_input_ids by processing prompt-only conversations (without assistant response)
    # Note: User-only conversations don't have EOS token, so we use inference_mode=False
    # and process as-is (no EOS to remove)
    prompt_inputs_list = []
    for item in batch:
        prompt_conversation = [
            {
                "role": "<|User|>",
                "content": item["prompt"],
                "images": item["images"],
            },
            {"role": "<|Assistant|>", "content": ""}  # Empty assistant response
        ]
        
        prompt_input = processor(
            conversations=prompt_conversation,
            images=item["images"],
            force_batchify=True,
            system_prompt="",
            inference_mode=False,  # User-only conversation doesn't have EOS token
        )
        prompt_inputs_list.append(prompt_input)
    
    # Batch prompt inputs
    prompt_inputs = stack_inputs(prompt_inputs_list)
    if "pixel_values" in prompt_inputs and prompt_inputs["pixel_values"] is not None:
        prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(dtype=torch.bfloat16)

    # Return in format expected by DPOTrainer
    return {
        "prompt_input_ids": prompt_inputs["input_ids"],
        "prompt_attention_mask": prompt_inputs["attention_mask"],
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "pixel_values_chosen": chosen_inputs.get("pixel_values"),
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
        "pixel_values_rejected": rejected_inputs.get("pixel_values"),
    }