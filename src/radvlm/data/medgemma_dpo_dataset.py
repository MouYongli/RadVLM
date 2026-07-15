import os
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn.functional as F

from src.radvlm.utils.config import DATA_PROCESSED_DIR, DPO_DATA_PROCESSED_DIR

class RadVLMDPODatasetMedGemma:
    def __init__(self, data, processor, tokenizer, max_seq_length=3072, split=None):
        """
        Args:
            data: List of dicts with 'image_paths', 'report_1', 'report_2', 'radiologist_preference', 'ground_truth'
            processor: MedGemma processor for image and text processing.
            tokenizer: Tokenizer for text tokenization.
            max_seq_length: Maximum sequence length
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.raw_data = data

        self.raw_data = self._add_image_paths(self.raw_data)
        
        # Remove invalid samples from the raw data
        self.raw_data = self._validate_data(self.raw_data)
        
        print(f"Initialized RadVLMDPODatasetMedGemma with {len(self.raw_data)} samples", flush=True)
    
    def _add_image_paths(self, data):
        """Add image paths to each sample based on the file field"""
        data_corrected = []
        for item in data:
            image_paths = item.get("image_paths", [])
            valid_images = []
            
            for img_path in image_paths:                
                valid_images.append(os.path.join(DATA_PROCESSED_DIR, img_path))
            
            # Keep item if at least one image matches the split
            if valid_images:
                item_copy = item.copy()
                item_copy["image_paths"] = valid_images
                data_corrected.append(item_copy)
        return data_corrected

    def _filter_data_by_split(self, data, split):
        """Filter data by split using local split file"""
        split_file = os.path.join(DPO_DATA_PROCESSED_DIR, "split-model23-10pctdatasetreports.csv")
        
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
    
    def get_split_data(self, split: str) -> list[dict]:
        filtered_data = self._filter_data_by_split(self.raw_data, split)

        result = []
        for item in filtered_data:
            if item["radiologist_preference"] == "report_1":
                chosen_report   = item["report_1"]
                rejected_report = item["report_2"]
            else:
                chosen_report   = item["report_2"]
                rejected_report = item["report_1"]

            image_paths = item["image_paths"]

            # Build the prompt-only messages (user turn with image placeholders).
            # We apply the chat template with add_generation_prompt=True so the
            # model knows to continue from the assistant turn.
            pil_images = [Image.open(p).convert("RGB") for p in image_paths]
            image_content = [{"type": "image", "image": img} for img in pil_images]
            text_content  = [{"type": "text",  "text": "Generate a radiology report for these X-rays."}]

            prompt_messages = [
                {"role": "user", "content": image_content + text_content},
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            ).strip()

            pil_images  = [Image.open(p).convert("RGB") for p in image_paths]

            result.append({
                "prompt":   prompt_text,
                # Completions are plain text — DPOTrainer tokenises them separately
                # from the prompt and does NOT pass images through them again.
                "chosen":   chosen_report,
                "rejected": rejected_report,
                # Store paths as strings; MedGemmaDPOTrainer.process_row loads them.
                "images":   pil_images,
            })

        return result

def dpo_collate_fn(batch, processor, max_seq_length=3072):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    print(f"Batch: ", batch)
    chosen_texts   = [b["chosen"]   for b in batch]
    rejected_texts = [b["rejected"] for b in batch]

    # Reload PIL images from paths (fast — already on local disk)
    chosen_images_list   = [[Image.open(p).convert("RGB") for p in b["images"]] for b in batch]
    rejected_images_list = [[Image.open(p).convert("RGB") for p in b["images"]] for b in batch]

    def encode(texts, images_list):
        enc = processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        if enc.get("pixel_values") is not None:
            enc["pixel_values"] = enc["pixel_values"].to(dtype=torch.bfloat16)
        return enc

    chosen_enc   = encode(chosen_texts,   chosen_images_list)
    rejected_enc = encode(rejected_texts, rejected_images_list)

    return {
        "input_ids_chosen":        chosen_enc["input_ids"],
        "attention_mask_chosen":   chosen_enc["attention_mask"],
        "pixel_values_chosen":     chosen_enc.get("pixel_values"),
        "input_ids_rejected":      rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
        "pixel_values_rejected":   rejected_enc.get("pixel_values"),
    }