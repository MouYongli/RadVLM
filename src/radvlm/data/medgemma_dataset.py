from torch.utils.data import Dataset
import os
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn.functional as F

from src.radvlm.utils.config import DATA_PROCESSED_DIR


class RadVLMDatasetMedGemma(Dataset):
    def __init__(self, data, processor, tokenizer, max_seq_length=3072, split=None, create_stats=False, mode='train', sample_fraction=1.0):
        """
        RadVLM Dataset for MedGemma model.
        Args:
            data (list): List of dictionaries with image and text pairs.
            processor: MedGemma processor for image and text processing.
            tokenizer: Tokenizer for text tokenization.
            max_seq_length (int): Maximum sequence length for tokenization.
            split (str, optional): Dataset split to use ('train', 'validate', 'test'). Defaults to None.
            create_stats (bool, optional): Whether to create dataset statistics. Defaults to False.
            mode (str): 'train' for training (with labels), 'eval' for evaluation (without labels). Defaults to 'train'.
            sample_fraction (float, optional): Fraction of data to sample for POC. Defaults to 1.0.
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.data = self._preprocess_reports(data, create_stats)
        self.max_seq_length = max_seq_length
        self.split = split
        self.mode = mode
        if split is not None:
            self.data = self._filter_data_by_split(self.data, split, sample_fraction=sample_fraction)
        
        # Set ignore index for label masking
        self.ignore_index = -100
        
        # Cache for prompt lengths (keyed by number of images)
        self._prompt_length_cache = {}
        
        print(f"Initialized RadVLMDatasetMedGemma with {len(self.data)} samples in {mode} mode.", flush=True)

    def _filter_data_by_split(self, data, split, keep_file_names=None, sample_fraction=1.0):
        # Read "mimic-cxr-split.csv" to filter data by split
        split_file = os.path.join(DATA_PROCESSED_DIR, "mimic-cxr-2.0.0-split.csv")
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}. Returning unfiltered data.", flush=True)
            return data
        
        df_split = pd.read_csv(split_file)
        split_dict = dict(zip(df_split['dicom_id'], df_split['split']))

        filtered_data = []
        for item in data:
            images = item["images"]
            if not images:
                continue
            
            filtered_data_item = {"file": item["file"], "content": item["content"], "images": []}
            # For each image, check if it belongs to the desired split. If yes, add it to the filtered images.
            for image in images:
                if not os.path.exists(image):
                    print(f"Image file does not exist: {image}", flush=True)
                    continue
                image_filename = os.path.basename(image).replace('.jpg', '')
                item_split = split_dict.get(image_filename)
                if item_split is not None:
                    if item_split == split:
                        filtered_data_item["images"].append(image)
                else:
                    print(f"No split information found for image: {image_filename}", flush=True)
            if filtered_data_item["images"]:
                filtered_data.append(filtered_data_item)
                if len(images) != len(filtered_data_item["images"]):
                    print(f"Item {item.get('file', 'unknown')} - kept {len(filtered_data_item['images'])} out of {len(images)} images for split '{split}'", flush=True)
                        
        print(f"Filtered data to {len(filtered_data)} items for split '{split}'", flush=True)

        # Apply sampling for POC (keep only sample_fraction of data)
        if sample_fraction < 1.0:
            random.seed(42)  # For reproducibility
            sample_size = int(len(filtered_data) * sample_fraction)
            filtered_data = random.sample(filtered_data, sample_size)
            print(f"Sampled {sample_size} items ({sample_fraction*100}%) for POC training", flush=True)

        return filtered_data

    def _preprocess_reports(self, data, create_stats=False):
        if create_stats:
            print("Creating dataset statistics...", flush=True)
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(DATA_PROCESSED_DIR, "../logs")
            os.makedirs(logs_dir, exist_ok=True)

            for split, data_split in [("full", data), ("train", self._filter_data_by_split(data, "train", keep_file_names=True)), ("validate", self._filter_data_by_split(data, "validate", keep_file_names=True)), ("test", self._filter_data_by_split(data, "test", keep_file_names=True))]:
                print(f"Creating statistics for split: {split} with {len(data_split)} items", flush=True)
                with open(os.path.join(logs_dir, f"{split}_dataset_size.txt"), 'w') as f:
                    f.write(f"Dataset size for {split} split: {len(data_split)} items\n")

                # Log report length statistics
                if data_split:
                    report_lengths = [(item["file"], len(self.tokenizer.encode(item["content"].strip(), add_special_tokens=False))) for item in data_split]
                    if report_lengths:
                        lengths = [length for _, length in report_lengths]
                        
                        with open(os.path.join(logs_dir, f"report_length_stats_{split}.txt"), 'w') as f:
                            f.write(f"Report length statistics for {split} split:\n")
                            f.write(f"Min: {min(lengths)}\n")
                            f.write(f"Max: {max(lengths)}\n")
                            f.write(f"Mean: {sum(lengths) / len(lengths):.2f}\n")
                            f.write(f"Median: {sorted(lengths)[len(lengths) // 2]}\n")
                        
                        with open(os.path.join(logs_dir, f"report_lengths_{split}.csv"), 'w') as f:
                            f.write("file,length\n")
                            for file, length in report_lengths:
                                f.write(f"{file},{length}\n")
                    else:
                        print(f"No valid reports found for length statistics in split '{split}'.", flush=True)

                    # Remove entries with empty reports
                    filtered_data = [item for item in data_split if (item["content"].strip() != "") and (item["content"].strip().lower() != "final report") and (item["content"].strip().lower() != "final report:")]
                    print(f"Removed empty reports. {len(filtered_data)} items remain. {len(data_split) - len(filtered_data)} items were removed.", flush=True)

                    with open(os.path.join(logs_dir, f"empty_report_removal_{split}.txt"), 'w') as f:
                        f.write(f"Empty report removal for {split} split:\n")
                        f.write(f"Items removed: {len(data_split) - len(filtered_data)}\n")
                        f.write(f"Items remaining: {len(filtered_data)}\n")

                    # Save short report examples to a file for inspection
                    short_report_file = os.path.join(logs_dir, f"short_reports_{split}.txt")
                    with open(short_report_file, 'w') as f:
                        short_reports = sorted(report_lengths, key=lambda x: x[1])[:10]
                        for file, length in short_reports:
                            f.write(f"{file}: {length} tokens\n")

                    # Save long report examples to a file for inspection
                    long_report_file = os.path.join(logs_dir, f"long_reports_{split}.txt")
                    with open(long_report_file, 'w') as f:
                        long_reports = sorted(report_lengths, key=lambda x: x[1], reverse=True)[:10]
                        for file, length in long_reports:
                            f.write(f"{file}: {length} tokens\n")
                else:
                    print(f"No data available for split '{split}' to create statistics.", flush=True)

        else:
            filtered_data = [item for item in data if (item["content"].strip() != "") and (item["content"].strip().lower() != "final report") and (item["content"].strip().lower() != "final report:")]
            print(f"Removed empty reports. {len(filtered_data)} items remain. {len(data) - len(filtered_data)} items were removed.", flush=True)

        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict: Contains input_ids, attention_mask, labels, study_id, and optionally report
        """
        try:
            item = self.data[idx]
            images = item["images"]
            report = item["content"]
            
            # Extract and clean study_id
            study_id = self._extract_study_id(item, idx)
            
            # Validate and filter images
            valid_images = self._validate_images(images, idx)
            if not valid_images:
                raise ValueError(f"No valid images found for item {idx} (study_id: {study_id})")
            
            # Load PIL images
            pil_images = [Image.open(img_path).convert('RGB') for img_path in valid_images]
            
            # Build messages in MedGemma format
            messages = self._build_messages(pil_images, report)
            

            result = {
                "images": pil_images,
                "messages": messages,
            }
            
            # Add ground truth report for evaluation metrics
            if self.mode == 'eval':
                result["study_id"] = study_id
                # result["images"] = valid_images
                result["report"] = report
            
            return result
            
        except Exception as e:
            print(f"Error processing item {idx} (study_id: {study_id if 'study_id' in locals() else 'unknown'}): {str(e)}", flush=True)
            if self.mode == 'train':
                # In training, skip problematic samples by returning None
                # DataLoader will handle this with collate_fn
                return None
            else:
                # In eval, we need all samples for proper metrics
                raise RuntimeError(f"Failed to process evaluation item {idx}: {str(e)}")

    def _extract_study_id(self, item, idx):
        """Extract study ID from item or generate from index."""
        study_id = item.get("file", f"study_{idx}")
        if isinstance(study_id, str) and "/" in study_id:
            study_id = os.path.basename(study_id).replace('.txt', '')
        return study_id

    def _validate_images(self, images, idx):
        """Validate that image files exist and are readable."""
        if not images:
            print(f"Warning: No images provided for item {idx}", flush=True)
            return []
        
        valid_images = []
        for img_path in images:
            if isinstance(img_path, str) and os.path.exists(img_path) and os.path.isfile(img_path):
                valid_images.append(img_path)
            else:
                print(f"Warning: Invalid or missing image at {img_path} for item {idx}", flush=True)
        
        if not valid_images:
            print(f"Warning: No valid images found for item {idx}", flush=True)
        
        return valid_images

    def _build_messages(self, pil_images, report):
        """
        Build messages structure in MedGemma chat template format.
        
        Following the format from medgemma.py:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Generate a radiology report for these X-rays."}
                ]
            }
        ]
        """
        # Build content list with images and prompt
        content = []
        for pil_image in pil_images:
            content.append({"type": "image", "image": pil_image})
        content.append({"type": "text", "text": "Generate a radiology report for these X-rays."})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # For training mode, add assistant response
        if self.mode == 'train':
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": report}]
            })
        
        return messages



from typing import Any


def create_collate_fn_medgemma(processor):
    """
    Factory function to create a collate function with processor bound.
    
    Args:
        processor: MedGemma processor for image and text processing
        
    Returns:
        collate function that can be used with DataLoader
    """
    def collate_fn_medgemma(examples: list[dict[str, Any]]):
        texts = []
        images = []
        for example in examples:
            images.append([i.convert("RGB") for i in example["images"]])
            texts.append(processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            ).strip())

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, with the padding and image tokens masked in
        # the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens that are not used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels

        batch["images"] = images

        # Add evaluation-specific fields if present
        if 'study_id' in examples[0]:
            batch["study_id"] = [item['study_id'] for item in examples]
        if 'report' in examples[0]:
            batch["report"] = [item['report'] for item in examples]
        return batch
    
    return collate_fn_medgemma