import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

from deepseek_vl2.utils.io import load_pil_images

from src.radvlm.utils.config import DATA_PROCESSED_DIR

class RadVLMDatasetDeepseek(Dataset):
    def __init__(self, data, processor, tokenizer, max_seq_length=3072, split=None, create_stats=False, mode='train'):
        """
        RadVLM Dataset for Deepseek VL2 model.
        Args:
            data (list): List of dictionaries with image and text pairs.
            processor: Deepseek VL2 processor for image and text processing.
            tokenizer: Tokenizer for text tokenization.
            max_seq_length (int): Maximum sequence length for tokenization.
            split (str, optional): Dataset split to use ('train', 'validate', 'test'). Defaults to None.
            create_stats (bool, optional): Whether to create dataset statistics. Defaults to False.
            mode (str): 'train' for training (with labels), 'eval' for evaluation (without labels). Defaults to 'train'.
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.data = self._preprocess_reports(data, create_stats)
        self.max_seq_length = max_seq_length
        self.split = split
        self.mode = mode
        if split is not None:
            self.data = self._filter_data_by_split(self.data, split)
        
        # Set ignore index for label masking
        self.ignore_index = -100
        
        # Cache for prompt lengths (keyed by number of images)
        self._prompt_length_cache = {}
        
        print(f"Initialized RadVLMDatasetDeepseek with {len(self.data)} samples in {mode} mode.", flush=True)

    def _filter_data_by_split(self, data, split, keep_file_names=None):
        # Read "mimic-cxr-split.csv" to filter data by split
        split_file = os.path.join(DATA_PROCESSED_DIR, "mimic-cxr-2.0.0-split.csv")
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}. Returning unfiltered data.", flush=True)
            return data
        import pandas as pd
        df_split = pd.read_csv(split_file)

        split_dict = dict(zip(df_split['dicom_id'], df_split['split']))

        filtered_data = []
        for item in data:
            images = item["images"]
            if not images:
                continue
            
            # if keep_file_names is None:
            #     filtered_data_item = {"content": item["content"], "images": []}
            # else:
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
                    report_lengths = [(item["file"],len(self.tokenizer.encode(item["content"].strip(),add_special_tokens=False))) for item in data_split]
                    if report_lengths:
                        lengths = [length for _, length in report_lengths]
                        
                        with open(os.path.join(logs_dir, f"report_length_stats_{split}.txt"), 'w') as f:
                            f.write(f"Report length statistics:\n")
                            f.write(f"Min: {min(lengths)}\n")
                            f.write(f"Max: {max(lengths)}\n")
                            f.write(f"Avg: {sum(lengths)/len(lengths):.2f}\n")
                            f.write(f"Total reports: {len(lengths)}\n")
                        with open(os.path.join(logs_dir, f"report_lengths_{split}.csv"), 'w') as f:
                            f.write("file,report_length\n")
                            for file, length in report_lengths:
                                f.write(f"{file},{length}\n")
                    else:
                        print(f"No valid reports found for length statistics in split '{split}'.", flush=True)
                        with open(os.path.join(logs_dir, f"report_length_stats_{split}.txt"), 'w') as f:
                            f.write(f"No valid reports found for length statistics in split '{split}'.\n")
                        with open(os.path.join(logs_dir, f"report_lengths_{split}.csv"), 'w') as f:
                            f.write("file,report_length\n")

                    # Remove entries with empty reports
                    filtered_data = [item for item in data_split if item["content"].strip() != ""]
                    print(f"Removed empty reports. {len(filtered_data)} items remain. {len(data_split) - len(filtered_data)} items were removed.", flush=True)

                    with open(os.path.join(logs_dir, f"empty_report_removal_{split}.txt"), 'w') as f:
                        f.write(f"Total items before removal: {len(data_split)}\n")
                        f.write(f"Total items after removal: {len(filtered_data)}\n")
                        f.write(f"Total empty reports removed: {len(data_split) - len(filtered_data)}\n")

                    # Save short report examples to a file for inspection
                    short_report_file = os.path.join(logs_dir, f"short_reports_{split}.txt")
                    with open(short_report_file, 'w') as f:
                        for item in filtered_data:
                            if len(self.tokenizer.encode(item["content"].strip(), add_special_tokens=False)) < 20:
                                f.write(f"Report: {item['content'].strip()}\nImages: {item['images']}\n\n")

                    # Save long report examples to a file for inspection
                    long_report_file = os.path.join(logs_dir, f"long_reports_{split}.txt")
                    with open(long_report_file, 'w') as f:
                        for item in filtered_data:
                            if len(self.tokenizer.encode(item["content"].strip(), add_special_tokens=False)) > 800:
                                f.write(f"Report: {item['content'].strip()}\nImages: {item['images']}\n\n")
                else:
                    print(f"No data available for split '{split}' to create statistics.", flush=True)
                    with open(os.path.join(logs_dir, f"{split}_dataset_size.txt"), 'w') as f:
                        f.write(f"No data available for split '{split}'.\n")
                    with open(os.path.join(logs_dir, f"report_length_stats_{split}.txt"), 'w') as f:
                        f.write(f"No data available for split '{split}' to create statistics.\n")
                    with open(os.path.join(logs_dir, f"report_lengths_{split}.csv"), 'w') as f:
                        f.write("file,report_length\n")
                    with open(os.path.join(logs_dir, f"empty_report_removal_{split}.txt"), 'w') as f:
                        f.write(f"No data available for split '{split}' to remove empty reports.\n")
                    with open(short_report_file, 'w') as f:
                        f.write("")
                    with open(long_report_file, 'w') as f:
                        f.write("")

        else:
            filtered_data = [item for item in data if item["content"].strip() != ""]
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
            # for image_path in valid_images:
            #     with Image.open(image_path) as img:
            #         print(f"Image resolution for study {study_id}: {img.size}", flush=True)
            
            # Build conversation
            conversation = self._build_conversation(valid_images, report)
            
            # Load PIL images
            pil_images = load_pil_images(conversation)
            
            # Process inputs with processor
            inputs = self._process_inputs(conversation, pil_images)
            
            # Extract tensors
            input_ids = inputs.input_ids.squeeze(0)
            attention_mask = inputs.attention_mask.squeeze(0)
            
            # Ensure correct length (processor should handle this, but double-check)
            input_ids, attention_mask = self._ensure_length(input_ids, attention_mask)
            
            # Create labels based on mode
            labels = self._create_labels(input_ids, attention_mask, len(valid_images), pil_images, report)

            # Get the assistant token ID
            assistant_token = "<|Assistant|>"
            assistant_token_ids = self.processor.tokenizer.encode(
                assistant_token, 
                add_special_tokens=False
            )
    
            # print("Assistant token IDs: ",assistant_token_ids, flush=True)
            
            # Convert to tensor for comparison if needed
            if isinstance(input_ids, torch.Tensor):
                input_ids_list = input_ids.tolist()
            else:
                input_ids_list = input_ids
    
            # # save input_ids_list to csv for debugging
            # import csv
            # import os
            # debug_dir = "./debug_tokens"
            # os.makedirs(debug_dir, exist_ok=True)
            
            # # Save the token IDs and their decoded values
            # csv_path = os.path.join(debug_dir, f"input_ids_debug_{len(os.listdir(debug_dir))}.csv")
            # with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['Position', 'Token_ID', 'Decoded_Token', 'Label'])
            #     for idx, token_id in enumerate(input_ids_list):
            #         try:
            #             decoded = self.processor.tokenizer.decode([token_id])
            #         except:
            #             decoded = "<ERROR>"
            #         writer.writerow([idx, token_id, decoded, labels[idx]])
            
            # Build result dictionary
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            
            # Add ground truth report for evaluation metrics
            if self.mode == 'eval':
                result["study_id"] = study_id
                result["images"] = valid_images
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

    def _build_conversation(self, images, report):
        """Build conversation structure based on mode."""
        image_tokens = "<image>" * len(images)
        prompt = f"{image_tokens}\nGenerate a radiology report for these X-rays."
        
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
                "images": images,
            },
            {
                "role": "<|Assistant|>",
                # For training, include the full report; for eval, include it for proper formatting
                "content": report if self.mode == 'train' else ""
            }
        ]
        
        return conversation

    def _process_inputs(self, conversation, pil_images):
        """Process conversation and images through DeepSeek VL2 processor."""
        return self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True
        )

    def _ensure_length(self, input_ids, attention_mask):
        """Ensure tensors are exactly max_seq_length (fallback if processor doesn't)."""
        current_length = input_ids.shape[0]
        
        if current_length < self.max_seq_length:
            # Pad if too short
            pad_length = self.max_seq_length - current_length
            input_ids = F.pad(input_ids, (0, pad_length), value=self.processor.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        elif current_length > self.max_seq_length:
            # Truncate if too long
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        return input_ids, attention_mask

    def _create_labels(self, input_ids, attention_mask, num_images, pil_images, report):
        """Create labels tensor based on mode."""
        if self.mode == 'train':
            labels = input_ids.clone()
            
            # Find where assistant response actually starts
            assistant_start_pos = self._find_assistant_start(input_ids)
            
            # Mask everything before the assistant's actual response
            if assistant_start_pos is not None and assistant_start_pos < len(labels):
                labels[:assistant_start_pos] = self.ignore_index
            else:
                # Fallback to old method if we can't find the token
                prompt_length = self._get_prompt_length(num_images, pil_images)
                if prompt_length < len(labels):
                    labels[:prompt_length] = self.ignore_index
            
            # Mask padding tokens
            labels[attention_mask == 0] = self.ignore_index
            
        else:  # eval mode
            # Tokenize ground truth report for evaluation metrics
            labels = self.processor.tokenizer(
                report,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                add_special_tokens=False
            ).input_ids.squeeze(0)
        
        return labels
    
    def _find_assistant_start(self, input_ids):
        """
        Find the position where the assistant's actual response starts.
        This searches for the assistant role token and returns the position after it.
        """
        # Get the assistant token ID
        assistant_token = "<|Assistant|>"
        assistant_token_ids = self.processor.tokenizer.encode(
            assistant_token, 
            add_special_tokens=False
        )

        # print("Assistant token IDs: ",assistant_token_ids, flush=True)
        
        # Convert to tensor for comparison if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids_list = input_ids.tolist()
        else:
            input_ids_list = input_ids
        
        # Search for the assistant token sequence
        for i in range(len(input_ids_list) - len(assistant_token_ids) + 1):
            if input_ids_list[i:i+len(assistant_token_ids)] == assistant_token_ids:
                # Return position AFTER the assistant token
                return i + len(assistant_token_ids)
        
        # If not found, return None to trigger fallback
        print(f"Warning: Could not find assistant token in input_ids", flush=True)
        return None

    def _get_prompt_length(self, num_images, pil_images):
        """Calculate the length of the user prompt in tokens with caching."""
        # Check cache first
        if num_images in self._prompt_length_cache:
            return self._prompt_length_cache[num_images]
        
        # Validate inputs
        if not pil_images or num_images <= 0:
            raise ValueError(f"Invalid inputs: num_images={num_images}, pil_images={len(pil_images) if pil_images else 0}")
        
        # Create prompt-only conversation
        image_tokens = "<image>" * num_images
        prompt = f"{image_tokens}\nGenerate a radiology report for these X-rays."
        
        # Use actual images for accurate tokenization
        images_to_use = pil_images[:num_images]
        
        prompt_conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
                "images": images_to_use,
            },
            {
                "role": "<|Assistant|>",
                "content": "Report:"
            }
        ]
    
        try:
            # Process prompt to get tokenized length
            prompt_inputs = self.processor(
                conversations=prompt_conversation,
                images=images_to_use,
                force_batchify=False,
                system_prompt="",
            )
            
            # Get prompt length (handle both batched/unbatched cases)
            prompt_ids = prompt_inputs.input_ids
            if isinstance(prompt_ids, list):
                prompt_length = len(prompt_ids)
            else:
                prompt_length = prompt_ids.shape[1] if len(prompt_ids.shape) > 1 else prompt_ids.shape[0]
            
            # Validate result
            if prompt_length <= 0:
                raise ValueError(f"Computed invalid prompt_length: {prompt_length}")
            
            # Cache and return
            self._prompt_length_cache[num_images] = prompt_length
            return prompt_length
            
        except Exception as e:
            print(f"ERROR calculating prompt length for {num_images} images: {e}", flush=True)
            # Fallback: DeepSeek-VL2 uses 576 tokens per image
            estimated_length = 10 + (num_images * 576) + 30
            print(f"Using estimated prompt length: {estimated_length}", flush=True)
            self._prompt_length_cache[num_images] = estimated_length
            return estimated_length


def collate_fn(batch):
    """
    Custom collate function to handle None values from failed samples.
    
    Args:
        batch: List of samples from __getitem__, may contain None values
        
    Returns:
        dict: Batched tensors with all None values filtered out
    """
    # Filter out None values (failed samples)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "study_id": [item["study_id"] for item in batch],
        "report": [item["report"] for item in batch] if "report" in batch[0] else None,
    }