import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from deepseek_vl2.utils.io import load_pil_images

from src.radvlm.utils.config import DATA_PROCESSED_DIR

class RadVLMDatasetDeepseek(Dataset):
    def __init__(self, data, processor, tokenizer, max_seq_length=2048, split=None, create_stats=False):
        """
        RadVLM Dataset for Deepseek VL2 model.
        Args:
            data (list): List of dictionaries with image and text pairs.
            processor: Deepseek VL2 processor for image and text processing.
            tokenizer: Tokenizer for text tokenization.
            max_seq_length (int): Maximum sequence length for tokenization.
            split (str, optional): Dataset split to use ('train', 'val', 'test'). Defaults to None.
            create_stats (bool, optional): Whether to create dataset statistics. Defaults to False.
        """
        self.data = self._preprocess_reports(data, create_stats)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        if split is not None:
            self.data = self._filter_data_by_split(self.data, split)

    def _filter_data_by_split(self, data, split, keep_file_names=None):
        # Read "mimic-cxr-split.csv" to filter data by split
        split_file = os.path.join(DATA_PROCESSED_DIR, "mimic-cxr-2.0.0-split.csv")
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}. Returning unfiltered data.")
            return data
        import pandas as pd
        df_split = pd.read_csv(split_file)
        filtered_data = []
        for item in data:
            images = item["images"]
            if not images:
                continue
            
            if keep_file_names is None:
                filtered_data_item = {"content": item["content"], "images": []}
            else:
                filtered_data_item = {"file": item["file"], "content": item["content"], "images": []}
            # For each image, check if it belongs to the desired split. If yes, add it to the filtered images.
            for image in images:
                if not os.path.exists(image):
                    print(f"Image file does not exist: {image}")
                    continue
                image_filename = os.path.basename(image).replace('.jpg', '')
                split_row = df_split[df_split['dicom_id'] == image_filename]
                if not split_row.empty:
                    item_split = split_row['split'].values[0]
                    if item_split == split:
                        filtered_data_item["images"].append(image)
            if filtered_data_item["images"]:
                filtered_data.append(filtered_data_item)
                if len(images) != len(filtered_data_item["images"]):
                    print(f"Item {item.get('file', 'unknown')} - kept {len(filtered_data_item['images'])} out of {len(images)} images for split '{split}'")
                        
        print(f"Filtered data to {len(filtered_data)} items for split '{split}'")

        return filtered_data


    def _preprocess_reports(self, data, create_stats=False):

        if create_stats:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(DATA_PROCESSED_DIR, "../logs")
            os.makedirs(logs_dir, exist_ok=True)

            for split, data_split in [("full", data), ("train", self._filter_data_by_split(data, "train", keep_file_names=True)), ("validate", self._filter_data_by_split(data, "validate", keep_file_names=True)), ("test", self._filter_data_by_split(data, "test", keep_file_names=True))]:
                with open(os.path.join(logs_dir, f"{split}_dataset_size.txt"), 'w') as f:
                    f.write(f"Dataset size for {split} split: {len(data_split)} items\n")

                # Log report length statistics
                if data_split:
                    report_lengths = [(item["file"],len(item["content"].strip())) for item in data_split]
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
                        print(f"No valid reports found for length statistics in split '{split}'.")
                        with open(os.path.join(logs_dir, f"report_length_stats_{split}.txt"), 'w') as f:
                            f.write(f"No valid reports found for length statistics in split '{split}'.\n")
                        with open(os.path.join(logs_dir, f"report_lengths_{split}.csv"), 'w') as f:
                            f.write("file,report_length\n")

                    # Remove entries with empty reports
                    filtered_data = [item for item in data_split if item["content"].strip() != ""]
                    print(f"Removed empty reports. {len(filtered_data)} items remain. {len(data_split) - len(filtered_data)} items were removed.")

                    with open(os.path.join(logs_dir, f"empty_report_removal_{split}.txt"), 'w') as f:
                        f.write(f"Total items before removal: {len(data_split)}\n")
                        f.write(f"Total items after removal: {len(filtered_data)}\n")
                        f.write(f"Total empty reports removed: {len(data_split) - len(filtered_data)}\n")

                    # Save short report examples to a file for inspection
                    short_report_file = os.path.join(logs_dir, f"short_reports_{split}.txt")
                    with open(short_report_file, 'w') as f:
                        for item in filtered_data:
                            if len(item["content"].strip().split()) < 20:
                                f.write(f"Report: {item['content'].strip()}\nImages: {item['images']}\n\n")

                    # Save long report examples to a file for inspection
                    long_report_file = os.path.join(logs_dir, f"long_reports_{split}.txt")
                    with open(long_report_file, 'w') as f:
                        for item in filtered_data:
                            if len(item["content"].strip().split()) > 800:
                                f.write(f"Report: {item['content'].strip()}\nImages: {item['images']}\n\n")
                else:
                    print(f"No data available for split '{split}' to create statistics.")
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
            print(f"Removed empty reports. {len(filtered_data)} items remain. {len(data) - len(filtered_data)} items were removed.")

        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = item["images"]
        report = item["content"]

        conversation = [
            {
            "role": "<|User|>",
            "content": "<image>"*len(images) + "\n Generate a radiology report for these X-rays.",
            "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        images = load_pil_images(conversation)

        inputs = self.processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt="",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)

        # Pad/truncate input_ids and attention_mask to max_seq_length
        input_ids = F.pad(input_ids, (0, self.max_seq_length - input_ids.shape[0]))[:self.max_seq_length]
        attention_mask = F.pad(attention_mask, (0, self.max_seq_length - attention_mask.shape[0]))[:self.max_seq_length]

        labels = self.tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True
        ).input_ids.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }