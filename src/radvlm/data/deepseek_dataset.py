import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from deepseek_vl2.utils.io import load_pil_images

from src.radvlm.utils.config import DATA_PROCESSED_DIR

class RadVLMDatasetDeepseek(Dataset):
    def __init__(self, data, processor, tokenizer, max_seq_length=2048, split=None):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        if split is not None:
            self.data = self._filter_data_by_split(self.data, split)

    def _filter_data_by_split(self, data, split):
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
            
            filtered_data_item = {"content": item["content"], "images": []}
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
                        
        print(f"Filtered data to {len(filtered_data)} items for split '{split}'")

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