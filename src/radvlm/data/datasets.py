import torch
from torch.utils.data import Dataset

from deepseek_vl2.utils.io import load_pil_images

import os
import torch.nn.functional as F

class RadVLMDataset(Dataset):
    def __init__(self, data, processor, tokenizer, max_seq_length=2048):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

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