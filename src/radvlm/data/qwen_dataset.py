import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import os
import torch.nn.functional as F

class RadVLMDatasetQwen(Dataset):
    def __init__(self, data, model_name="Qwen/Qwen2.5-VL-7B-Instruct", max_seq_length=2048):
        self.data = data
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        images = item["images"]
        report = item["content"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [{"type": "text", "text": "Generate a radiology report for these X-rays."}],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Pad/truncate input_ids and attention_mask to max_seq_length
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        input_ids = F.pad(input_ids, (0, self.max_seq_length - input_ids.shape[0]))[:self.max_seq_length]
        attention_mask = F.pad(attention_mask, (0, self.max_seq_length - attention_mask.shape[0]))[:self.max_seq_length]

        labels = self.processor.tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True
        ).input_ids.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }