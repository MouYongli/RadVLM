import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, default_data_collator, TrainingArguments, Trainer
import pydicom
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model

import sys
sys.path.append('/home/gustke/Projects/RadVLM')

import os
import torch.nn.functional as F

from src.radvlm import build_dataset

here = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(here, "..", "..", ".." , "models/checkpoints")))  # Adjust the path as needed

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


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
    
# === Load processor, tokenizer, model ===
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt.config.use_cache = False  # Disable cache for training
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# === Freeze vision encoder ===
for name, param in vl_gpt.named_parameters():
    if "vision_tower" in name or "visual" in name:
        param.requires_grad = False
# === Optional: Freeze cross-modal components ===
for name, param in vl_gpt.named_parameters():
    if "cross_modal" in name or "vision" in name:
        param.requires_grad = False

lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adjust based on your model's attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
vl_gpt = get_peft_model(vl_gpt, lora_config)

if __name__ == "__main__":
    raw_data = build_dataset.load_dataset()
    dataset = RadVLMDataset(raw_data, vl_chat_processor, tokenizer, max_seq_length=2048)
    print("Training the model...")
    # train_model(dataset, optimizer)
    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=1,
        save_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # if using GPU with float16 support
    )
    print("Setting up the Trainer...")
    trainer = Trainer(
        model=vl_gpt,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Starting training...")
    trainer.train()
    # Save the trained model
    vl_gpt.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
    print("Training completed.")