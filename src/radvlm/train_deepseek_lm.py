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

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from src.radvlm.data import radvlm_dataset

here = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(here, "..", "..", ".." , "models/checkpoints")))  # Adjust the path as needed

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
        train_dataset=radvlm_dataset,
        eval_dataset=radvlm_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Starting training...")
    trainer.train()
    # Save the trained model
    vl_gpt.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
    print("Training completed.")