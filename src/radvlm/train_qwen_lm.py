from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
import transformers
print(transformers.__file__)
print(transformers.__version__)

import os
# from PIL import Image
# import numpy as np
from peft import LoraConfig, get_peft_model

import sys
sys.path.append('/home/gustke/Projects/RadVLM')

import os
import torch.nn.functional as F

from src.radvlm.data import radvlm_dataset_qwen as radvlm_dataset

try:
    here = os.path.dirname(os.path.abspath(__file__))

    model = radvlm_dataset.model
    processor = radvlm_dataset.processor

    # === Freeze vision encoder ===
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name:
            param.requires_grad = False
    # === Optional: Freeze cross-modal components ===
    for name, param in model.named_parameters():
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
    model = get_peft_model(model, lora_config)

    if __name__ == "__main__":
        print("Training the model...")
        # train_model(dataset, optimizer)
        data_collator = None  # Define your data collator here if needed

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=1,
            save_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=True,  # if using GPU with float16 support

        )

        # Initialize Trainer or your custom training loop here
        print("Setting up the Trainer...")
        trainer = Trainer(model=model, args=training_args, train_dataset=radvlm_dataset, eval_dataset=radvlm_dataset, data_collator=data_collator)
        print("Starting training...")
        trainer.train()
        print("Training completed.")

        # Save the trained model
        model.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
        print("Training completed.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are in the correct conda environment (qwenenv) and that the dataset is properly loaded.")