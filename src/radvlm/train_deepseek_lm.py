import torch
from transformers import AutoModelForCausalLM, default_data_collator, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.append('/home/gustke/Projects/RadVLM')

import os
import torch.nn.functional as F
from accelerate import Accelerator


# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from src.radvlm.data import radvlm_dataset_deepseek as radvlm_dataset


def setup_model_with_lora(model_path: str):
    """
    Load DeepSeek-VL2 and apply LoRA
    
    Args:
        model_path: HuggingFace model ID or local path
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Freeze vision encoder (we only want to adapt the language model)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
            print(f"Frozen: {name}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank (increase for more capacity: 32, 64)
        lora_alpha=32,  # LoRA scaling factor
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  # Apply LoRA to attention and MLP layers
        inference_mode=False,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# try:
#     here = os.path.dirname(os.path.abspath(__file__))

#     model_path = "deepseek-ai/deepseek-vl2-small"
#     vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
#     vl_gpt.config.use_cache = False  # Disable cache for training
#     vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

#     tokenizer = radvlm_dataset.processor.tokenizer

#     # === Freeze vision encoder ===
#     for name, param in vl_gpt.named_parameters():
#         if "vision_tower" in name or "visual" in name:
#             param.requires_grad = False
#     # === Optional: Freeze cross-modal components ===
#     for name, param in vl_gpt.named_parameters():
#         if "cross_modal" in name or "vision" in name:
#             param.requires_grad = False

#     # lora_config = LoraConfig(
#     #     r=8,  # Rank of LoRA matrices
#     #     lora_alpha=16,
#     #     target_modules=["q_proj", "v_proj"],  # Adjust based on your model's attention modules
#     #     lora_dropout=0.05,
#     #     bias="none",
#     #     task_type="CAUSAL_LM"
#     # )
#     # vl_gpt = get_peft_model(vl_gpt, lora_config)

#     if __name__ == "__main__":
        
#         print("Training the model...")
#         data_collator = default_data_collator

#         training_args = TrainingArguments(
#             output_dir="./results",
#             per_device_train_batch_size=4,
#             per_device_eval_batch_size=4,
#             num_train_epochs=3,
#             evaluation_strategy="steps",
#             save_strategy="steps",
#             logging_steps=1,
#             save_steps=100,
#             learning_rate=5e-5,
#             weight_decay=0.01,
#             fp16=False,  # if using GPU with float16 support
#             bf16=True,  # if using GPU with bfloat16 support
#         )
#         print("Setting up the Trainer...")
#         trainer = Trainer(
#             model=vl_gpt,
#             args=training_args,
#             train_dataset=radvlm_dataset,
#             eval_dataset=radvlm_dataset,
#             tokenizer=tokenizer,
#             data_collator=data_collator,
#         )
#         print("Starting training...")
#         trainer.train()
#         # Save the trained model
#         vl_gpt.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
#         print("Training completed.")
# except Exception as e:
#     print(f"An error occurred: {e}")
#     print("Please ensure you are in the correct conda environment (deepseekenv) and that the dataset is properly loaded.")