import torch
from transformers import AutoModelForCausalLM, default_data_collator, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

import torch.nn.functional as F
# from accelerate import Accelerator


# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

# from src.radvlm.data import radvlm_dataset_deepseek as radvlm_dataset
from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.deepseek_dataset import RadVLMDatasetDeepseek

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

def setup_model_with_lora(model_path: str):
    """
    Load DeepSeek-VL2 and apply LoRA
    
    Args:
        model_path: HuggingFace model ID or local path
    """
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )

    print("Setting up model with LoRA...", flush=True)

    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Model loaded.", flush=True)
    
    # Freeze vision encoder (we only want to adapt the language model)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
            # print(f"Frozen: {name}")
    
    print("Vision encoder frozen.", flush=True)
    
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

    print("Applying LoRA...", flush=True)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("LoRA applied.", flush=True)
    print("Model setup with LoRA complete.", flush=True)
    return model


def train_deepseek_vl2():
    """Main training function with DeepSpeed and Accelerate"""
    
    import wandb
    wandb.init(
        project="deepseek-vl2-mimic-cxr",
        name="lora-r16-lr2e-4-3epochs",
        config={
            "model": "deepseek-vl2-small",
            "dataset": "mimic-cxr",
            "lora_r": 16,
            "learning_rate": 2e-4,
            "epochs": 3
        }
    )

    # Initialize Accelerator
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=4,
    #     mixed_precision='bf16',
    #     log_with="wandb",
    #     project_dir="./logs"
    # )
    
    # Model setup
    model_path = "deepseek-ai/deepseek-vl2-small" 
    model = setup_model_with_lora(model_path)
    
    # Load processor and tokenizer
    from transformers import AutoProcessor
    # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer = processor.tokenizer
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    print("Processor and tokenizer loaded.", flush=True)
    print("Preparing datasets...", flush=True)
    # Prepare datasets
    raw_data = load_dataset()
    
    train_dataset = RadVLMDatasetDeepseek(raw_data, processor, tokenizer, split='train', mode="train")
    
    val_dataset = RadVLMDatasetDeepseek(raw_data, processor, tokenizer, split='validate', mode="eval")
    
    print("Datasets prepared.", flush=True)
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )
    
    # Training arguments
    output_dir = "../results/pretraining/deepseek-vl2-mimic-cxr"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False, # Deepseek-VL2 does not support gradient checkpointing
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",  # Evaluate every eval_steps
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="loss",  # Use validation loss as metric
        greater_is_better=False,  # Lower loss is better
        save_safetensors=True,  # Use safetensors format (more efficient)
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb",  # Options: "wandb", "tensorboard", "none"
        run_name="deepseek-vl2-mimic-cxr",  # Name for wandb run
        remove_unused_columns=False,
        # DeepSpeed config (disabled for single GPU)
        # deepspeed=os.path.join(here, "ds_config.json"),
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Check for existing checkpoints to resume from
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                      if d.startswith("checkpoint")]
        if checkpoints:
            # Get the latest checkpoint
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Found checkpoint: {checkpoint}. Resuming training...", flush=True)
        else:
            print("No checkpoint found. Starting from scratch...", flush=True)
    else:
        print("No output directory found. Starting from scratch...", flush=True)
    
    # Start training (resume from checkpoint if available)
    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    trainer.save_model("../results/pretraining/deepseek-vl2-mimic-cxr-final")
    
    print("Training complete!", flush=True)


if __name__ == "__main__":
    train_deepseek_vl2()

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