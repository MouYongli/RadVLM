import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, default_data_collator, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

here = os.path.dirname(os.path.abspath(__file__))

import torch.nn.functional as F

# Fix for PyTorch 2.6 weights_only loading issue
# Monkey-patch torch.load to use weights_only=False by default
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Fix for PyTorch 2.6 weights_only loading issue with numpy in checkpoints
try:
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype])
except Exception:
    pass
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
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            # Language model
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            
            # Vision encoder (last 10 blocks + pooler)
            # "vision.blocks.17.attn.qkv",
            # "vision.blocks.17.attn.proj",
            # "vision.blocks.18.attn.qkv",
            # "vision.blocks.18.attn.proj",
            # "vision.blocks.19.attn.qkv",
            # "vision.blocks.19.attn.proj",
            # "vision.blocks.20.attn.qkv",
            # "vision.blocks.20.attn.proj",
            "vision.blocks.21.attn.qkv",
            "vision.blocks.21.attn.proj",
            "vision.blocks.22.attn.qkv",
            "vision.blocks.22.attn.proj",
            "vision.blocks.23.attn.qkv",
            "vision.blocks.23.attn.proj",
            "vision.blocks.24.attn.qkv",
            "vision.blocks.24.attn.proj",
            "vision.blocks.25.attn.qkv",
            "vision.blocks.25.attn.proj",
            "vision.blocks.26.attn.qkv",
            "vision.blocks.26.attn.proj",
            
            # "vision.blocks.17.mlp.fc1",
            # "vision.blocks.18.mlp.fc1",
            # "vision.blocks.19.mlp.fc1",
            # "vision.blocks.20.mlp.fc1",
            "vision.blocks.21.mlp.fc1",
            "vision.blocks.21.mlp.fc2",
            "vision.blocks.22.mlp.fc1",
            "vision.blocks.22.mlp.fc2",
            "vision.blocks.23.mlp.fc1",
            "vision.blocks.23.mlp.fc2",
            "vision.blocks.24.mlp.fc1",
            "vision.blocks.24.mlp.fc2",
            "vision.blocks.25.mlp.fc1",
            "vision.blocks.25.mlp.fc2",
            "vision.blocks.26.mlp.fc1",
            "vision.blocks.26.mlp.fc2",
            
            # Attention pooler
            "vision.attn_pool.q",
            "vision.attn_pool.kv",
            "vision.attn_pool.proj",
            "vision.attn_pool.mlp.fc1",
            "vision.attn_pool.mlp.fc2",

            # # Projector
            # "projector.layers.0",
            # "projector.layers.2",
        ],
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
        name="lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-6vision-corrected-early-stopping",
        config={
            "model": "deepseek-vl2-small",
            "dataset": "mimic-cxr",
            "lora_r": 8,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05, # 5% warmup
            "epochs": 3,
            "data_fraction": 1.0,
            "early_stopping_patience": 6
        }
    )
    
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
    raw_data = load_dataset(["p10", "p11", "p12"])
    
    train_dataset = RadVLMDatasetDeepseek(raw_data, processor, tokenizer, split='train', mode="train")
    
    val_dataset = RadVLMDatasetDeepseek(raw_data, processor, tokenizer, split='validate', mode="train")
    
    print("Datasets prepared.", flush=True)
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )
    
    # Training arguments
    output_dir = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-6vision-corrected-early-stopping"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False, # Deepseek-VL2 does not support gradient checkpointing
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05, # 5% warmup
        logging_steps=50,
        save_steps=200,  
        eval_steps=200,
        evaluation_strategy="steps",  # Evaluate every eval_steps
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="eval_loss",  # Use validation loss as metric
        greater_is_better=False,  # Lower loss is better
        save_safetensors=True,  # Use safetensors format (more efficient)
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb",  # Options: "wandb", "tensorboard", "none"
        run_name="deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-6vision-corrected-early-stopping",  # Name for wandb run
        remove_unused_columns=False,
        # DeepSpeed config (disabled for single GPU)
        # deepspeed=os.path.join(here, "ds_config.json"),
        dataloader_num_workers=0,  # KEY FIX
        dataloader_pin_memory=False,  # KEY FIX
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=6,  # Stop if no improvement for 6 eval_steps (1200 steps)
                early_stopping_threshold=0.0005  # Minimum improvement to reset patience
            )
        ]
    )
    
    # Check for existing checkpoints to resume from
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d) 
            for d in os.listdir(output_dir) 
            if d.startswith("checkpoint") and not d.endswith(("emergency", "interrupted"))
        ]
        
        if checkpoints:
            # Filter for valid checkpoints (must have required files)
            valid_checkpoints = []
            for ckpt in checkpoints:
                # Check for essential files
                required_files = ["trainer_state.json", "adapter_config.json"]
                has_model = (
                    os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors")) or
                    os.path.isfile(os.path.join(ckpt, "training_args.bin"))
                )
                
                if has_model and all(os.path.isfile(os.path.join(ckpt, f)) for f in required_files):
                    valid_checkpoints.append(ckpt)
                else:
                    print(f"⚠️  Skipping incomplete checkpoint: {os.path.basename(ckpt)}", flush=True)
            
            if valid_checkpoints:
                # Get the latest valid checkpoint
                checkpoint = max(valid_checkpoints, key=os.path.getctime)
                print(f"✓ Found valid checkpoint: {os.path.basename(checkpoint)}")
                print(f"  Resuming training from step {checkpoint.split('-')[-1]}...\n", flush=True)
            else:
                print("No valid checkpoints found. Starting from scratch...\n", flush=True)
        else:
            print("No checkpoints found. Starting from scratch...\n", flush=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print("Created output directory. Starting from scratch...\n", flush=True)
    
    
    # Start training (resume from checkpoint if available)
    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    trainer.save_model("/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-6vision-corrected-early-stopping-final")
    print("Training complete!", flush=True)


if __name__ == "__main__":
    train_deepseek_vl2()
