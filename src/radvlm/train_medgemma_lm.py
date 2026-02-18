import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import default_data_collator, TrainingArguments, Trainer, EarlyStoppingCallback, AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.medgemma_dataset import RadVLMDatasetMedGemma, create_collate_fn_medgemma
from src.radvlm.utils.config import MEDGEMMA_BASE_MODEL_PATH

def setup_model_with_lora(model_path: str = MEDGEMMA_BASE_MODEL_PATH):
    """
    Load DeepSeek-VL2 and apply LoRA
    
    Args:
        model_path: HuggingFace model ID or local path
    """

    print("Setting up model with LoRA...", flush=True)

    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True
    )
    # processor = AutoProcessor.from_pretrained(model_path)

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
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            # Language model (Gemma-style - check your LM layer names)
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            
            # Vision encoder: last 6 layers (21-26)
            "model.vision_tower.vision_model.encoder.layers.21.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.21.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.21.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.21.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.21.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.21.mlp.fc2",
            
            "model.vision_tower.vision_model.encoder.layers.22.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.22.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.22.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.22.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.22.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.22.mlp.fc2",
            
            "model.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.23.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.23.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.23.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.23.mlp.fc2",
            
            "model.vision_tower.vision_model.encoder.layers.24.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.24.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.24.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.24.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.24.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.24.mlp.fc2",
            
            "model.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.25.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.25.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.25.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.25.mlp.fc2",
            
            "model.vision_tower.vision_model.encoder.layers.26.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.26.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.26.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.26.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.26.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.26.mlp.fc2",
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


def train_medgemma_lm():
    """Main training function with DeepSpeed and Accelerate"""
    
    import wandb
    wandb.init(
        project="medgemma-1.5-mimic-cxr",
        name="poc-lora-r32-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision",
        config={
            "model": "medgemma-1.5-4b-it",
            "dataset": "mimic-cxr",
            "lora_r": 32,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05, # 5% warmup
            "epochs": 3,
            "data_fraction": 1.00,
            "early_stopping_patience": 6
        }
    )
    
    # Model setup
    model_path = MEDGEMMA_BASE_MODEL_PATH
    model = setup_model_with_lora(model_path)
    
    # Load processor and tokenizer
    from transformers import AutoProcessor
    # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer = processor.tokenizer
    processor = AutoProcessor.from_pretrained(model_path)
    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer
    
    print("Processor and tokenizer loaded.", flush=True)
    print("Preparing datasets...", flush=True)
    # Prepare datasets
    raw_data = load_dataset()
    
    train_dataset = RadVLMDatasetMedGemma(raw_data, processor, tokenizer, split='train', mode="train", sample_fraction=1.00)  # Use 100% of training data
    
    val_dataset = RadVLMDatasetMedGemma(raw_data, processor, tokenizer, split='validate', mode="train")
    
    print("Datasets prepared.", flush=True)
    # Data collator
    # from transformers import DataCollatorForLanguageModeling
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False  # Causal LM
    # )
    
    # Create collate function with processor bound
    collate_fn = create_collate_fn_medgemma(processor)
    
    # Training arguments
    output_dir = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r32-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision"
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
        eval_strategy="steps",  # Evaluate every eval_steps
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="loss",  # Use validation loss as metric
        greater_is_better=False,  # Lower loss is better
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb",  # Options: "wandb", "tensorboard", "none"
        run_name="medgemma-1.5-mimic-cxr-poc-lora-r32-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision",  # Name for wandb run
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
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=6,  # Stop if no improvement for 6 eval_steps
                early_stopping_threshold=0.001  # Minimum change to qualify as improvement
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
    trainer.save_model("/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r32-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision-final")
    
    print("Training complete!", flush=True)


if __name__ == "__main__":
    train_medgemma_lm()