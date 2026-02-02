import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from functools import partial
import json
import os
import sys
from datasets import Dataset
sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

from src.radvlm.data.build_dataset import load_preference_dataset
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from src.radvlm.data.deepseek_dpo_dataset import RadVLMDPODataset, dpo_collate_fn
from src.radvlm.trainer.dpo_trainer import DPOTrainer

def setup_model_with_lora(model_path: str, base_model_path: str = "deepseek-ai/deepseek-vl2-small"):
    """
    Load DeepSeek-VL2 with LoRA adapter for DPO finetuning
    
    Args:
        model_path: Path to pretrained model (either base model or LoRA adapter)
        base_model_path: Path to base model (used if model_path is an adapter)
        
    Returns:
        model: PEFT model ready for DPO training
        processor: DeepseekVLV2Processor for the model
    """
    
    # Check if model_path contains a LoRA adapter
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_adapter:
        # Load pretrained LoRA adapter for continued finetuning
        print(f"Detected LoRA adapter at {model_path}", flush=True)
        
        # Read base model path from adapter config
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path", base_model_path)
        
        print(f"Loading base model from {base_model_path}...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"Loading pretrained LoRA adapter from {model_path}...", flush=True)
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Pretrained adapter loaded successfully.", flush=True)
        
    else:
        # Load base model and apply new LoRA adapter
        print(f"Loading base model from {model_path}...", flush=True)
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA for language model layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
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
            ],
            inference_mode=False,
        )
        
        print("Applying new LoRA adapter...", flush=True)
        model = get_peft_model(model, lora_config)
    
    # Load processor and tokenizer
    processor_path = base_model_path if is_adapter else model_path
    print(f"Loading processor from {processor_path}...", flush=True)
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(processor_path)
    
    # Freeze vision encoder to only finetune language model
    print("Freezing vision encoder...", flush=True)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
    
    # Ensure LoRA parameters are trainable (important when loading pretrained adapter)
    if is_adapter:
        print("Enabling gradients for LoRA parameters...", flush=True)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
    # Print trainable parameters summary
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    print("Model setup complete.", flush=True)
    return model, processor


def train_dpo():
    """Main DPO training function"""
    
    import wandb
    wandb.init(
        project="deepseek-vl2-mimic-cxr-dpo",
        name="dpo-lora-r16-lr5e-5-beta0.1",
        config={
            "model": "deepseek-vl2-small",
            "dataset": "mimic-cxr-preference",
            "method": "DPO",
            "lora_r": 16,
            "learning_rate": 5e-5,
            "beta": 0.1,
            "epochs": 1,
            "max_seq_length": 3072
        }
    )
    
    # Model setup
    model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-final")
    base_model_path = "deepseek-ai/deepseek-vl2-small"
        
    print("Setting up policy model...", flush=True)
    model, processor = setup_model_with_lora(model_path, base_model_path)
    tokenizer = processor.tokenizer
    
    # For DPOTrainer, we can use ref_model=None to avoid loading a second model
    # DPO will use the initial model state as reference
    print("Using implicit reference model to save memory...", flush=True)
    ref_model = None
    
    print("Processor and tokenizer loaded.", flush=True)
    print("Loading preference datasets...", flush=True)
    
    # Load preference data
    max_seq_length = 3072
    raw_data = load_preference_dataset()
    
    # Create PyTorch datasets to validate and filter data
    train_pytorch_dataset = RadVLMDPODataset(
        raw_data, 
        processor, 
        tokenizer, 
        max_seq_length=max_seq_length,
        split='train'
    )
    
    val_pytorch_dataset = RadVLMDPODataset(
        raw_data, 
        processor, 
        tokenizer,
        max_seq_length=max_seq_length, 
        split='validate'
    )
    
    # Convert to Hugging Face Dataset format (required by DPOTrainer)
    train_data_list = [train_pytorch_dataset[i] for i in range(len(train_pytorch_dataset))]
    val_data_list = [val_pytorch_dataset[i] for i in range(len(val_pytorch_dataset))]
    
    # Filter out None values
    train_data_list = [item for item in train_data_list if item is not None]
    val_data_list = [item for item in val_data_list if item is not None]
    
    # Create HF datasets
    train_dataset = Dataset.from_list(train_data_list)
    val_dataset = Dataset.from_list(val_data_list)
    
    print(f"Train dataset size: {len(train_dataset)}", flush=True)
    print(f"Val dataset size: {len(val_dataset)}", flush=True)
    
    # DPO Configuration
    output_dir = "../results/dpo/deepseek-vl2-mimic-cxr"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # DPO typically needs fewer epochs than SFT
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # Lower than SFT
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name="deepseek-vl2-dpo",
        remove_unused_columns=False,
    )
    
    # Custom data collator
    
    def collate_fn_wrapper(batch):
        return dpo_collate_fn(batch, processor, tokenizer, max_seq_length=max_seq_length)
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn_wrapper,
        # DPO-specific parameters
        beta=0.1,  # Temperature parameter for DPO (0.1-0.5 typical range)
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
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
    
    # Start training
    print("Starting DPO training...", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    final_output = "../results/dpo/deepseek-vl2-mimic-cxr-final"
    trainer.save_model(final_output)
    print(f"Model saved to {final_output}", flush=True)
    
    print("DPO training complete!", flush=True)
    wandb.finish()


if __name__ == "__main__":
    train_dpo()