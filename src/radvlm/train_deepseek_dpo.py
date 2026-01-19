import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer
from functools import partial
import json
import os
import sys
sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

from src.radvlm.data.build_dataset import load_preference_dataset
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from src.radvlm.data.deepseek_dpo_dataset import RadVLMDPODataset, dpo_collate_fn


def setup_model_with_lora(model_path: str):
    """
    Load DeepSeek-VL2 with optional LoRA adapter
    
    Args:
        model_path: HuggingFace model ID or local path to base model
        adapter_path: Path to pretrained LoRA adapter (from SFT training)
    """
    
    print("Loading base model...", flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {model_path} onto {device}...", flush=True)
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)

    # Load processor and tokenizer from base model (they don't change during training)

    # Read base_model_path from model config if available
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
            if "base_model_name_or_path" in adapter_config:
                base_model_path = adapter_config["base_model_name_or_path"]
                print(f"Base model path found in adapter config: {base_model_path}", flush=True)

    print(f"Loading processor and tokenizer from {base_model_path}...", flush=True)
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer
    
    # Freeze vision encoder
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
    
    print("Vision encoder frozen.", flush=True)
    
    # If no adapter was loaded, apply new LoRA
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
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
        
        print("Applying LoRA...", flush=True)
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    print("Model setup complete.", flush=True)
    
    return model


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
      
        
    print("Setting up policy model...", flush=True)
    model = setup_model_with_lora(model_path)
    
    # Create reference model (frozen copy for DPO)
    print("Setting up reference model...", flush=True)
    ref_model = setup_model_with_lora(model_path)
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference model created.", flush=True)
    
    # Load processor
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    print("Processor and tokenizer loaded.", flush=True)
    print("Loading preference datasets...", flush=True)
    
    # Load preference data
    max_seq_length = 3072
    raw_data = load_preference_dataset()
    
    train_dataset = RadVLMDPODataset(
        raw_data, 
        processor, 
        tokenizer, 
        max_seq_length=max_seq_length,
        split='train'
    )
    
    val_dataset = RadVLMDPODataset(
        raw_data, 
        processor, 
        tokenizer,
        max_seq_length=max_seq_length, 
        split='validate'
    )
    
    print(f"Train dataset size: {len(train_dataset)}", flush=True)
    print(f"Val dataset size: {len(val_dataset)}", flush=True)
    
    # DPO Configuration
    output_dir = "../results/dpo/deepseek-vl2-mimic-cxr"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # DPO typically needs fewer epochs than SFT
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
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
        # DPO-specific parameters
        beta=0.1,  # Temperature parameter for DPO (0.1-0.5 typical range)
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    )
    
    # Custom data collator
    
    data_collator = partial(
        dpo_collate_fn, 
        processor=processor, 
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Check for checkpoints
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                      if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Found checkpoint: {checkpoint}. Resuming training...", flush=True)
    
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