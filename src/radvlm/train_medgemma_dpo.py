import torch
from transformers import default_data_collator, TrainingArguments, Trainer, EarlyStoppingCallback, AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from functools import partial
import json
import os
import sys
sys.path.append('/home/ug301051/jupyterlab/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

from src.radvlm.data.build_dataset import load_preference_dataset

from src.radvlm.data.medgemma_dpo_dataset import RadVLMDPODatasetMedGemma, dpo_collate_fn
from src.radvlm.utils.config import MEDGEMMA_BASE_MODEL_PATH

def setup_model_with_lora(model_path: str = MEDGEMMA_BASE_MODEL_PATH):
    """
    Load DeepSeek-VL2 and apply LoRA
    
    Args:
        model_path: HuggingFace model ID or local path
    """

    print("Setting up model with LoRA...", flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = os.path.abspath(model_path)
    print(f"Loading MedGemma model from {model_path} onto {device}...", flush=True)
    
    # Load base model first
    base_model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_BASE_MODEL_PATH, local_files_only=True)

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Then load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True).to(device)

    # processor = AutoProcessor.from_pretrained(model_path)

    print("Model loaded.", flush=True)
    
    # Freeze vision encoder (we only want to adapt the language model)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
            # print(f"Frozen: {name}")
    
    print("Vision encoder frozen.", flush=True)
    
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                # Language model
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            inference_mode=False,
        )

        print("Applying LoRA...", flush=True)
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print("LoRA applied.", flush=True)
    
    model.print_trainable_parameters()
    
    print("Model setup with LoRA complete.", flush=True)
    return model

def train_dpo():
    """Main DPO training function"""
    
    import wandb
    wandb.init(
        project="medgemma-1.5-mimic-cxr-dpo",
        name="dpo-lora-r16-lr5e-5-beta0.1",
        config={
            "model": "medgemma-1.5",
            "dataset": "mimic-cxr-preference",
            "method": "DPO",
            "lora_r": 16,
            "learning_rate": 5e-5,
            "beta": 0.1,
            "epochs": 1,
            "max_seq_length": 3072
        }
    )
    
    model_path = os.path.join(here, "../../results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r16-lr1e-4-3epochs-linear-5pctwarmup-3earlystop-10pct-final") 
    # Load model with LoRA
    model = setup_model_with_lora(model_path=model_path)
    
    # Load processor and tokenizer from base model (they don't change during training)
    base_model_path = MEDGEMMA_BASE_MODEL_PATH
    print(f"Loading processor and tokenizer from {base_model_path}...", flush=True)
    processor: AutoProcessor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer
    
    # Load DPO dataset
    print("Loading DPO dataset...", flush=True)
    preference_data = load_preference_dataset()
    train_dataset = RadVLMDPODatasetMedGemma(preference_data, processor, tokenizer, max_seq_length=3072, split="train")
    val_dataset = RadVLMDPODatasetMedGemma(preference_data, processor, tokenizer, max_seq_length=3072, split="validate")
    
    # Create partial collate function with processor and tokenizer
    collate_fn = partial(dpo_collate_fn, processor=processor, tokenizer=tokenizer, max_seq_length=3072)
    
    output_dir = "../results/dpo/medgemma-1.5-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1"
    # Set up DPO trainer
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=50,
        report_to="wandb",
        run_name="medgemma-dpo-lora-r16-lr5e-5-beta0.1",
        remove_unused_columns=False,  # Important for custom collate_fn
        beta=0.1,  # KL penalty coefficient for DPO
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Using same dataset for eval for simplicity; ideally should have a separate eval set
        data_collator=collate_fn,
    )

    # Check for checkpoints
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                      if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Found checkpoint: {checkpoint}. Resuming training...", flush=True)
    
    print("Starting DPO training...", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)

    final_output_dir = "../results/dpo/medgemma-1.5-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1-final"
    trainer.save_model(final_output_dir)
    print(f"Final model saved to {final_output_dir}", flush=True)
    print("DPO training complete!", flush=True)
    wandb.finish()

if __name__ == "__main__":
    train_dpo()