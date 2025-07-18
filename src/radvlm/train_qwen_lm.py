from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
import transformers
from pycocoevalcap.cider.cider import Cider

import os
# from PIL import Image
# import numpy as np
from peft import LoraConfig, get_peft_model

import sys
sys.path.append('/home/gustke/Projects/RadVLM')

import os
import torch.nn.functional as F
import evaluate
import numpy as np

import wandb
wandb.login()

os.environ["WANDB_PROJECT"] = "train_qwen_lm"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

from src.radvlm.data import radvlm_dataset_qwen as radvlm_dataset

def compute_metrics(eval_pred):
    """
    Optimized compute_metrics function for Trainer
    eval_pred: EvalPrediction object with predictions and label_ids
    """
    print("\nComputing metrics...")
    predictions, labels = eval_pred
    
    # Handle tuple predictions (logits, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert logits to token IDs if needed (for generation tasks)
    if predictions.dtype == np.float32 or predictions.dtype == np.float64:
        predictions = np.argmax(predictions, axis=-1)
    
    # Ensure we have numpy arrays
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Replace -100 labels with pad_token_id for proper decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Ensure integer dtype for token IDs
    predictions = predictions.astype(np.int32)
    labels = labels.astype(np.int32)
    
    # Decode predictions and labels back to text
    print("Decoding predictions and labels...")
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Decoding error: {e}")
        # Fallback: decode one by one and handle errors
        decoded_preds = []
        decoded_labels = []
        for pred, label in zip(predictions, labels):
            try:
                decoded_preds.append(tokenizer.decode(pred, skip_special_tokens=True))
                decoded_labels.append(tokenizer.decode(label, skip_special_tokens=True))
            except Exception as decode_error:
                print(f"Error decoding individual sequence: {decode_error}")
                decoded_preds.append("")
                decoded_labels.append("")
    
    # Clean up decoded text (remove extra whitespace)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    print("Sample Decoded prediction: ", decoded_preds[0][:75] if decoded_preds else "No predictions")
    # Compute BLEU score
    bleu_metric = evaluate.load("bleu")
    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(f"✓ BLEU computed successfully: {bleu_score['bleu']}")

    # Compute METEOR score
    meteor_metric = evaluate.load("meteor")
    meteor_score = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(f"✓ METEOR computed successfully: {meteor_score['meteor']}")

    # Compute ROUGE-L score
    rouge_metric = evaluate.load("rouge")
    rouge_score = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(f"✓ ROUGE-L computed successfully: {rouge_score['rougeL']}")

    # Prepare data for CIDEr evaluation

    gts = {}
    res = {}
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        img_id = str(i)
        gts[img_id] = [label]  # List of reference strings
        res[img_id] = [pred]   # List with single prediction string
    
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    print(f"✓ CIDEr computed successfully: {cider_score}")
        

    # Compute BertScore
    bert_score_metric = evaluate.load("bertscore")
    bert_score = bert_score_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    bert_score = float(np.mean(bert_score['f1'])) if bert_score['f1'] else 0.0
    print(f"✓ BERTScore computed successfully: {bert_score}")
        
    # Calculate average prediction length
    # avg_pred_length = sum(len(pred.split()) for pred in decoded_preds) / len(decoded_preds) if decoded_preds else 0
    
    return {
        "bleu": bleu_score["bleu"],
        "meteor": meteor_score["meteor"],
        "rougeL": rouge_score["rougeL"],
        "cider": cider_score,
        "bertscore": bert_score,
        "eval_samples": len(decoded_preds),
        # "avg_prediction_length": avg_pred_length
    }


try:
    here = os.path.dirname(os.path.abspath(__file__))

    model = radvlm_dataset.model
    processor = radvlm_dataset.processor
    tokenizer = processor.tokenizer

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
            label_names=["labels"],  # Ensure labels are included in the loss computation
            report_to="wandb", # Log to Weights & Biases
        )

        # Initialize Trainer or your custom training loop here
        print("Setting up the Trainer...")
        trainer = Trainer(model=model, args=training_args, train_dataset=radvlm_dataset, eval_dataset=radvlm_dataset, data_collator=data_collator, compute_metrics=compute_metrics)
        print("Starting training...")
        trainer.train()
        print("Training completed.")

        # Save the trained model
        model.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
        print("Training completed.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are in the correct conda environment (qwenenv) and that the dataset is properly loaded.")