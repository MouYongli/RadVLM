import os
import numpy as np
import evaluate
from transformers import Trainer, EvalPrediction
from pycocoevalcap.cider.cider import Cider

def compute_metrics(tokenizer, eval_pred):
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