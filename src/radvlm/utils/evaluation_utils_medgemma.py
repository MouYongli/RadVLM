from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
import torch
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from radgraph import F1RadGraph
import nltk
from peft import PeftModel

from src.radvlm.utils.evaluation_utils import compute_metrics

class MedGemmaEvaluator:
    """Evaluator class for MedGemma models"""
    
    def __init__(self, model_path, base_model_path = "/hpcwork/ug301051/models/medgemma-1.5-4b-it", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator for MedGemma model
        
        Args:
            model_path: Path to MedGemma model
            base_model_path: Path to base MedGemma model for processor
        """
        self.device = device
        # Normalize path to resolve relative components
        model_path = os.path.abspath(model_path)
        print(f"Loading MedGemma model from {model_path} onto {self.device}...", flush=True)
        
        # Load base model first
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, local_files_only=True)

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Then load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.to(self.device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True).to(self.device)

        print("Loading processor and tokenizer...", flush=True)
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        
        self.tokenizer = self.processor.tokenizer

        self.model.eval()
        print("MedGemma model loaded and set to evaluation mode.", flush=True)

    def generate_report(self, images, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate radiology report given image paths
        
        Args:
            images: List of image file paths
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [
                    {"type": "text", "text": "Generate a radiology report for these X-rays."}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        print("Running inference...", flush=True)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
        print("Inference complete.", flush=True)
        generated_report = self.processor.decode(generation, skip_special_tokens=True)
        return generated_report


    def compute_loss_and_perplexity(self, images, labels):
        """
        Compute cross-entropy loss and perplexity for a given batch.
        
        Args:
            images: List of image file paths
            labels: Ground truth token IDs (tensor)
        
        Returns:
            tuple: (cross_entropy_loss, perplexity)
        """
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [
                    {"type": "text", "text": "Generate a radiology report for these X-rays."}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        # Prepare labels: ensure they're 2D and on the correct device
        if labels is not None:
            labels = labels.to(self.model.device)
            # If labels are 1D, unsqueeze to make them 2D (batch_size=1, seq_len)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            # Pad or truncate labels to match input length
            input_length = inputs["input_ids"].shape[-1]
            label_length = labels.shape[-1]
            if label_length < input_length:
                # Pad with -100 (ignore index)
                padding = torch.full((labels.shape[0], input_length - label_length), -100, 
                                    dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, padding], dim=1)
            elif label_length > input_length:
                # Truncate to input length
                labels = labels[:, :input_length]
        
        # Forward pass with labels to compute loss
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        # Extract loss and compute perplexity
        loss = outputs.loss.item()
        perplexity = torch.exp(outputs.loss).item()
                
        return loss, perplexity

    def evaluate_reports(self, dataloader, max_samples=None):
        """
        Evaluate generated reports against ground truth
        
        Args:
            dataloader: DataLoader providing batches with images, labels, reports
            max_samples: Maximum number of samples to evaluate (None for all)
        
        Returns:
            tuple: (study_ids, generated_reports, ground_truth_reports, losses, perplexities)
        """
        generated_reports = []
        ground_truth_reports = []
        study_ids = []
        losses = []
        perplexities = []
        
        sample_count = 0
        
        for batch in tqdm(dataloader, desc="Generating reports"):
            # print(batch, flush=True)
            for i in range(len(batch['study_id'])):
                if max_samples and sample_count >= max_samples:
                    break
                
                study_id = batch['study_id'][i]
                images = batch['images'][i] if batch['images'][i] is not None else None
                gt_report = batch['report'][i]
                labels = batch['labels'][i]
                
                # Generate report
                generated = self.generate_report(images)
                
                # Compute loss and perplexity
                try:
                    loss, perplexity = self.compute_loss_and_perplexity(images, labels)
                    losses.append(loss)
                    perplexities.append(perplexity)
                except Exception as e:
                    print(f"Warning: Loss computation failed for study {study_id}: {e}", flush=True)
                    losses.append(float('nan'))
                    perplexities.append(float('nan'))
                
                generated_reports.append(generated)
                ground_truth_reports.append(gt_report)
                study_ids.append(study_id)
                
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
        
        return study_ids, generated_reports, ground_truth_reports, losses, perplexities
    
    def compute_metrics(self, generated_reports, ground_truth_reports, losses=None, perplexities=None):
        """
        Compute evaluation metrics for generated reports
        
        Args:
            generated_reports: List of generated report strings
            ground_truth_reports: List of ground truth report strings
            losses: List of cross-entropy losses (optional)
            perplexities: List of perplexity values (optional)
        Returns:
            Dictionary containing average scores and individual scores for:
            - BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR
            - RadGraph (F1, Precision, Recall)
            - Cross-Entropy Loss, Perplexity
        """
        return compute_metrics(generated_reports, ground_truth_reports, losses, perplexities)