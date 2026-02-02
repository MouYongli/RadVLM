from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from radgraph import F1RadGraph
import nltk

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from src.radvlm.utils.evaluation_utils import compute_metrics


class DeepSeekVL2Evaluator:
    """Unified evaluator class for DeepSeek-VL2 models (base and pretrained)"""
    
    def __init__(self, model_path, base_model_path="deepseek-ai/deepseek-vl2-small", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator for DeepSeek-VL2 language model
        
        Args:
            model_path: Path to model (can be base model or fine-tuned model)
            base_model_path: Path to base pretrained model for processor/tokenizer
        """
        self.device = device
        print(f"Loading model from {model_path} onto {self.device}...", flush=True)
        
        # Load model
        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)

        # Read base_model_path from adapter config if available (for fine-tuned models)
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            import json
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                if "base_model_name_or_path" in adapter_config:
                    base_model_path = adapter_config["base_model_name_or_path"]
                    print(f"Base model path found in adapter config: {base_model_path}", flush=True)

        print(f"Loading processor and tokenizer from {base_model_path}...", flush=True)
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(base_model_path)
        self.tokenizer = self.processor.tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("Model loaded and set to evaluation mode.", flush=True)
    
    def generate_report(self, images, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate radiology report given image paths
        
        Args:
            images: List of image file paths
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>"*len(images) + "\n Generate a radiology report for these X-rays.",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load PIL images from the conversation
        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)
        
        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        attention_mask = prepare_inputs.attention_mask.to(self.device)

        # Generate report
        with torch.no_grad():
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.4,
                no_repeat_ngram_size=4,
                length_penalty=1.0
            )
        
        generated_report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
                "role": "<|User|>",
                "content": "<image>"*len(images) + "\n Generate a radiology report for these X-rays.",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load PIL images from the conversation
        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)
        
        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        attention_mask = prepare_inputs.attention_mask.to(self.device)

        # Align labels with input embeddings length
        input_length = inputs_embeds.shape[1]
        labels_aligned = labels.to(self.device)
        
        # If labels are longer than inputs, truncate them
        if labels_aligned.shape[0] > input_length:
            labels_aligned = labels_aligned[:input_length]
        # If labels are shorter than inputs, pad them with -100 (ignore index)
        elif labels_aligned.shape[0] < input_length:
            padding = torch.full((input_length - labels_aligned.shape[0],), -100, 
                               dtype=labels_aligned.dtype, device=self.device)
            labels_aligned = torch.cat([labels_aligned, padding])
        
        # Add batch dimension if needed
        if labels_aligned.dim() == 1:
            labels_aligned = labels_aligned.unsqueeze(0)

        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model.language(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_aligned,
                return_dict=True
            )
        
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
            perplexities: List of perplexities (optional)
        Returns:
            dict: Dictionary of computed metrics including:
            - BLEU-1, BLEU-2, BLEU-3, BLEU-4
            - ROUGE-1, ROUGE-2, ROUGE-L
            - METEOR
            - RadGraph (F1, Precision, Recall)
            - Cross-Entropy Loss, Perplexity
        """
        return compute_metrics(generated_reports, ground_truth_reports, losses, perplexities)