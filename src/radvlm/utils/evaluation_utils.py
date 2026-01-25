from transformers import AutoModelForCausalLM
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
            perplexities: List of perplexity values (optional)
        
        Returns:
            Dictionary containing average scores and individual scores for:
            - BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR
            - RadGraph (F1, Precision, Recall)
            - Cross-Entropy Loss, Perplexity
        """
        
        # Download required NLTK data
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
        
        print("\nComputing evaluation metrics...", flush=True)
        
        # Handle single string inputs by converting to lists
        if isinstance(generated_reports, str):
            generated_reports = [generated_reports]
        if isinstance(ground_truth_reports, str):
            ground_truth_reports = [ground_truth_reports]
        
        # Validate inputs
        assert len(generated_reports) == len(ground_truth_reports), \
            f"Mismatch: {len(generated_reports)} generated vs {len(ground_truth_reports)} ground truth reports"
        
        # Initialize score lists
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        meteor_scores = []
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Compute BLEU, ROUGE, and METEOR scores for each pair
        print("Computing BLEU, ROUGE, and METEOR scores...", flush=True)
        for gen_report, gt_report in zip(generated_reports, ground_truth_reports):
            # BLEU
            reference = gt_report.split()
            hypothesis = gen_report.split()
            bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0))
            bleu3 = sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0))
            bleu4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
            
            # ROUGE
            rouge_scores = scorer.score(gt_report, gen_report)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
            
            # METEOR
            meteor = meteor_score([gt_report.split()], gen_report.split())
            meteor_scores.append(meteor)
        
        # Compute RadGraph scores in batches to avoid OOM
        print("Computing RadGraph scores...", flush=True)
        try:          
            f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
            
            # Process in batches to avoid OOM
            batch_size = 16
            all_f1_simple_scores = []
            all_f1_partial_scores = []
            all_f1_complete_scores = []
            
            for i in range(0, len(generated_reports), batch_size):
                batch_end = min(i + batch_size, len(generated_reports))
                batch_hyps = generated_reports[i:batch_end]
                batch_refs = ground_truth_reports[i:batch_end]
                
                print(f"Processing RadGraph batch {i//batch_size + 1}/{(len(generated_reports) + batch_size - 1)//batch_size} (samples {i+1}-{batch_end})...", flush=True)
                
                _, reward_list, _, _ = f1radgraph(hyps=batch_hyps, refs=batch_refs)
                print(f"RadGraph batch {i//batch_size + 1} processed.", flush=True)
                all_f1_simple_scores.extend(reward_list[0])
                all_f1_partial_scores.extend(reward_list[1])
                all_f1_complete_scores.extend(reward_list[2])
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Compute mean scores
            radgraph_f1_mean = (
                sum(all_f1_simple_scores) / len(all_f1_simple_scores),
                sum(all_f1_partial_scores) / len(all_f1_partial_scores),
                sum(all_f1_complete_scores) / len(all_f1_complete_scores)
            )
            
            # Combine individual scores
            radgraph_individual = []
            for i in range(len(generated_reports)):
                t = (all_f1_simple_scores[i], all_f1_partial_scores[i], all_f1_complete_scores[i])
                radgraph_individual.append(t)          
        except Exception as e:
            print(f"Warning: RadGraph computation failed: {e}", flush=True)
            radgraph_f1_mean = (0.0, 0.0, 0.0)
            radgraph_individual = [(0.0, 0.0, 0.0)] * len(generated_reports)
        
        # Calculate averages
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
        avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0
        avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0
        avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        
        # Calculate loss and perplexity averages if provided
        avg_loss = None
        avg_perplexity = None
        if losses is not None and len(losses) > 0:
            valid_losses = [l for l in losses if not torch.isnan(torch.tensor(l))]
            avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
        if perplexities is not None and len(perplexities) > 0:
            valid_perplexities = [p for p in perplexities if not torch.isnan(torch.tensor(p))]
            avg_perplexity = sum(valid_perplexities) / len(valid_perplexities) if valid_perplexities else 0.0
        
        return {
            "bleu1": avg_bleu1,
            "bleu2": avg_bleu2,
            "bleu3": avg_bleu3,
            "bleu4": avg_bleu4,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "meteor": avg_meteor,
            "radgraph_f1": radgraph_f1_mean,
            "cross_entropy_loss": avg_loss,
            "perplexity": avg_perplexity,
            "bleu1_scores": bleu1_scores,
            "bleu2_scores": bleu2_scores,
            "bleu3_scores": bleu3_scores,
            "bleu4_scores": bleu4_scores,
            "rouge1_scores": rouge1_scores,
            "rouge2_scores": rouge2_scores,
            "rougeL_scores": rougeL_scores,
            "meteor_scores": meteor_scores,
            "radgraph_individual": radgraph_individual,
            "losses": losses if losses is not None else [],
            "perplexities": perplexities if perplexities is not None else [],
            "eval_samples": len(generated_reports)
        }
