from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.deepseek_dataset import RadVLMDatasetDeepseek
from src.radvlm.utils.config import DATA_PROCESSED_DIR

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# def load_dataset(data_dir = DATA_PROCESSED_DIR) -> list:
#     """
#     Load dataset with image and text pairs.

#     Returns:
#         dataset (list): List of dictionaries with image and text pairs.
#     """
#     dataset = []

#     try:
#         # Load images and texts from the dataset
#         # data_dir = DATA_PROCESSED_DIR
#         print(f"Loading dataset from: {data_dir}", flush=True)
        
#         if not os.path.exists(data_dir):
#             raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        
#         for root, _, files in os.walk(data_dir):
#             for file in files:
#                 if file.endswith('.txt'):
#                     with open(os.path.join(root, file), 'r') as f:
#                         # Read radiology report text
#                         text_content = f.read().strip()

#                     # Check if the corresponding image directory exists
#                     if not os.path.exists(os.path.join(root, file.replace('.txt', ''))):
#                         print(f"Image directory for {file} does not exist.", flush=True)
#                         continue # If no images exist for this report, skip this datapoint
#                     else:
#                         image_path = os.path.abspath(os.path.join(root, file.replace('.txt', ''))).replace("raw", "processed/2048")
#                         if not os.path.exists(image_path):
#                             raise FileNotFoundError(f"Image directory not found: {image_path}")
#                         # Add the text and resized images to the dataset
#                         dataset.append({
#                             "file": os.path.join(root, file),
#                             "content": text_content,
#                             "images": [os.path.join(root, file.replace('.txt', ''), i) for i in os.listdir(image_path) if i.endswith('.jpg')]  
#                         })

#         return dataset
#     except Exception as e:
#         print(f"Error loading dataset: {e}", flush=True)
#         return []
    

class DeepSeekVL2Evaluator:
    def __init__(self, model_path, base_model_path="deepseek-ai/deepseek-vl2-small", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator for DeepSeek-VL2 language model
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapter)
            base_model_path: Path to base pretrained model for processor/tokenizer
        """
        self.device = device
        print(f"Loading model from {model_path} onto {self.device}...", flush=True)
        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)

        # Load processor and tokenizer from base model (they don't change during training)

        # Read base_model_path from model config if available
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
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_path,
        #     trust_remote_code=True
        # )
        
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

        # Process inputs
        # inputs = self.processor(
        #     conversations=conversation,
        #     images=pil_images,
        #     force_batchify=True,
        #     system_prompt=""
        # )

        # # Move inputs to device
        # input_ids = inputs.input_ids.to(self.device)

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
            # outputs = self.model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_new_tokens,
            #     temperature=temperature,
            #     top_p=top_p,
            #     do_sample=True,
            #     pad_token_id=self.tokenizer.eos_token_id
            # )
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        # print("Pad token ID: ",self.tokenizer.eos_token_id,flush=True)
        
        generated_report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_report

    def evaluate_reports(self, dataloader, max_samples=None):
        """
        Evaluate generated reports against ground truth
        """
        generated_reports = []
        ground_truth_reports = []
        study_ids = []
        
        sample_count = 0
        
        for batch in tqdm(dataloader, desc="Generating reports"):
            # print("\n\nBatch: ", batch, "\n\n", flush=True)
            for i in range(len(batch['study_id'])):
                if max_samples and sample_count >= max_samples:
                    break
                
                study_id = batch['study_id'][i]
                images = batch['images'][i] if batch['images'][i] is not None else None
                gt_report = batch['report'][i]
                
                # Generate report
                generated = self.generate_report(images)
                # print(f"\nStudy ID: {study_id}", flush=True)
                # print(f"Ground Truth Report: {gt_report}", flush=True)
                # print(f"Generated Report: {generated}", flush=True)
                
                generated_reports.append(generated)
                ground_truth_reports.append(gt_report)
                study_ids.append(study_id)
                
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
        print(f"Study IDs: {study_ids}")
        return study_ids, generated_reports, ground_truth_reports


    def compute_metrics(self, generated_reports, ground_truth_reports):
        """
        Compute evaluation metrics for generated reports
        
        Args:
            generated_reports: List of generated report strings
            ground_truth_reports: List of ground truth report strings
        """
        print("\nComputing evaluation metrics...", flush=True)
        
        # Handle single string inputs by converting to lists
        if isinstance(generated_reports, str):
            generated_reports = [generated_reports]
        if isinstance(ground_truth_reports, str):
            ground_truth_reports = [ground_truth_reports]
        
        # Validate inputs
        assert len(generated_reports) == len(ground_truth_reports), \
            f"Mismatch: {len(generated_reports)} generated vs {len(ground_truth_reports)} ground truth reports"
        
        # Compute BLEU scores for each pair
        bleu_scores = []
        for gen_report, gt_report in zip(generated_reports, ground_truth_reports):
            # BLEU expects reference as list of tokens and hypothesis as list of tokens
            reference = gt_report.split()
            hypothesis = gen_report.split()
            bleu = sentence_bleu([reference], hypothesis)
            bleu_scores.append(bleu)
        
        # Calculate average BLEU score
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
        return {
            "bleu": avg_bleu,
            "bleu_scores": bleu_scores,  # Individual scores if you need them
            "eval_samples": len(generated_reports)
        }


def evaluate_pre_training():
    """Evaluate the pre-trained DeepSeek VL2 model from file on a test set"""
    
    print("Evaluating pre-trained DeepSeek VL2 model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-final")
    evaluator = DeepSeekVL2Evaluator(model_path=model_path)
    raw_data = load_dataset()
    # print("Raw data item example:", raw_data[0], flush=True)
    val_dataset = RadVLMDatasetDeepseek(raw_data, evaluator.processor, evaluator.tokenizer, split='train', mode="eval") # TODO: change split to 'validate'/'test' when available
    # print("val dataset item example:", val_dataset[0], flush=True)

    # Custom collate function that includes all necessary fields
    def collate_fn(batch):
        return {
            "study_id": [item['study_id'] for item in batch],
            "input_ids": torch.stack([item['input_ids'] for item in batch]),
            "attention_mask": torch.stack([item['attention_mask'] for item in batch]),
            "labels": torch.stack([item['labels'] for item in batch]),
            "images": [item['images'] for item in batch],
            "report": [evaluator.tokenizer.decode(item['labels'], skip_special_tokens=True) for item in batch]  # Decode labels to get ground truth text
        }

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    study_ids, generated_reports, ground_truth_reports = evaluator.evaluate_reports(val_dataloader, max_samples=500)
    # save the study_ids and generated reports to text file for further inspection
    output_file = 'generated_reports.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATED REPORTS EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        
        for study_id, gen_report, gt_report in zip(study_ids, generated_reports, ground_truth_reports):
            f.write(f"Study ID: {study_id}\n")
            f.write("-" * 80 + "\n")
            f.write("Generated Report:\n")
            f.write(f"{gen_report}\n\n")
            f.write("Ground Truth Report:\n")
            f.write(f"{gt_report}\n")
            f.write("=" * 80 + "\n\n")

    print(f"Reports saved to {output_file}", flush=True)

    metrics_dict = evaluator.compute_metrics(generated_reports, ground_truth_reports)

    print("Evaluation Metrics:", metrics_dict, flush=True)
    print("Evaluation complete!", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    evaluate_pre_training()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")