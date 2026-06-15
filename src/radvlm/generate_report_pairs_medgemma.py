from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
import json

from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.medgemma_dataset import RadVLMDatasetMedGemma, create_collate_fn_medgemma
from src.radvlm.utils.config import DATA_PROCESSED_DIR, MEDGEMMA_BASE_MODEL_PATH


class MedGemmaReportPairGenerator:
    def __init__(self, model_path, base_model_path=MEDGEMMA_BASE_MODEL_PATH, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize report pair generator for MedGemma model
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapter)
            base_model_path: Path to base pretrained model for processor/tokenizer
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        # Normalize path to resolve relative components
        model_path = os.path.abspath(model_path)
        print(f"Loading model from {model_path} onto {self.device}...", flush=True)
        
        # Load base model first
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, local_files_only=True)

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Then load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.to(self.device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True).to(self.device)

        print(f"Loading processor and tokenizer from {base_model_path}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        self.tokenizer = self.processor.tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("Model loaded and set to evaluation mode.", flush=True)

    def generate_report(self, images, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True):
        """
        Generate radiology report given image paths
        
        Args:
            images: List of PIL Image objects
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (used when do_sample=True)
            top_p: Top-p sampling parameter (used when do_sample=True)
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            
        Returns:
            Generated report text
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
        
        with torch.inference_mode():
            if do_sample:
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False
                )
            generation = generation[0][input_len:]
        
        generated_report = self.processor.decode(generation, skip_special_tokens=True)
        return generated_report

    def generate_report_pairs(self, dataloader, max_samples=None, compute_loss=False, 
                             max_new_tokens=2000, temperature=0.7, top_p=0.9):
        """
        Generate report pairs for all samples in the dataloader
        
        Args:
            dataloader: DataLoader providing batches of data
            max_samples: Maximum number of samples to process (for testing)
            max_new_tokens: Maximum tokens to generate per report
            temperature: Sampling temperature for report B
            top_p: Top-p sampling parameter for report B
        
        Returns:
            Dictionary containing:
                - study_ids: List of study IDs
                - generated_reports_a: List of generated reports (greedy)
                - generated_reports_b: List of generated reports (sampling)
                - ground_truth_reports: List of ground truth reports (if available)
        """
        study_ids = []
        generated_reports_a = []
        generated_reports_b = []
        ground_truth_reports = []
        image_paths_batch = []

        sample_count = 0
        
        for batch in tqdm(dataloader, desc="Generating report pairs"):
            # Process each sample in the batch individually for better control
            for i in range(len(batch['study_id'])):
                if max_samples and sample_count >= max_samples:
                    break
                
                study_id = batch['study_id'][i]
                images = batch['images'][i] if batch['images'][i] is not None else None
                image_paths = batch.get('image_paths', [[]])[i]
                image_paths_batch.append(image_paths)
                gt_report = batch.get('report', [None])[i]
                
                try:
                    # Generate Report A (Greedy Decoding)
                    report_a = self.generate_report(
                        images, 
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                    
                    # Generate Report B (Sampling)
                    report_b = self.generate_report(
                        images,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
                                        
                    study_ids.append(study_id)
                    generated_reports_a.append(report_a)
                    generated_reports_b.append(report_b)
                    if gt_report:
                        ground_truth_reports.append(gt_report)
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error processing study {study_id}: {e}", flush=True)
                    continue
            
            if max_samples and sample_count >= max_samples:
                break

        results = {
            'study_ids': study_ids,
            'image_paths': image_paths_batch,
            'generated_reports_a': generated_reports_a,
            'generated_reports_b': generated_reports_b,
        }
        
        if ground_truth_reports:
            results['ground_truth_reports'] = ground_truth_reports
        
        
        return results


def generate_reports():
    """Generate two reports using the pre-trained MedGemma model for a given dataset for DPO preference data curation.
    """
    
    print("Generating report pairs using pre-trained MedGemma model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-30pctdata-bf16-final"
    
    print(f"Initializing report generator with model: {model_path}", flush=True)
    report_generator = MedGemmaReportPairGenerator(model_path=model_path)
    
    print("Loading dataset...", flush=True)
    raw_data = load_dataset()

    dataset_exclude = RadVLMDatasetMedGemma(raw_data, report_generator.processor, report_generator.tokenizer, split='train', mode="eval", sample_fraction=0.3)
    # study_ids_to_exclude = []
    # for i in range(len(dataset_exclude)):
    #     study_ids_to_exclude.append(dataset_exclude[i]["study_id"])

    study_ids_to_exclude = {dataset_exclude[i]["study_id"] for i in range(len(dataset_exclude))}
    dpo_dataset = RadVLMDatasetMedGemma(
        raw_data, 
        report_generator.processor, 
        report_generator.tokenizer, 
        split='train', 
        mode="eval", 
        sample_fraction=0.06, # 3% of remaining data ~ 2500 data points
        exclude=study_ids_to_exclude
    )
    
    collate_fn = create_collate_fn_medgemma(report_generator.processor)

    dpo_dataloader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    results = report_generator.generate_report_pairs(
        dpo_dataloader, 
        max_new_tokens=2000,
        temperature=0.7,
        top_p=0.9
    )
    
    study_ids = results['study_ids']
    generated_reports_a = results['generated_reports_a']
    generated_reports_b = results['generated_reports_b']
    ground_truth_reports = results.get('ground_truth_reports', [])
    
    print(f"\nGenerated {len(study_ids)} report pairs", flush=True)

    # Save reports to text file
    output_dir = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "medgemma-1.5-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-30pctdata-bf16-final-generated-report-pairs-6pctdataset.txt")
    print(f"\nSaving reports to {output_file}", flush=True)

    with open(output_file, 'w') as f:
        for idx, (study_id, report_a, report_b) in enumerate(zip(study_ids, generated_reports_a, generated_reports_b)):
            f.write(f"Study ID: {study_id}\n")
            f.write("Report A (Greedy Decoding):\n")
            f.write(report_a + "\n\n")
            f.write("Report B (Sampling):\n")
            f.write(report_b + "\n")
            
            if ground_truth_reports and idx < len(ground_truth_reports):
                f.write("\nGround Truth:\n")
                f.write(ground_truth_reports[idx] + "\n")
    
    # Also save results as JSON for easier parsing later
    json_output_file = os.path.join(output_dir, "medgemma-1.5-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-30pctdata-bf16-final-generated-report-pairs-6pctdataset.json")
    print(f"\nSaving reports to {json_output_file}", flush=True) 
    if len(ground_truth_reports) != len(study_ids):
        print("Warning: Number of ground truth reports does not match number of generated reports.", flush=True) 
    # Convert to list of dictionaries format
    json_data = []
    for i in range(len(results['study_ids'])):
        sample_dict = {
            'study_id': results['study_ids'][i],
            'image_paths': results['image_paths'][i],
            'report_1': results['generated_reports_a'][i],
            'report_2': results['generated_reports_b'][i]
        }
        if results.get('ground_truth_reports') and i < len(results['ground_truth_reports']):
            sample_dict['ground_truth'] = results['ground_truth_reports'][i]
        json_data.append(sample_dict)
    
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print("\nReport pair generation complete!", flush=True)
    return json_data


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    generate_reports()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")