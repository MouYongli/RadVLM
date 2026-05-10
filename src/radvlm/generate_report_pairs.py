from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from datetime import datetime
import json
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

class DeepSeekVL2ReportPairGenerator:
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
    

    def generate_report_pair(self, images, max_new_tokens=512):
        """
        Generate two radiology reports with different decoding strategies for DPO training.
        Designed for human evaluation of grammar, coherence, and writing style.
        
        Strategy A: Conservative greedy decoding - typically more grammatical and structured
        Strategy B: Sampling-based - more varied but may have stylistic differences
        
        Args:
            images: List of image file paths
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            tuple: (report_a, report_b) - two generated reports with different characteristics
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

        # Process inputs (shared for both generations)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)
        
        # Run image encoder to get the image embeddings (shared)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
            # Report A: Greedy decoding with high repetition penalty
            # More conservative, structured, and grammatically consistent
            outputs_a = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                use_cache=True,
                repetition_penalty=1.4,
                no_repeat_ngram_size=4,
                length_penalty=1.0
            )
            
            # Report B: Sampling with moderate temperature
            # More varied writing style, potentially more natural but less predictable
            outputs_b = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Enable sampling
                temperature=0.8,  # Moderate temperature for variation
                top_p=0.92,  # Nucleus sampling
                use_cache=True,
                repetition_penalty=1.2,  # Lower penalty for more natural flow
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        report_a = self.tokenizer.decode(outputs_a[0], skip_special_tokens=True)
        report_b = self.tokenizer.decode(outputs_b[0], skip_special_tokens=True)
        
        return report_a, report_b

def generate_reports():
    """Generate two reports using the pre-trained DeepSeek VL2 model for a given dataset for DPO preference data curation."""
    
    print("Generating reports using pre-trained DeepSeek VL2 model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-p13-p15-6vision-final")
    report_generator = DeepSeekVL2ReportPairGenerator(model_path=model_path)
    raw_data = load_dataset(["p18"])
    
    dpo_dataset = RadVLMDatasetDeepseek(raw_data, report_generator.processor, report_generator.tokenizer, split="train", mode="eval", sample_fraction=0.25)
    # print(dpo_dataset[0])
    
    # Custom collate function that includes all necessary fields
    def collate_fn(batch):
        return {
            "study_id": [item['study_id'] for item in batch],
            "input_ids": torch.stack([item['input_ids'] for item in batch]),
            "attention_mask": torch.stack([item['attention_mask'] for item in batch]),
            "labels": torch.stack([item['labels'] for item in batch]),
            "images": [item['images'] for item in batch],
            "report": [report_generator.tokenizer.decode(item['labels'], skip_special_tokens=True) for item in batch]  # Decode labels to get ground truth text
        }

    dpo_dataloader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    generated_reports = []
    study_ids = []
    ground_truth_reports = []
    image_paths = []
    
    for batch in tqdm(dpo_dataloader, desc="Generating reports"):
        # print("\n\nBatch: ", batch, "\n\n", flush=True)
        for i in range(len(batch['study_id'])):
            
            study_id = batch['study_id'][i]
            images = batch['images'][i] if batch['images'][i] is not None else None
            gt_report = batch['report'][i]
            
            # Generate reports
            report_a, report_b = report_generator.generate_report_pair(images)
            
            generated_reports.append((report_a, report_b))
            study_ids.append(study_id)
            ground_truth_reports.append(gt_report)
            image_paths.append(images)

    output_file = os.path.join(here, "../../results/dpo_dataset/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-p13-p15-6vision-final-generated-report-pairs-p18.txt")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for study_id, (report_a, report_b) in zip(study_ids, generated_reports):
            f.write(f"Study ID: {study_id}\n")
            f.write("Report A (Greedy Decoding):\n")
            f.write(report_a + "\n\n")
            f.write("Report B (Sampling):\n")
            f.write(report_b + "\n")
            f.write("="*80 + "\n")
    
    # Also save results as JSON for easier parsing later
    json_output_file = os.path.join(here, "../../results/dpo_dataset/lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-p13-p15-6vision-final-generated-report-pairs-p18.json")
    print(f"\nSaving reports to {json_output_file}", flush=True) 
    if len(ground_truth_reports) != len(study_ids):
        print("Warning: Number of ground truth reports does not match number of generated reports.", flush=True) 
    # Convert to list of dictionaries format
    json_data = []
    for i in range(len(study_ids)):
        sample_dict = {
            'study_id': study_ids[i],
            'image_paths': image_paths[i] if i < len(image_paths) else None,
            'report_1': generated_reports[i][0],
            'report_2': generated_reports[i][1]
        }
        if i < len(ground_truth_reports):
            sample_dict['ground_truth'] = ground_truth_reports[i]
        json_data.append(sample_dict)
    
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=4)


    print("Report pair generation complete!", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    generate_reports()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")