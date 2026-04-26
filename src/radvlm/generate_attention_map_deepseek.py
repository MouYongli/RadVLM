from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset
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
from src.radvlm.utils.attention_visualizer import AttentionVisualizer

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2AttentionMapGenerator:
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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("Model loaded and set to evaluation mode.", flush=True)
    

    def generate_attention_map(self, images, max_new_tokens=512, layer_idx=-1):
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>"*len(images) + "\n Generate a radiology report for these X-rays.",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    
        pil_images = load_pil_images(conversation)
    
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        # image_token_id = self.processor.image_token_id  # or check processor config
        # print("image_token_id: ", image_token_id)
        # input_ids = prepare_inputs.input_ids[0]          # (seq_len,)
        
        # image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

        images_seq_mask = prepare_inputs["images_seq_mask"][0]
        image_positions = images_seq_mask.nonzero(as_tuple=True)[0]
        
        # After building image_positions, filter out newline/separator tokens.
        # DeepSeek-VL2 typically uses token id 13 (\n) or a dedicated image_newline id.
        # IMAGE_NEWLINE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("\n")
        # # or check: model.config.image_token_id, model.config.image_newline_token_id
        
        # # input_ids shape: (seq_len,)
        # patch_mask = (input_ids[image_positions] != IMAGE_NEWLINE_TOKEN_ID)
        # image_positions_clean = image_positions[patch_mask]
    
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
    
        with torch.no_grad():
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.4,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                output_attentions=True,
                return_dict_in_generate=True,
            )
    
        # Only take the first generated token's attention — it has the full (seq_len, seq_len) shape
        # outputs.attentions: tuple[num_tokens] → tuple[num_layers] → (batch, heads, seq_len, seq_len)

        # Instead of first token only, average across all generated tokens
        raw_maps = [outputs.attentions[t][layer_idx].detach().cpu().float()
            for t in range(len(outputs.attentions))]

        # Extract the last query row from each step → shape (batch, heads, kv_len_t)
        # kv_len grows by 1 each step, so pad to the max length
        last_rows = [m[:, :, -1, :] for m in raw_maps]  # (batch, heads, kv_len_t)
        
        max_len = last_rows[-1].shape[-1]  # longest sequence (last token)
        
        padded = torch.stack([
            torch.nn.functional.pad(row, (0, max_len - row.shape[-1]))  # pad right with zeros
            for row in last_rows
        ], dim=0)  # (num_tokens, batch, heads, max_len)
        
        attention_map = padded.mean(dim=0)  # (batch, heads, max_len)
        
        # Re-add a dummy query dim if downstream code expects 4D: (batch, heads, 1, max_len)
        attention_map = attention_map.unsqueeze(2)

        # attention_map = outputs.attentions[0][layer_idx].detach().cpu().float() # (batch, heads, seq_len, seq_len)
        return attention_map, image_positions

def generate_attention_maps():
    """Generate attention maps using the pre-trained DeepSeek VL2 model for a given dataset."""
    
    print("Generating attention maps using pre-trained DeepSeek VL2 model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision-final")
    # model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-linear-5pctwarmup-3earlystop-100pct-final")
    
    attention_generator = DeepSeekVL2AttentionMapGenerator(model_path=model_path)
    raw_data = load_dataset()
    
    dpo_dataset = RadVLMDatasetDeepseek(raw_data, attention_generator.processor, attention_generator.tokenizer, split="train", mode="eval", sample_fraction=0.25)
    dpo_dataset = Subset(dpo_dataset, list(range(min(5, len(dpo_dataset)))))  # Limit to first 5 samples for attention map generation
    
    # Custom collate function that includes all necessary fields
    def collate_fn(batch):
        return {
            "study_id": [item['study_id'] for item in batch],
            "input_ids": torch.stack([item['input_ids'] for item in batch]),
            "attention_mask": torch.stack([item['attention_mask'] for item in batch]),
            "labels": torch.stack([item['labels'] for item in batch]),
            "images": [item['images'] for item in batch],
            "report": [attention_generator.tokenizer.decode(item['labels'], skip_special_tokens=True) for item in batch]  # Decode labels to get ground truth text
        }

    dpo_dataloader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    generated_attention_maps = []
    study_ids = []
    ground_truth_reports = []
    image_paths = []
    
    for batch in tqdm(dpo_dataloader, desc="Generating reports"):
        # print("\n\nBatch: ", batch, "\n\n", flush=True)
        for i in range(len(batch['study_id'])):
            
            study_id = batch['study_id'][i]
            images = batch['images'][i] if batch['images'][i] is not None else None
            gt_report = batch['report'][i]
            
            # Generate attention map
            attention_map, image_positions = attention_generator.generate_attention_map(images, layer_idx=-1)
            
            generated_attention_maps.append(attention_map)
            study_ids.append(study_id)
            ground_truth_reports.append(gt_report)
            image_paths.append(images)

            # Visualize attention map for this sample
            attention_visualizer = AttentionVisualizer(save_dir=os.path.join(here, "../../results/attention_visualizations"))
            fig = attention_visualizer.visualize_attention_heatmap(
                attention_map,
                head_idx=0,
                title=f"Attention Heatmap for Study {study_id}",
                save_path=os.path.join(here, f"../../results/attention_visualizations/study_{study_id}_heatmap.png")
            )

            for image_idx, image_path in enumerate(images):
                fig = attention_visualizer.visualize_attention_to_image(
                    image_path,
                    attention_map,
                    head_idx=0,
                    image_positions=image_positions,
                    title=f"Attention Overlay for Study {study_id} - Image {image_idx}",
                    save_path=os.path.join(here, f"../../results/attention_visualizations/study_{study_id}_image_{image_idx}_overlay.png")
                )
            
            

    # output_file = os.path.join(here, "../../results/attention_maps/deepseek-vl2-attention-maps.txt")

    # # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # with open(output_file, 'w') as f:
    #     for study_id, attention_map in zip(study_ids, generated_reports):
    #         f.write(f"Study ID: {study_id}\n")
    #         f.write("Attention Map (Greedy Decoding):\n")
    #         f.write(str(attention_map) + "\n\n")
    #         f.write("="*80 + "\n")
    
    # # Also save results as JSON for easier parsing later
    # json_output_file = os.path.join(here, "../../results/attention_maps/deepseek-vl2-attention-maps.json")
    # print(f"\nSaving reports to {json_output_file}", flush=True) 
    # if len(ground_truth_reports) != len(study_ids):
    #     print("Warning: Number of ground truth reports does not match number of generated reports.", flush=True) 
    # # Convert to list of dictionaries format
    # json_data = []
    # for i in range(len(study_ids)):
    #     sample_dict = {
    #         'study_id': study_ids[i],
    #         'image_paths': image_paths[i] if i < len(image_paths) else None,
    #         'report_1': generated_reports[i][0],
    #         'report_2': generated_reports[i][1]
    #     }
    #     if i < len(ground_truth_reports):
    #         sample_dict['ground_truth'] = ground_truth_reports[i]
    #     json_data.append(sample_dict)
    
    # with open(json_output_file, 'w') as f:
    #     json.dump(json_data, f, indent=4)


    print("Attention map generation complete!", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    generate_attention_maps()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")