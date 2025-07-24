import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class RewardDataset(Dataset):
    """Custom dataset for reward model training."""
    def __init__(self, data_path, tokenizer, processor=None, kg_evaluator=None, max_image_resolution=2048):
        
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.kg_evaluator = kg_evaluator
        if max_image_resolution not in [2048, 1024]:
            raise ValueError("max_image_resolution must be either 2048 or 1024.")
        self.max_image_resolution = max_image_resolution
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item by index.
        This method retrieves the data item, processes it, and returns the necessary inputs for the model.
        It handles both the reports and the images, ensuring they are formatted correctly for the model.
        It also calculates clinical scores if a knowledge graph evaluator is provided.
        """
        item = self.data[idx]

        try:
            image_paths = item["image_paths"]
            report_1 = item["report_1"]
            report_2 = item["report_2"]
            preference = item["radiologist_preference"]
            clinical_score_1 = self.get_clinical_score(report_1, item["ground_truth"])
            clinical_score_2 = self.get_clinical_score(report_2, item["ground_truth"])
        except KeyError as e:
            raise KeyError(f"Missing key in data item {idx}: {e}")

        # Validate input types
        if not isinstance(report_1, str) or not isinstance(report_2, str):
            raise ValueError("Report must be a string.")
        if not isinstance(image_paths, list) or not all(isinstance(p, str) for p in image_paths):
            raise ValueError("Image paths must be a list of strings.")
        if preference not in ["report_1", "report_2"]:
            raise ValueError("Preference must be either 'report_1' or 'report_2'.")
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Format input for the model
        text1, image_inputs1, video_inputs1 = self.format_input(report_1, image_paths)
        text2, image_inputs2, video_inputs2 = self.format_input(report_2, image_paths)
        
        # Tokenize inputs
        inputs_1 = self.processor(text=[text1], images=image_inputs1, videos=video_inputs1, padding=True, return_tensors="pt")
        inputs_2 = self.processor(text=[text2], images=image_inputs2, videos=video_inputs2, padding=True, return_tensors="pt")
        

        return {
            "input_ids_1": inputs_1.input_ids.squeeze(0),
            "attention_mask_1": inputs_1.attention_mask.squeeze(0),
            "input_ids_2": inputs_2.input_ids.squeeze(0),
            "attention_mask_2": inputs_2.attention_mask.squeeze(0),
            "preference": torch.tensor(1 if preference == "report_1" else 0, dtype=torch.long),
            "clinical_score_1": torch.tensor(clinical_score_1, dtype=torch.float),
            "clinical_score_2": torch.tensor(clinical_score_2, dtype=torch.float),
        }
    
    def load_data(self, data_path):
        """Load data from the specified path."""
        # The data is expected to be in this JSON format:
        # [
        #     {"image_path": "path/to/xray.jpg", "report_1": "First report text", "report_2": "Second report text", "radiologist_preference": "report_1", "ground_truth": "Ground truth text"},
        #     ...
        # ]
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, 'r') as f:
            return json.load(f)
        
    def get_clinical_score(self, report, ground_truth):
        """Calculate clinical score based on the report and ground truth."""
        if self.kg_evaluator:
            return self.kg_evaluator.evaluate(report, ground_truth)
        else:
            # Placeholder for a default scoring mechanism
            return 0.0
    
    def format_input(self, report, image_paths):
        """Format input for the model."""
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path.replace("raw", f"processed/{self.max_image_resolution}")} for image_path in image_paths] +
                           [{"type": "text", "text": "Generate a radiology report for these X-rays."}],
            },
            {"role": "assistant", "content": report}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        return text, image_inputs, video_inputs

    
# Example usage
dataset = RewardDataset(
    data_path='Projects/RadVLM/src/radvlm/data/preference_data.json',
    processor=AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True), 
    tokenizer=AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True).tokenizer,
    kg_evaluator=None,  # TODO: Replace with actual knowledge graph evaluator
    max_image_resolution=2048,  # Either 2048 or 1024
)

i = dataset[0]
print("Sample data item:")
print(i)
print(f"Shape of input_ids_1: {i['input_ids_1'].shape}")
print(f"Shape of input_ids_2: {i['input_ids_2'].shape}")
print(f"Shape of attention_mask_1: {i['attention_mask_1'].shape}")
print(f"Shape of attention_mask_2: {i['attention_mask_2'].shape}")
