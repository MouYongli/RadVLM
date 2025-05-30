import torch
from transformers import AutoModelForCausalLM
import pydicom
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model

import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(here, "..", "..", ".." , "models/checkpoints")))  # Adjust the path as needed

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# === Data preprocessing ===

def transform_dcm_to_jpg(input_image_path: str) -> str:
    """
    Transform DICOM image to JPEG format.
    
    Args:
        input_image_path (str): Path to the input DICOM image.
        
    Returns:
        output_image_path (str): Path to the converted JPEG image.
    """
    # Check if the JPEG file already exists
    jpeg_path = input_image_path.replace('.dcm', '.jpg')
    if os.path.exists(jpeg_path):
        return jpeg_path

    # Read the DICOM file
    dcm_image = pydicom.dcmread(input_image_path)
    pixel_array = dcm_image.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept
    # Rescale slope and intercept are used to convert pixel values to Hounsfield units (HU) in CT images
    # They are not always present, so we use default values of 1 and 0 if they are not found
    slope = getattr(dcm_image, 'RescaleSlope', 1)
    intercept = getattr(dcm_image, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept

    # Apply windowing if available
    # Windowing is used to enhance the contrast of the image
    window_center = getattr(dcm_image, 'WindowCenter', None)
    window_width = getattr(dcm_image, 'WindowWidth', None)
    if window_center and window_width:
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, lower, upper)
    else:
        lower, upper = pixel_array.min(), pixel_array.max()

    # Normalize to 8-bit
    pixel_array = ((pixel_array - lower) / (upper - lower + 1e-8) * 255).astype(np.uint8)
    image = Image.fromarray(pixel_array, mode='L')
    image.save(jpeg_path, 'JPEG')
    return jpeg_path


def load_dataset() -> list:
    """
    Load dataset with image and text pairs.

    Returns:
        dataset (list): List of dictionaries with image and text pairs.
    """
    dataset = []

    # Convert all DICOM files in the directory data/raw/MIMIC-CXR/p10 to JPEG
    for root, dirs, files in os.walk(os.path.join(here, "..", "..", "data", "raw", "MIMIC-CXR", "p10")):
        for file in files:
            if file.endswith('.dcm'):
                dcm_path = os.path.join(root, file)
                jpg_path = transform_dcm_to_jpg(dcm_path)

    # Load images and texts from the dataset
    data_dir = os.path.join(here, "..", "..", "data", "raw", "MIMIC-CXR", "p10")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text_content = f.read().strip()
                image_path = os.path.join(root, file.replace('.txt', ''))
                dataset.append({
                    "content": text_content,
                    "images": [os.path.join(root, file.replace('.txt', ''), i) for i in os.listdir(image_path) if i.endswith('.jpg')]
                    
                })
    
    return dataset

# === Train the model ===
def train_model(input_data: list, optimizer: torch.optim.Optimizer):
    """
    Train the model on the input data.

    Args:
        input_data (list): List of dictionaries with prepared input data.
    """
    for item in input_data:
        images = item["images"]
        report = item["content"]

        conversation = [
            {
            "role": "<|User|>",
            "content": "<image>"*len(images) + "\n Generate a radiology report for these X-rays.",
            "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)

        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # Tokenize the text
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Create labels for the model
        labels = tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            max_length=prepare_inputs.input_ids.shape[1],  # match input length
            truncation=True
        ).input_ids.to(vl_gpt.device)

        # run the model to get the response
        outputs = vl_gpt(
            input_ids=prepare_inputs.input_ids,
            attention_mask=prepare_inputs.attention_mask,
            images=prepare_inputs.images if hasattr(prepare_inputs, "images") else None,
            images_spatial_crop=prepare_inputs.images_spatial_crop if hasattr(prepare_inputs, "images_spatial_crop") else None,
            images_seq_mask=prepare_inputs.images_seq_mask if hasattr(prepare_inputs, "images_seq_mask") else None,
    
            labels=labels
        )

        loss = outputs.loss
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        


# === Load processor, tokenizer, model ===
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt.config.use_cache = False  # Disable cache for training
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# === Freeze vision encoder ===
for name, param in vl_gpt.named_parameters():
    if "vision_tower" in name or "visual" in name:
        param.requires_grad = False
# === Optional: Freeze cross-modal components ===
for name, param in vl_gpt.named_parameters():
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
vl_gpt = get_peft_model(vl_gpt, lora_config)

# === Optimizer ===
optimizer = torch.optim.AdamW(vl_gpt.parameters(), lr=1e-4)

if __name__ == "__main__":
    dataset = load_dataset()
    # print(dataset)
    # print("Preparing input data...")
    # input_data = prepare_input_data(dataset)
    # print(input_data[0])  # Print the first item for debugging
    print("Training the model...")
    train_model(dataset, optimizer)
    # Save the trained model
    vl_gpt.save_pretrained(os.path.join(here, "..", "..", "models", "deepseek-vl2-finetuned"))
    print("Training completed.")