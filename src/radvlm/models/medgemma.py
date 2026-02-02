from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
from src.radvlm.utils.config import MEDGEMMA_BASE_MODEL_PATH
def run_medgemma_example():
    print("Loading MedGemma model and processor...", flush=True)
    model_id = MEDGEMMA_BASE_MODEL_PATH
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded successfully.", flush=True)
    # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate a radiology report for these X-rays."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    print("Running inference...", flush=True)
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        generation = generation[0][input_len:]
    print("Inference complete.", flush=True)
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

if __name__ == "__main__":
    try:
        run_medgemma_example()
    except Exception as e:
        print("An error occurred while running the MedGemma example:", str(e))