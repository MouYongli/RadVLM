from .build_dataset import load_dataset
import subprocess
import json

import os

try: 
    raw_data = load_dataset()
    print(f"Loaded {len(raw_data)} items from the dataset.")

    # check which conda environment is used

    result = subprocess.run(['conda', 'info', '--json'], capture_output=True, text=True, check=True)
    conda_info = json.loads(result.stdout)

    # Extract active environment
    active_env = conda_info.get('active_prefix_name', 'unknown')
    print(f"Active environment: {active_env}")

    if active_env == "deepseekenv":

        from .deepseek_dataset import RadVLMDatasetDeepseek
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

        model_path = "deepseek-ai/deepseek-vl2-tiny"
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        radvlm_dataset_deepseek = RadVLMDatasetDeepseek(raw_data, vl_chat_processor, tokenizer, max_seq_length=2048)
        print(f"Created RadVLMDataset for Deepseek with {len(radvlm_dataset_deepseek)} items.")

    elif active_env == "qwenenv":
        from .qwen_dataset import RadVLMDatasetQwen
        radvlm_dataset_qwen = RadVLMDatasetQwen(raw_data, model_name="Qwen/Qwen2.5-VL-7B-Instruct", max_seq_length=2048)
        print(f"Created RadVLMDataset for Qwen with {len(radvlm_dataset_qwen)} items.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are in the correct conda environment (deepseekenv or qwenenv) and that the dataset is properly loaded.")