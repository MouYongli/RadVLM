from PIL import Image
import os
import sys
import pydicom
import numpy as np
import subprocess

sys.path.append('/home/gustke/Projects/RadVLM')

from deepseek_vl2.models import DeepseekVLV2Processor
from src.radvlm.utils.config import DATA_ORIGINAL_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.radvlm.data.deepseek_dataset import RadVLMDatasetDeepseek
from src.radvlm.data.build_dataset import load_dataset

if __name__ == "__main__":

    print("Creating Dataset Statistics for RadVLMDatasetDeepseek...", flush=True)

    raw_data = load_dataset()

    print(f"Loaded raw dataset with {len(raw_data)} items.", flush=True)

    model_path = "deepseek-ai/deepseek-vl2-small"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    radvlm_dataset_deepseek = RadVLMDatasetDeepseek(raw_data, vl_chat_processor, tokenizer, max_seq_length=2048, create_stats=True)
    print(f"Created Statistics for RadVLMDataset for each split.", flush=True)
    