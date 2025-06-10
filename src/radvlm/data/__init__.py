from .datasets import RadVLMDataset
from .build_dataset import load_dataset
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

raw_data = load_dataset()
print(f"Loaded {len(raw_data)} items from the dataset.")
radvlm_dataset = RadVLMDataset(raw_data, vl_chat_processor, tokenizer, max_seq_length=2048)
print(f"Created RadVLMDataset with {len(radvlm_dataset)} items.")