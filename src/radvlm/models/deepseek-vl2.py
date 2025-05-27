import sys
sys.path.append("/home/gustke/Projects/Deepseek_Tutorial/DeepSeek-VL2")

import os

import torch

from transformers import AutoModelForCausalLM
here = os.path.dirname(os.path.abspath(__file__))



from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from deepseek_vl2.utils.io import load_pil_images

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.get_device_name(0))


def run_deepseek_vl2(input_image_path: str, input_text: list) -> str:
    """
    Run Deepseek VL2 model on a given image and conversation.

    Args:
        input_image_path (str): Path to the input image.
        conversation (list): List of conversation turns.

    Returns:
        str: The generated response from the model.
    """


    # specify the path to the model

    model_path = "deepseek-ai/deepseek-vl2-tiny"


    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)

    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


    ## single image conversation example
    conversation = [

        {
        "role": "<|User|>",
        "content": "<image>\n" + input_text,
        "images": [input_image_path],

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

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    return f"{prepare_inputs['sft_format'][0]}\n{answer}"
    # return (f"{prepare_inputs['sft_format'][0]}",
    # answer)

if __name__ == "__main__":
    
    input_image_path = sys.argv[1]
    input_text = sys.argv[2]
    response = run_deepseek_vl2(input_image_path, input_text)
    print(response)