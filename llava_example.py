import argparse

import torch
from PIL import Image
from transformers import TextIteratorStreamer
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def model_generate(model, image_file_path, image_processor, prompt, tokenizer):
    image = load_image(image_file_path)
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].half().cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=100,
            streamer=streamer,
            use_cache=True
        )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(f"推理结果：{outputs}")


def run_inference(model, tokenizer, image_processor):
    prompt = "What are the things I should be cautious about when I visit here?"
    with torch.no_grad():
        # 对模型进行编译
        model = torch.compile(model, mode="max-autotune")
        image_file_path = "./images/000000039769.jpg"
        model_generate(model, image_file_path, image_processor, prompt, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='IDEA-CCNL/Wenzhong-GPT2-110M', help='模型文件路径')
    parser.add_argument('--model-base', type=str, default=None)
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    if '7b' in model_name:
        model_name = 'checkpoints/llava-v1.5-7b'
    elif "13b" in model_name:
        model_name = 'checkpoints/llava-v1.5-13b'

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                           model_name)
    run_inference(model, tokenizer, image_processor)


if __name__ == '__main__':
    main()
