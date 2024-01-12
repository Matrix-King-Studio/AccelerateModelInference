import argparse
import time

import pandas as pd
import torch
from PIL import Image
from optimum.gptq import GPTQQuantizer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from transformers import TextIteratorStreamer


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def get_quantizer_params(quantization_type, context_len):
    quantizer_params = {
        "gptq-all-8bit": {"bits": 8},
        "gptq-all-8bit-exllama-v1": {"bits": 8, "exllama_config": {"version": 1}},
        "gptq-all-8bit-exllama-v2": {"bits": 8, "exllama_config": {"version": 2}},
        "gptq-all-4bit": {"bits": 4},
        "gptq-all-4bit-exllama-v1": {"bits": 4, "exllama_config": {"version": 1}},
        "gptq-all-4bit-exllama-v2": {"bits": 4, "exllama_config": {"version": 2}},
    }
    params = quantizer_params.get(quantization_type, None)
    if params:
        params["dataset"] = [
            "Is any trash can filled with garbage or not?",
            "Are both the cyclist and passenger in the picture wearing he helmets?",
            "Is someone fighting or engaged in a sparring match or wrestling in this picture?",
            "Answer the question using a single word or phrase."
        ]
        params["model_seqlen"] = context_len
    return params


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
            max_new_tokens=1,
            streamer=streamer,
            use_cache=True
        )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(f"推理结果：{outputs}")


def run_inference(model, tokenizer, image_processor, context_len, quantization_type):
    prompt = "Are both the cyclist and passenger in the picture wearing he helmets?"
    prompt += "\nAnswer the question using a single word or phrase."
    with torch.no_grad():
        if quantization_type == 'fp16':
            model.half()
        elif quantization_type == 'bf16':
            model.bfloat16()
        elif "gptq" in quantization_type:
            if "exllama" in quantization_type:
                model.half()  # exllama kernel for GPTQ requires a float16 input activation
            params = get_quantizer_params(quantization_type, context_len)
            quantizer = GPTQQuantizer(**params)
            model = quantizer.quantize_model(model, tokenizer)
        # 对模型进行编译
        model = torch.compile(model, mode="max-autotune")
        image_file_path = "/data/data1/LLAVA_v1.5_20231011/data/test/test_longtail/helmet/helmet_pos/"
        for _ in range(10):  # Warmup
            model_generate(model, image_file_path, image_processor, prompt, tokenizer)
        print("\tWarmup finished.")
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(20):  # 实际推理
            model_generate(model, image_file_path, image_processor, prompt, tokenizer)
        torch.cuda.synchronize()
        inference_time = (time.time() - start_time) * 1000 / 20  # ms
        memory_usage = torch.cuda.memory_allocated() / 1024 ** 3  # G
        return inference_time, memory_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="/data/data1/LLAVA_v1.5_20231011/code/LLaVA-main/checkpoints/llava-v1.5-13b '", help='模型文件路径')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--output-csv', type=str, default='quantize_result.csv', help='输出CSV文件路径')
    args = parser.parse_args()

    disable_torch_init()
    results = []
    for quantization_type in [
        'Baseline', 'fp16', 'bf16',
        "gptq-all-8bit", "gptq-all-4bit", "gptq-all-4bit-exllama-v1", "gptq-all-4bit-exllama-v2",
    ]:
        print(f"正在测试 {quantization_type} 量化类型...")
        # 每次重新创建模型
        model_name = get_model_name_from_path(args.model_path)
        if '7b' in model_name:
            model_name = 'checkpoints/llava-v1.5-7b'
        elif "13b" in model_name:
            model_name = 'checkpoints/llava-v1.5-13b'

        tokenizer, model, image_processor, cxt_len = load_pretrained_model(args.model_path, args.model_base,
                                                                           model_name)

        torch.cuda.empty_cache()  # 清理 GPU 缓存
        # 开始测试推理时间
        inference_time, memory_usage = run_inference(model, tokenizer, image_processor, cxt_len, quantization_type)
        results.append({'量化类型': quantization_type, '推理时间(ms)': inference_time, '显存占用(G)': memory_usage})
        torch.cuda.empty_cache()  # 清理 GPU 缓存

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
