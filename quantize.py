import argparse
import pandas as pd
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from optimum.gptq import GPTQQuantizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_quantizer_params(quantization_type):
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
        params["dataset"] = ["c4"]
        params["model_seqlen"] = 1024
    return params


def run_inference(model, tokenizer, quantization_type, question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        if quantization_type == 'fp16':
            model.half()
        elif quantization_type == 'bf16':
            model.bfloat16()
        elif "gptq" in quantization_type:
            if "exllama" in quantization_type:
                model.half()    # exllama kernel for GPTQ requires a float16 input activation
            params = get_quantizer_params(quantization_type)
            quantizer = GPTQQuantizer(**params)
            model = quantizer.quantize_model(model, tokenizer)
        for _ in range(10):  # 预热
            model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(20):  # 实际推理
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            print(f"\t第 {i + 1} 次推理结果：{tokenizer.decode(output[0])}")
        torch.cuda.synchronize()
    inference_time = (time.time() - start_time) * 1000 / 20  # ms
    memory_usage = torch.cuda.memory_allocated() / 1024 ** 3  # G
    return inference_time, memory_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='IDEA-CCNL/Wenzhong-GPT2-110M', help='模型文件路径')
    parser.add_argument('--question', type=str, default="北京是中国的", help='问题')
    parser.add_argument('--output_csv', type=str, default='quantize_result.csv', help='输出CSV文件路径')
    args = parser.parse_args()

    results = []
    for quantization_type in [
        # 'Baseline', 'fp16', 'bf16',
        "gptq-all-8bit", "gptq-all-4bit", "gptq-all-4bit-exllama-v1", "gptq-all-4bit-exllama-v2",
    ]:
        print(f"正在测试 {quantization_type} 量化类型...")
        # 每次重新创建模型
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.to(device)

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        inference_time, memory_usage = run_inference(model, tokenizer, quantization_type, args.question)
        results.append({'量化类型': quantization_type, '推理时间(ms)': inference_time, '显存占用(G)': memory_usage})

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
