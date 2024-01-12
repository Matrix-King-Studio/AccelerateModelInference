import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from test_utils import calculate_average_inference_time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hf_model_path = "IDEA-CCNL/Wenzhong-GPT2-110M"
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
model.to(device)

question = "北京是中国的"
inputs = tokenizer(question, return_tensors="pt").to(device)

# 测试基线
print(f"Begin inference time test: baseline")
baseline_avg_inference_time = calculate_average_inference_time(model, inputs)
print(f"Baseline average inference time: {baseline_avg_inference_time} ms")

# 测试 torch.compile
# print(f"Begin inference time test: torch.compile")
# model = torch.compile(model)
# torch_compile_avg_inference_time = calculate_average_inference_time(model, inputs)
# print(f"torch.compile average inference time: {torch_compile_avg_inference_time} ms")

# 测试 bf16
print(f"Begin inference time test: bf16")
model = GPT2LMHeadModel.from_pretrained(hf_model_path, cache_dir="./checkpoints", torch_dtype=torch.bfloat16)
model.to(device)
bf16_avg_inference_time = calculate_average_inference_time(model, inputs)
print(f"bf16 average inference time: {bf16_avg_inference_time} ms")

# 测试 SDPA Flash
print(f"Begin inference time test: sdpa_flash")
sdpa_flash_avg_inference_time = calculate_average_inference_time(model, inputs, sdpa_flash=True)
print(f"SDPA-Flash average inference time: {sdpa_flash_avg_inference_time} ms")

# 测试 SDPA Math
print(f"Begin inference time test: sdpa_math")
sdpa_math_avg_inference_time = calculate_average_inference_time(model, inputs, sdpa_math=True)
print(f"SDPA-Math average inference time: {sdpa_math_avg_inference_time} ms")

# 测试 SDPA Mem Efficient
print(f"Begin inference time test: sdpa_mem_efficient")
sdpa_mem_efficient_avg_inference_time = calculate_average_inference_time(model, inputs, sdpa_mem_efficient=True)
print(f"SDPA-Mem-Efficient average inference time: {sdpa_mem_efficient_avg_inference_time} ms")
