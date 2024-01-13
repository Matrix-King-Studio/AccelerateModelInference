import torch as th
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
model = GPT2LMHeadModel.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M").to(device)

inputs = tokenizer("北京是中国的", return_tensors="pt").to(device)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=th.profiler.schedule(skip_first=1, wait=1, warmup=1, active=2),
             with_stack=True, profile_memory=True, record_shapes=True) as prof:
    for i in range(5):
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        print(f"{i + 1} outputs: {tokenizer.decode(outputs[0])}")
        prof.step()

print(" cpu_time_total ".center(50, "-"))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(" cuda_time_total ".center(50, "-"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(" self_cpu_memory_usage ".center(50, "-"))
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

prof.export_chrome_trace("trace.json")
