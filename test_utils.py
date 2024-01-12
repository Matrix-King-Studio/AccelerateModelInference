import time

import torch

from torch.profiler import profile, record_function, ProfilerActivity


def sdpa_state():
    sdpa_flash_state = torch.backends.cuda.flash_sdp_enabled()
    sdpa_math_state = torch.backends.cuda.math_sdp_enabled()
    sdpa_mem_efficient_state = torch.backends.cuda.mem_efficient_sdp_enabled()
    print(f"SDPA-Flash: {sdpa_flash_state}, ", end="")
    print(f"SDPA-Math: {sdpa_math_state}, ", end="")
    print(f"SDPA-Mem-Efficient: {sdpa_mem_efficient_state}")


def model_inference(model, inputs, record=False):
    if record:
        with profile(activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ], with_stack=True, profile_memory=True, record_shapes=True) as prof:
            with record_function("model_inference"):
                model.generate(**inputs, max_new_tokens=100, do_sample=False)

        prof.export_chrome_trace("trace.json")
        prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
    else:
        model.generate(**inputs, max_new_tokens=100, do_sample=False)


def calculate_average_inference_time(
        model,
        inputs,
        iteration=9,
        sdpa_flash=False,
        sdpa_math=False,
        sdpa_mem_efficient=False
):
    print("Begin warmup")
    for _ in range(10):
        model(**inputs)
    print("Warmup finished")

    start_time = time.time()
    for i in range(iteration):
        if sdpa_flash or sdpa_math or sdpa_mem_efficient:
            if not i:  # 在第一次迭代时打印 SDPA 状态
                sdpa_state()
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=sdpa_flash,
                    enable_math=sdpa_math,
                    enable_mem_efficient=sdpa_mem_efficient
            ):
                model_inference(model, inputs, record=i == iteration - 1)
        else:
            model_inference(model, inputs, record=i == iteration - 1)
        if i and not (i + 1) % 10:
            print(f"Finished inference {i + 1}")
    end_time = time.time()

    average_inference_time = (end_time - start_time) / 20 * 1000
    return average_inference_time
