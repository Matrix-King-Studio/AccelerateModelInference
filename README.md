# AccelerateModelInference

统一命名一下吧，优化前的结果文件，后缀添加 "_before"；优化后的结果文件，后缀添加"_after"。

## llama.cpp

```shell
cd /tmp && git clone https://huggingface.co/Qwen/Qwen-7B
python convert-hf-to-gguf.py /tmp/Qwen-7B/
./quantize /tmp/Qwen-7B/ggml-model-f16.gguf qwen7b-q4_0.gguf q4_0
./main -m qwen7b-q4_0.gguf  -n 512 --color -i -cml -f prompts/chat-with-qwen.txt
```
