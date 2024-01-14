# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen-7B,trust_remote_code=True,load_in_8bit=True \
#     --tasks arc_challenge,hellaswag,piqa \
#     --device cpu \
#     --batch_size 16 \
#     --output_path ./lm-eval-output

lm_eval --model Llama.cpp --model_args pretrained=/tmp/Qwen-7B-gguf/qwen7b-q4_0.gguf --tasks arc_challenge,hellaswag,piqa --device cpu --batch_size 16 --output_path ./lm-eval-output
