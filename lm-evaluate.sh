lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen-7B,trust_remote_code=True \
    --tasks arc_challenge,hellaswag,piqa \
    --device cpu \
    --batch_size auto \
    --output_path ./lm-eval-output
