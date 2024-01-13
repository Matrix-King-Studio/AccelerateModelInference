lm_eval --model hf \
    --model_args pretrained=/tmp/Qwen-7B/,trust_remote_code=True \
    --tasks arc_challenge,hellaswag,piqa \
    --device cpu \
    --batch_size 64 \
    --output_path ./lm-eval-output
