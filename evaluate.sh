lm_eval --model hf \
    --model_args pretrained=gpt2,trust_remote_code=True \
    --tasks arc_challenge,hellaswag,piqa \
    --device cpu \
    --batch_size auto \
    --output_path ./output \
    --log_samples
