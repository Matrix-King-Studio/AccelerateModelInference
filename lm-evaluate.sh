lm_eval --model hf \
    --model_args pretrained=IDEA-CCNL/Wenzhong-GPT2-110M,trust_remote_code=True \
    --tasks arc_challenge,hellaswag,piqa \
    --device cpu \
    --batch_size auto \
    --output_path ./lm-eval-output \
    --log_samples
