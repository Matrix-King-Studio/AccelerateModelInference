lm_eval --model hf \
    --model_args pretrained=IDEA-CCNL/Wenzhong-GPT2-110M \
    --tasks hellaswag,piqa \
    --device cpu \
    --batch_size auto \
    --output_path ./output \
    --log_samples
