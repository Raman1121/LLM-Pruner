CUDA_VISIBLE_DEVICES=0
micro_batch_size=8  # per_device_bs
batch_size=32       # bs after accumulation
model_path="/home/rdutt/LLM-Pruner/scripts/prune_log/llama3.2_prune/pytorch_model.bin"
learning_rate=1e-4

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python /home/rdutt/LLM-Pruner/post_training.py --prune_model "$model_path" \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate "$learning_rate" \
      --micro_batch_size "$micro_batch_size" \
      --batch_size "$batch_size" \
      --output_dir tune_log/llama3.2_healed \