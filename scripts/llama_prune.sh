prune_ckpt_path='llama3.2_prune'
tune_ckpt_path='llama_0.2'
base_model="/home/rdutt/Llama-3.2-3B"
CUDA_VISIBLE_DEVICES=0

# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python llama3.py --pruning_ratio 0.25 \
#                  --device cuda --eval_device cuda \
#                  --base_model $base_model \
#                  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 26 \
#                  --block_attention_layer_start 4 --block_attention_layer_end 26 \
#                  --save_ckpt_log_name llama3.2_prune \
#                  --pruner_type taylor --taylor param_first \
#                  --max_seq_len 2048 \
#                  --test_after_train --test_before_train --save_model

# echo "[START] - Finish Pruning Model"

micro_batch_size=8  # per_device_bs
batch_size=32
echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=0 python post_training.py \
                            --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
                            --data_path yahma/alpaca-cleaned \
                            --output_dir tune_log/$tune_ckpt_path \
                            --lora_r 8 \
                            --num_epochs 1 \
                            --learning_rate 1e-4 \
                            --micro_batch_size "$micro_batch_size" \
                            --batch_size "$batch_size"

echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.25 --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model
# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
# echo "to use the pruned model"



