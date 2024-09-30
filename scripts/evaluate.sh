#!/bin/bash
export PYTHONPATH='.'

base_model="/home/rdutt/Llama-3.2-3B/" # e.g., decapoda-research/llama-7b-hf
tune_ckpt_name="/home/rdutt/LLM-Pruner/tune_log/llama_0.2/"
prune_ckpt="/home/rdutt/LLM-Pruner/prune_log/llama3.2_prune/"
# epochs=("${@:4}")
epoch=1555

cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin
echo "###### RUNNING #####"
tune_id="${tune_ckpt_name##*/}"
python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,arc_challenge,piqa --device cuda:0 --output_path results/${tune_id}_$epoch.json --no_cache

# for epoch in "${epochs[@]}"; 
# do
#     cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
#     mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin
#     echo "###### RUNNING #####"
#     tune_id="${tune_ckpt_name##*/}"
#     python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_$epoch.json --no_cache
# done