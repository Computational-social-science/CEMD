#!/bin/bash

if [ "$#" -ne 10 ]; then
    echo "Use: bash $0 <dataset_name> <test_size> <train_ratio> <sft_type> <lora_rank> <learning_rate> <with_or_without_info> <data_version> <num_epochs> <device> "
    exit 1
fi

dataset_name=$1
test_size=$2
train_ratio=$3
sft_type=$4
lora_rank=$5
learning_rate=$6
with_or_without_info=$7
data_version=$8
num_epochs=$9
device=${10}


if [ "$dataset_name" == "liar2" ]; then
    split_type="8:1:1"
else
    split_type=$(echo "10 - $test_size * 10" | bc | awk '{print int($1)}'):$(echo "$test_size * 10" | bc | awk '{print int($1)}')
fi

custom_train_dataset_path=my_data/$dataset_name/$with_or_without_info/train_valid_split/$split_type/subtrain_data$data_version/train_data_$train_ratio.jsonl

output_name="lr=$learning_rate-$(date +"%Y%m%d-%H:%M:%S")"
if [ "$train_ratio" = "1" ] || [ -z "$train_ratio" ]; then
    train_ratio=1.0
    custom_train_dataset_path=my_data/$dataset_name/$with_or_without_info/train_valid_split/$split_type/train_data$data_version.jsonl
fi

lora_alpha=$(expr $lora_rank \* 4)

max_length=8192

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=$device \
python llm_sft.py \
    --model_type meta-llama-3-8B-instruct \
    --model_id_or_path /home/css/models/Meta-Llama-3-8B-Instruct \
    --check_model_is_latest false \
    --sft_type $sft_type \
    --tuner_backend peft \
    --template_type _llama3 \
    --dtype AUTO \
    --add_output_dir_suffix false \
    --output_dir output/$dataset_name/Llama-3-8B-Instruct/$with_or_without_info/data$data_version-split=$split_type-ratio=$train_ratio/dora-r=$lora_rank/"$output_name" \
    --dataset $custom_train_dataset_path#-1 \
    --dataset_test_ratio 0 \
    --num_train_epochs $num_epochs \
    --max_length $max_length \
    --max_new_tokens 512 \
    --check_dataset_strategy warning \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --use_dora true \
    --lora_dtype AUTO \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --use_flash_attn false \
    --full_determinism true \
    --do_sample false

