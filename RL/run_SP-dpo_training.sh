#!/bin/bash

mkdir -p Align3GR/RL/Instruments_easy/output

# easy stage

torchrun --nproc_per_node=8 --master_port=3391  softmax_DPO.py \
    --output_dir="Align3GR/RL/Instruments_easy/output" \
    --logging_dir="Align3GR/RL/Instruments_easy/output/logs" \
    --model_name="LLM-Research/llama-2-7b" \
    --dataset="instruments" \
    --resume_from_checkpoint="Align3GR/RL/Instruments_SFT/output/final_checkpoint" \
    --wandb_project="RL" \
    --wandb_name="sp-dpo-instruments-easy" \
    --beta=1.0 \
    --neg_num=20 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --cutoff_len=512 \
    --eval_step=0.1 \
    --data_train="Align3GR/RL/sample_data/Instrument_easy/data_easy_train.jsonl" \
    --data_val="Align3GR/RL/sample_data/Instrument_easy/data_easy_val.jsonl"

echo "easy done!"

# medium stage

mkdir -p Align3GR/RL/Instruments_medium/output

torchrun --nproc_per_node=8 --master_port=3391  softmax_DPO.py \
    --output_dir="Align3GR/RL/Instruments_medium/output" \
    --logging_dir="Align3GR/RL/Instruments_medium/output/logs" \
    --model_name="LLM-Research/llama-2-7b" \
    --dataset="instruments" \
    --resume_from_checkpoint="Align3GR/RL/Instruments_easy/output/final_checkpoint" \
    --wandb_project="RL" \
    --wandb_name="sp-dpo-instruments-medium" \
    --beta=1.0 \
    --neg_num=20 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --cutoff_len=512 \
    --eval_step=0.1 \
    --data_train="Align3GR/RL/sample_data/Instrument_medium/data_medium_train.jsonl" \
    --data_val="Align3GR/RL/sample_data/Instrument_medium/data_medium_val.jsonl"

echo "medium done!"

# hard stage

mkdir -p Align3GR/RL/Instruments_hard/output

torchrun --nproc_per_node=8 --master_port=3391  softmax_DPO.py \
    --output_dir="Align3GR/RL/Instruments_hard/output" \
    --logging_dir="Align3GR/RL/Instruments_hard/output/logs" \
    --model_name="LLM-Research/llama-2-7b" \
    --dataset="instruments" \
    --resume_from_checkpoint="Align3GR/RL/Instruments_medium/output/final_checkpoint" \
    --wandb_project="RL" \
    --wandb_name="sp-dpo-instruments-hard" \
    --beta=1.0 \
    --neg_num=20 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --cutoff_len=512 \
    --eval_step=0.1 \
    --data_train="Align3GR/RL/sample_data/Instrument_hard/data_hard_train.jsonl" \
    --data_val="Align3GR/RL/sample_data/Instrument_hard/data_hard_val.jsonl"

echo "hard done!"