#!/bin/bash


# dislike stage

mkdir -p Align3GR/RL/Instruments_dislike/output

torchrun --nproc_per_node=8 --master_port=3391  softmax_DPO.py \
    --output_dir="Align3GR/RL/Instruments_dislike/output" \
    --logging_dir="Align3GR/RL/Instruments_dislike/output/logs" \
    --model_name="LLM-Research/llama-2-7b" \
    --dataset="instruments" \
    --resume_from_checkpoint="Align3GR/RL/Instruments_hard/output/final_checkpoint" \
    --wandb_project="RF-DPO" \
    --wandb_name="rf-dpo-instruments-dislike" \
    --beta=1.0 \
    --neg_num=20 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --cutoff_len=512 \
    --eval_step=0.1 \
    --data_train="Align3GR/RL/sample_data/Instrument_dislike/data_dislike_train.jsonl" \
    --data_val="Align3GR/RL/sample_data/Instrument_dislike/data_dislike_val.jsonl"

echo "dislike done!" 



# neutral stage
mkdir -p Align3GR/RL/Instruments_neutral/output


torchrun --nproc_per_node=8 --master_port=3391  softmax_DPO.py \
    --output_dir="Align3GR/RL/Instruments_neutral/output" \
    --logging_dir="Align3GR/RL/Instruments_neutral/output/logs" \
    --model_name="LLM-Research/llama-2-7b" \
    --dataset="instruments" \
    --resume_from_checkpoint="Align3GR/RL/Instruments_dislike/output/final_checkpoint" \
    --wandb_project="RF-DPO" \
    --wandb_name="rf-dpo-instruments-neutral" \
    --beta=1.0 \
    --neg_num=20 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --cutoff_len=512 \
    --eval_step=0.1 \
    --data_train="Align3GR/RL/sample_data/Instrument_neutral/data_neutral_train.jsonl" \
    --data_val="Align3GR/RL/sample_data/Instrument_neutral/data_neutral_val.jsonl"

echo "neutral done!" 


