# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

wandb login

DATASET=Instruments # Beauty Yelp
# BASE_MODEL=huggyllama/llama-7b # LLaMA
BASE_MODEL=LLM-Research/llama-2-7b # LLaMA
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/


# step 1 task: seqrec
# step 2 task: 
torchrun --nproc_per_node=8 --master_port=3391  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 128 \
    --learning_rate 0.0002 \
    --epochs 4 \
    --model_max_length 2048 \
    --tasks seqrec,item2index,index2item,user2index,index2user,fusionseqrec,itemsearch,preferenceobtain \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json\
    --wandb_run_name aligngr\
    --temperature 1.0
