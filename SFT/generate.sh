# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET=Instruments # Beauty Yelp
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/res.json
BASE_MODEL=LLM-Research/llama-2-7b # LLaMA

python generate_output.py \
    --ckpt_path ./ckpt/$DATASET/ckpt \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \  # test for 20, generate for 200 
    --test_prompt_ids 0 \
    --index_file .index.json

