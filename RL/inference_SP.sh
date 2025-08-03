# The number of processes can only be one for inference

export CUDA_VISIBLE_DEVICES=0

python inference.py \
        --dataset Instruments \ # Beauty Yelp
        --batch_size 32 \
        --base_model BASE_MODEL \
        --resume_from_checkpoint CKPT \
        --results_file RESULTS_FILE \
        --num_beams 200 \
        --neg_num 20 \
        --save_predictions Align3GR/RL/Instruments_hard/output/predictions.jsonl