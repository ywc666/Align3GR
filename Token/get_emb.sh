# 1. get user semantic embedding
python semantic_embedding_user.py \
    --dataset Instruments \ # Beauty Yelp
    --root data \
    --gpu_id 0 \
    --plm_name llama \
    --plm_checkpoint_llama huggyllama/llama-7b \
    --plm_checkpoint_t5 sentence-transformers/sentence-t5-base \
    --max_sent_len 2048 \
    --word_drop_ratio -1

# 2. get item semantic embedding
python semantic_embedding_item.py \
    --dataset Instruments \ # Beauty Yelp   
    --root data \
    --gpu_id 0 \
    --plm_name llama \
    --plm_checkpoint_llama huggyllama/llama-7b \
    --plm_checkpoint_t5 sentence-transformers/sentence-t5-base \
    --max_sent_len 2048 \
    --word_drop_ratio -1

# 3. get user cf embedding
python getting_user_cf.py \
    --cf_emb_file data/Instruments/item_DIN_MODEL/DIN_MODEL_4000.pt \
    --inters_file data/Instruments/Instruments.inter.json \
    --save_file data/Instruments/item_DIN_MODEL/user_cf_emb.npy \
    --pooling_method attention \
    --attention_decay 0.9
