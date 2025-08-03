python getting_user_cf.py \
    --cf_emb_file data/Instruments/item_DIN_MODEL/DIN_MODEL_4000.pt \
    --inters_file data/Instruments/Instruments.inter.json \
    --save_file data/Instruments/item_DIN_MODEL/user_cf_emb.npy \
    --pooling_method attention \
    --attention_decay 0.9