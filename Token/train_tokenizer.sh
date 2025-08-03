wandb login

python ./Token/main.py \
  --device cuda:0 \
  --item_data_path data/Instruments/item_embeddings_t5-base.npy \
  --user_data_path data/Instruments/user_embeddings_t5-base.npy \
  --alpha 0.01 \
  --beta 0.0001 \
  --item_cf_emb data/Instruments/item_DIN_MODEL/DIN_MODEL_4000.pt \
  --user_cf_emb data/Instruments/item_DIN_MODEL/user_cf_emb.npy \
  --ckpt_dir ckpt_unified