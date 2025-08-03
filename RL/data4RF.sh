# The number of processes can only be one for inference

export CUDA_VISIBLE_DEVICES=0

python data4RF.py \
        --review_data_path Align3GR/data/Instruments/Musical_Instruments_5.json \
        --item2id_path Align3GR/data/Instruments/Instruments.item2id \
        --user2id_path Align3GR/data/Instruments/Instruments.user2id \
        --item_title_path Align3GR/data/Instruments/Instruments.item.json \
        --user2SCID_path Align3GR/data/Instruments/Instruments.user.index.json \
        --item2SCID_path Align3GR/data/Instruments/Instruments.item.index.json