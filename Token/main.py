import argparse
import json
import random
import torch
import numpy as np
from time import time
import logging
import wandb
from torch.utils.data import DataLoader
import socket
from datasets import EmbDataset, Unified_EmbDataset
from models.rqvae import Unified_RQVAE
from trainer import  Unified_Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
    parser = argparse.ArgumentParser(description="Token")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=32, )
    parser.add_argument('--eval_step', type=int, default=100, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--item_data_path", type=str, default="Instruments/item_embeddings_t5-base.npy", help="Input data path.")
    parser.add_argument("--user_data_path", type=str, default="Instruments/user_embeddings_t5-base.npy", help="Input data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:2", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.0001, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--item_cf_emb', type=str, default="Instruments/item_DIN_MODEL/DIN_MODEL_4000.pt", help='cf emb')
    parser.add_argument('--user_cf_emb', type=str, default="Instruments/item_DIN_MODEL/user_cf_emb.npy", help='cf emb')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="Token/ckpt_unified", help="output directory for model")

    parser.add_argument("--inter_path", type=str, default= 'data/Instruments/Instruments.inter.json', help="inter matrix")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    wandb.init( project='Token',
                entity='',
                notes=socket.gethostname(),
                name='Token',
                dir='./checkpoint',
                job_type="training",
                mode="disabled",    
                reinit=True)

    print(args)
    logging.basicConfig(level=logging.DEBUG)
    item_cf_emb = torch.load(args.item_cf_emb).squeeze().detach().numpy()
    user_cf_emb = torch.load(args.user_cf_emb).squeeze().detach().numpy()

    # inter matrix
    inter = json.load(open(args.inter_path, 'r')) 

    """build dataset"""
    all_data = Unified_EmbDataset(args.item_data_path,args.user_data_path,inter,user_cf_emb,item_cf_emb)
    
    model = Unified_RQVAE(in_dim=all_data.user_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  user_cf_embedding = user_cf_emb,
                  item_cf_embedding = item_cf_emb,
                  inter = inter
                  )
    print(model)
    all_data_loader = DataLoader(all_data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)


    trainer = Unified_Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(all_data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)




