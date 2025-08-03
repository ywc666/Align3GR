import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import Unified_EmbDataset
from models.rqvae import Unified_RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="TokenAlign")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')
    parser.add_argument('--inter_path', type=str, default='Instruments.inter.json')
    parser.add_argument('--item_data_path', type=str, default='Instruments.item.json')
    parser.add_argument('--user_data_path', type=str, default='Instruments.user.json')
    parser.add_argument('--item_cf_emb', type=str)
    parser.add_argument('--user_cf_emb', type=str)


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
# ckpt_path = args_setting.root_path + f'alpha{args_setting.alpha}-beta{args_setting.beta}/'+args_setting.checkpoint
ckpt_path = args_setting.checkpoint

output_dir = f"./data/{dataset}/"
user_output_file = f"{dataset}.user_index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
item_output_file = f"{dataset}.item_index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
user_output_file = os.path.join(output_dir,user_output_file)
item_output_file = os.path.join(output_dir,item_output_file)
device = torch.device("cuda:0")


ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"] 
state_dict = ckpt["state_dict"]

item_cf_emb = torch.load(args_setting.item_cf_emb).squeeze().detach().numpy()
user_cf_emb = torch.load(args_setting.user_cf_emb).squeeze().detach().numpy()

# interaction matrix
inter_path = args_setting.inter_path
inter = json.load(open(inter_path, 'r')) 


all_data = Unified_EmbDataset(args_setting.item_data_path,args_setting.user_data_path,inter,user_cf_emb,item_cf_emb)

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

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(all_data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices_user = []
all_indices_str_user = []
all_indices_item = []
all_indices_str_item = []
prefix_user = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]
prefix_item = ["<e_{}>","<f_{}>","<g_{}>","<h_{}>"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

user_labels = {"0":[],"1":[],"2":[], "3":[]}
item_labels = {"0":[],"1":[],"2":[], "3":[]}
user_embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.User_rq.vq_layers]
item_embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.Item_rq.vq_layers]


for idx, emb in enumerate(user_embs):
    centers, label = constrained_km(emb)
    user_labels[str(idx)] = label

for idx, emb in enumerate(item_embs):
    centers, label = constrained_km(emb)
    item_labels[str(idx)] = label

for d in tqdm(data_loader):
    user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb = d[0], d[1], d[2], d[3], d[4], d[5]
    user_data = user_data.to(device)
    user_labels = user_labels.to(device)
    user_cf_emb = user_cf_emb.to(device)
    item_data = item_data.to(device)
    item_labels = item_labels.to(device)
    item_cf_emb = item_cf_emb.to(device)
    
    # indices = model.get_indices(d, use_sk=False)
    indices_item, indices_user = model.get_indices(user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb, use_sk=False)

    indices_item = indices_item.view(-1, indices_item.shape[-1]).cpu().numpy()
    indices_user = indices_user.view(-1, indices_user.shape[-1]).cpu().numpy()

    for index in indices_user:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix_user[i].format(int(ind)))
        all_indices_user.append(code)
        all_indices_str_user.append(str(code))

    for index in indices_item:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix_item[i].format(int(ind)))
        all_indices_item.append(code)
        all_indices_str_item.append(str(code))
    # break

all_indices_user = np.array(all_indices_user)
all_indices_str_user = np.array(all_indices_str_user)
all_indices_item = np.array(all_indices_item)
all_indices_str_item = np.array(all_indices_str_item)


for vq in model.User_rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
if model.User_rq.vq_layers[-1].sk_epsilon == 0.0:
    model.User_rq.vq_layers[-1].sk_epsilon = 0.003

for vq in model.Item_rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
if model.Item_rq.vq_layers[-1].sk_epsilon == 0.0:
    model.Item_rq.vq_layers[-1].sk_epsilon = 0.003

# model.rq.vq_layers[-1].sk_epsilon = 0.1
tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str_user) or check_collision(all_indices_str_item):
        break

    collision_user_groups = get_collision_item(all_indices_str_user)
    collision_item_groups = get_collision_item(all_indices_str_item)
    print(collision_user_groups)
    print(len(collision_user_groups))
    print(collision_item_groups)
    print(len(collision_item_groups))

    for collision_items in collision_user_groups:
        d = all_data[collision_items]
        user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb = d[0], d[1], d[2], d[3], d[4], d[5]
        indices_item, indices_user = model.get_indices(user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb, use_sk=True)

        # indices = model.get_indices(d, use_sk=True)
        indices_item = indices_item.view(-1, indices_item.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices_item):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix_user[i].format(int(ind)))

            all_indices_user[item] = code
            all_indices_str_user[item] = str(code)

    for collision_items in collision_item_groups:
        d = all_data[collision_items]
        user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb = d[0], d[1], d[2], d[3], d[4], d[5]
        indices_item, indices_user = model.get_indices(user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb, use_sk=True)

        indices_item = indices_item.view(-1, indices_item.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices_item):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix_item[i].format(int(ind)))

            all_indices_item[item] = code
            all_indices_str_item[item] = str(code)
    tt += 1


print("All user indices number: ",len(all_indices_user))
print("Max number of user conflicts: ", max(get_indices_count(all_indices_str_user).values()))

print("All item indices number: ",len(all_indices_item))
print("Max number of item conflicts: ", max(get_indices_count(all_indices_str_item).values()))

tot_user_item = len(all_indices_str_user)
tot_user_indice = len(set(all_indices_str_user.tolist()))
print("User Collision Rate",(tot_user_item-tot_user_indice)/tot_user_item)

tot_item_item = len(all_indices_str_item)
tot_item_indice = len(set(all_indices_str_item.tolist()))
print("Item Collision Rate",(tot_item_item-tot_item_indice)/tot_item_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices_user.tolist()):
    all_indices_dict[item] = list(indices)
for item, indices in enumerate(all_indices_item.tolist()):
    all_indices_dict[item] = list(indices)



with open(user_output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)

with open(item_output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)