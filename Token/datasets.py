import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)



class Unified_EmbDataset(data.Dataset):

    def __init__(self,item_data_path,user_data_path,inter_data,user_cf_emb,item_cf_emb):

        self.item_data_path = item_data_path
        self.user_data_path = user_data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.item_embeddings = np.load(item_data_path)
        self.user_embeddings = np.load(user_data_path)
        self.item_dim = self.item_embeddings.shape[-1]
        self.user_dim = self.user_embeddings.shape[-1]
        self.inter = []
        self.user_cf_emb = user_cf_emb
        self.item_cf_emb = item_cf_emb
        for user,value in inter_data.items():
            for item in value:
                self.inter.append((int(user),int(item)))

    def __getitem__(self, index):
        user_id,item_id = self.inter[index] 
        item_emb = self.item_embeddings[item_id]
        user_emb = self.user_embeddings[user_id]
        item_cf_emb = self.item_cf_emb[item_id]
        user_cf_emb = self.user_cf_emb[user_id]
        tensor_item_emb=torch.FloatTensor(item_emb)
        tensor_user_emb=torch.FloatTensor(user_emb)
        tensor_item_cf_emb=torch.FloatTensor(item_cf_emb)
        tensor_user_cf_emb=torch.FloatTensor(user_cf_emb)
        return tensor_item_emb, tensor_user_emb, tensor_item_cf_emb, tensor_user_cf_emb, user_id, item_id

    def __len__(self):
        return len(self.inter)