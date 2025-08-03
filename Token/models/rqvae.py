import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import wandb
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer



class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0  
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy


        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, labels, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q
    
    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    
    def vq_initialization(self,x, use_sk=True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, emb_idx, dense_out, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF_Loss
        cf_embedding_in_batch = self.cf_embedding[emb_idx]
        cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
        cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)

        total_loss = rqvae_n_diversity_loss + self.alpha * cf_loss

        return total_loss, cf_loss, loss_recon, quant_loss


class Align_RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0  
        ):
        super(Align_RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy

        ef_dim = self.cf_embedding.shape[1]
        # self.SC_encoder = MLPLayers(layers=[self.in_dim + ef_dim, self.in_dim], dropout=self.dropout_prob,bn=self.bn)
        # concat
        self.encode_layer_dims = [self.in_dim + ef_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, labels, emb_idx, use_sk=True):
        # x = self.encoder(x)
        # x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        # out = self.decoder(x_q)

        # return out, rq_loss, indices, x_q
        cf_embedding_in_batch = self.cf_embedding[emb_idx]
        cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(x.device)
        x = self.encoder(torch.cat((x, cf_embedding_in_batch), dim=-1))
        x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q # indices are quantization indices, x_q is quantized embeddings
    
    def vq_initialization(self,x, emb_idx,use_sk=True):
        # self.rq.vq_ini(self.encoder(x))
        # concat
        cf_embedding_in_batch = self.cf_embedding[emb_idx]
        cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(x.device)
        self.rq.vq_ini(self.encoder(torch.cat((x, cf_embedding_in_batch), dim=-1)))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, emb_idx, dense_out, xs=None):

        cf_embedding_in_batch = self.cf_embedding[emb_idx]
        cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(xs.device)
        xs = torch.cat((xs, cf_embedding_in_batch), dim=-1)

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF_Loss
        # cf_embedding_in_batch = self.cf_embedding[emb_idx]
        # cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
        # cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)

        total_loss = rqvae_n_diversity_loss

        return total_loss, 0, loss_recon, quant_loss


class Unified_RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 user_cf_embedding = 0,
                 item_cf_embedding = 0,
                 inter = 0,
                 gamma = 0.0
        ):
        super(Unified_RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.user_cf_embedding = user_cf_embedding
        self.item_cf_embedding = item_cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.inter = inter
        self.N = 60
        self.gamma = gamma
        cf_dim = self.user_cf_embedding.shape[1]

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        # Item
        self.Item_SC_encoder = MLPLayers(layers=[self.in_dim + cf_dim, self.in_dim], dropout=self.dropout_prob,bn=self.bn)

        self.Item_encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout_prob,bn=self.bn)
        self.Item_rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.Item_decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout_prob,bn=self.bn)
        # User
        self.User_SC_encoder = MLPLayers(layers=[self.in_dim + cf_dim, self.in_dim], dropout=self.dropout_prob,bn=self.bn)
        self.User_encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout_prob,bn=self.bn)
        self.User_rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.User_decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout_prob,bn=self.bn)

    def u2i_loss(self,user_embeddings, pos_item_embeddings, user_id, item_id, K, inter):
        """
        Args:
            user_embeddings: (B, d)
            pos_item_embeddings: (B, d)
            user_id: (B,)
            item_id: (B,)
            K: int
            inter: defaultdict(list), user_id -> list of interacted item_ids
        """
        B, d = user_embeddings.shape
        device = user_embeddings.device
        loss = 0.0

        # Used to build mapping from item_id to embedding within batch
        batch_item_map = {}
        for i in range(B):
            batch_item_map[item_id[i].item()] = pos_item_embeddings[i]

        all_batch_item_ids = list(batch_item_map.keys())

        for i in range(B):
            uid = user_id[i].item()
            iid = item_id[i].item()

            u_emb = user_embeddings[i]         # (d,)
            pos_emb = pos_item_embeddings[i]   # (d,)

            # Remove items that the user has already interacted with from current batch
            interacted = set(inter[uid])
            candidate_neg_ids = [j for j in all_batch_item_ids if j not in interacted and j != iid]

            if len(candidate_neg_ids) < K:
                # If not enough K items, allow repeated sampling (can be changed to more robust strategy)
                sampled_neg_ids = random.choices(candidate_neg_ids, k=K)
            else:
                sampled_neg_ids = random.sample(candidate_neg_ids, k=K)

            neg_embs = torch.stack([batch_item_map[j] for j in sampled_neg_ids], dim=0)  # (K, d)

            # Construct logits
            logits_pos = torch.matmul(u_emb, pos_emb)                    # scalar
            logits_neg = torch.matmul(neg_embs, u_emb)                   # (K,)
            logits = torch.cat([logits_pos.unsqueeze(0), logits_neg], dim=0)  # (1 + K,)

            labels = torch.zeros(1, dtype=torch.long, device=device)     # Positive sample at position 0
            loss += torch.nn.functional.cross_entropy(logits.unsqueeze(0), labels)

        return loss / B
    
    def u2i_loss_vectorized(user_emb, pos_item_emb, user_id, item_id, K, inter):
        """
        Vectorized implementation of user-to-item loss with negative sampling.
    
        Args:
            user_emb: User embeddings tensor of shape (B, d)
            pos_item_emb: Positive item embeddings tensor of shape (B, d)
            user_id: User ID tensor of shape (B,)
            item_id: Positive item ID tensor of shape (B,)
            K: Number of negative samples per positive sample
            inter: Dictionary mapping user_id to interacted item_ids
        """
        B, d = user_emb.shape
        device = user_emb.device
    
        # Build intra-batch mapping: item_id -> embedding
        batch_item_map = {iid.item(): emb for iid, emb in zip(item_id, pos_item_emb)}
        all_batch_iids = list(batch_item_map.keys())
    
        # Precompute negative candidate mask: (B, num_batch_items)
        neg_candidate_mask = torch.zeros(B, len(all_batch_iids), dtype=torch.bool, device=device)
        for i in range(B):
            uid = user_id[i].item()
            interacted = set(inter[uid])
            # Mask items not interacted and not the current positive item
            neg_candidate_mask[i] = torch.tensor(
                [j not in interacted and j != item_id[i].item() for j in all_batch_iids], 
                device=device
            )
    
        # Sample negative indices: (valid_B, K)
        neg_indices = []
        for i in range(B):
            candidate_ids = torch.where(neg_candidate_mask[i])[0]
            if len(candidate_ids) < K:
                # Skip if insufficient candidates (avoid noisy samples)
                continue
            neg_indices.append(candidate_ids[torch.randperm(len(candidate_ids))[:K]])
        neg_indices = torch.stack(neg_indices)  # (valid_B, K)
    
        # Get negative embeddings: (valid_B, K, d)
        neg_embs = torch.stack([batch_item_map[all_batch_iids[idx]] for idx in neg_indices], dim=1)
    
        # Normalize embeddings for stable similarity
        u_emb_norm = F.normalize(user_emb, dim=-1)          # (B, d)
        pos_emb_norm = F.normalize(pos_item_emb, dim=-1)    # (B, d)
        neg_emb_norm = F.normalize(neg_embs, dim=-1)        # (valid_B, K, d)
    
        logits_pos = (u_emb_norm * pos_emb_norm).sum(dim=-1)  # (B,)
        logits_neg = torch.einsum('bd,bkd->bk', u_emb_norm, neg_emb_norm)  # (valid_B, K)
    
        # Cross-entropy loss (positive sample at index 0)
        logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)  # (valid_B, 1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
    
        return loss
    

    def forward(self, user_semantic_emb, user_labels, user_cf_emb, item_semantic_emb, item_labels, item_cf_emb, user_id, item_id, use_sk=True):
        # x = self.encoder(x)
        # x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        # out = self.decoder(x_q)

        # return out, rq_loss, indices, x_q

        # item
        SC_item_emb = self.Item_SC_encoder(torch.cat((item_semantic_emb, item_cf_emb), dim=-1))
        x_item_q, rq_loss_item, indices_item = self.Item_rq(SC_item_emb, item_labels, use_sk=use_sk)
        out_item = self.Item_decoder(x_item_q)
        # user
        SC_user_emb = self.User_SC_encoder(torch.cat((user_semantic_emb, user_cf_emb), dim=-1))
        x_user_q, rq_loss_user, indices_user = self.User_rq(SC_user_emb, user_labels, use_sk=use_sk)
        out_user = self.User_decoder(x_user_q)

        # u2i
        u2i_loss = self.u2i_loss(SC_user_emb, SC_item_emb, user_id, item_id, self.N, self.inter)

        return out_item, out_user, rq_loss_item, rq_loss_user, u2i_loss, indices_item, indices_user # indices are quantization indices, x_q is quantized embeddings
    
    def vq_initialization(self,x, emb_idx,use_sk=True,mode='item'):
        # self.rq.vq_ini(self.encoder(x))
        if mode == 'item':
            cf_embedding_in_batch = self.item_cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(x.device)
            SC_emdedding = self.Item_SC_encoder(torch.cat((x, cf_embedding_in_batch), dim=-1))
            self.Item_rq.vq_ini(self.Item_encoder(SC_emdedding))
        elif mode == 'user':
            cf_embedding_in_batch = self.user_cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(x.device)
            SC_emdedding = self.User_SC_encoder(torch.cat((x, cf_embedding_in_batch), dim=-1))
            self.User_rq.vq_ini(self.User_encoder(SC_emdedding))
        else:
            raise ValueError('incompatible mode')

    @torch.no_grad()
    def get_indices(self, user_data, user_labels, user_cf_emb, item_data, item_labels, item_cf_emb, use_sk=False):
        SC_item_emb = self.Item_SC_encoder(torch.cat((item_data, item_cf_emb), dim=-1))
        SC_user_emb = self.User_SC_encoder(torch.cat((user_data, user_cf_emb), dim=-1))
        x_e_item = self.Item_encoder(SC_item_emb)
        x_e_user = self.User_encoder(SC_user_emb)
        _, _, indices_item = self.Item_rq(x_e_item, item_labels, use_sk=use_sk)
        _, _, indices_user = self.User_rq(x_e_user, user_labels, use_sk=use_sk)
        return indices_item, indices_user

    def compute_loss(self, out_item, out_user, item_data, user_data, rq_loss_item, rq_loss_user, u2i_loss):

        if self.loss_type == 'mse':
            item_loss_recon = F.mse_loss(out_item, item_data, reduction='mean')
            user_loss_recon = F.mse_loss(out_user, user_data, reduction='mean')
        elif self.loss_type == 'l1':
            item_loss_recon = F.l1_loss(out_item, item_data, reduction='mean')
            user_loss_recon = F.l1_loss(out_user, user_data, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        item_rqvae_n_diversity_loss = item_loss_recon + self.quant_loss_weight * rq_loss_item
        user_rqvae_n_diversity_loss = user_loss_recon + self.quant_loss_weight * rq_loss_user

        total_loss = self.gamma * (item_rqvae_n_diversity_loss + user_rqvae_n_diversity_loss) + self.alpha * (u2i_loss)

        return total_loss, item_rqvae_n_diversity_loss, user_rqvae_n_diversity_loss,rq_loss_item,rq_loss_user, u2i_loss