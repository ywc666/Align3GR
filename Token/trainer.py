import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
import wandb
from datasets import EmbDataset
from torch.utils.data import DataLoader

def calculate_codebook_perplexity(indices, num_embeddings):
    """
    Calculate perplexity of codebook indices
    
    Args:
        indices: codebook indices [batch_size, num_layers] or [batch_size]
        num_embeddings: number of embeddings in each codebook
    
    Returns:
        perplexity: perplexity of codebook indices
    """
    if indices.dim() == 1:
        # Single codebook case
        indices = indices.unsqueeze(-1)
    
    batch_size, num_layers = indices.shape
    total_perplexity = 0.0
    
    for layer_idx in range(num_layers):
        layer_indices = indices[:, layer_idx]
        
        # Calculate occurrence count of each index
        index_counts = torch.zeros(num_embeddings, device=indices.device)
        unique_indices, counts = torch.unique(layer_indices, return_counts=True)
        index_counts[unique_indices] = counts.float()
        
        # Calculate probability distribution
        probabilities = index_counts / batch_size
        
        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        
        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy)
        total_perplexity += perplexity.item()
    
    # Return average perplexity
    return total_perplexity / num_layers

def calculate_vae_perplexity(model_output, target_data, loss_type='mse', method='reconstruction'):
    """
    Calculate perplexity of VAE model
    
    Args:
        model_output: model output [batch_size, feature_dim]
        target_data: target data [batch_size, feature_dim]
        loss_type: loss type ('mse', 'l1')
        method: perplexity calculation method ('reconstruction', 'log_likelihood', 'cross_entropy')
    
    Returns:
        perplexity: perplexity value
    """
    if method == 'reconstruction':
        # Perplexity based on reconstruction loss
        if loss_type == 'mse':
            recon_loss = F.mse_loss(model_output, target_data, reduction='mean')
        elif loss_type == 'l1':
            recon_loss = F.l1_loss(model_output, target_data, reduction='mean')
        else:
            recon_loss = F.mse_loss(model_output, target_data, reduction='mean')
        
        # Perplexity = exp(reconstruction loss)
        perplexity = torch.exp(recon_loss)
        
    elif method == 'log_likelihood':
        # Perplexity based on log likelihood (assuming Gaussian distribution)
        if loss_type == 'mse':
            # MSE loss corresponds to log likelihood of Gaussian distribution
            log_likelihood = -F.mse_loss(model_output, target_data, reduction='mean')
        else:
            log_likelihood = -F.l1_loss(model_output, target_data, reduction='mean')
        
        # Perplexity = exp(-log_likelihood)
        perplexity = torch.exp(-log_likelihood)
        
    elif method == 'cross_entropy':
        # Perplexity based on cross entropy
        if loss_type == 'mse':
            mse_loss = F.mse_loss(model_output, target_data, reduction='none')
            # Calculate loss for each sample
            sample_losses = mse_loss.mean(dim=-1)  # [batch_size]
            # Calculate "entropy"
            entropy = -torch.log(torch.clamp(sample_losses, min=1e-8)).mean()
            perplexity = torch.exp(-entropy)
        else:
            l1_loss = F.l1_loss(model_output, target_data, reduction='none')
            sample_losses = l1_loss.mean(dim=-1)
            entropy = -torch.log(torch.clamp(sample_losses, min=1e-8)).mean()
            perplexity = torch.exp(-entropy)
    else:
        raise ValueError(f"Unknown perplexity method: {method}")
    
    return perplexity

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}


    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def calculate_perplexity(self, indices, num_embeddings):
        """
        calculate the perplexity of the codebook indices
        
        Args:
            indices: codebook indices [batch_size, num_layers]
            num_embeddings: number of embeddings in each codebook
        
        Returns:
            perplexity: perplexity of the codebook indices
        """
        return calculate_codebook_perplexity(indices, num_embeddings)

    def calculate_codebook_utilization(self, indices, num_embeddings):
        """
        calculate the utilization of the codebook (based on the index entropy)
        
        Args:
            indices: codebook indices [batch_size, num_layers] or [batch_size]
            num_embeddings: number of embeddings in each codebook
        
        Returns:
            utilization: utilization of the codebook (entropy, the larger the better)
        """
        if indices.dim() == 1:
            # single codebook case
            indices = indices.unsqueeze(-1)
        
        batch_size, num_layers = indices.shape
        total_entropy = 0.0
        
        for layer_idx in range(num_layers):
            layer_indices = indices[:, layer_idx]
            
            # calculate the number of occurrences of each index
            index_counts = torch.zeros(num_embeddings, device=indices.device)
            unique_indices, counts = torch.unique(layer_indices, return_counts=True)
            index_counts[unique_indices] = counts.float()
            
            # calculate the probability distribution
            probabilities = index_counts / batch_size
            
            # calculate the entropy (avoid log(0))
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            total_entropy += entropy.item()
        
        # return the average entropy
        return total_entropy / num_layers

    def constrained_km(self, data, n_clusters=10):
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
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data)    

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out = self.model(data, self.labels)

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, emb_idx, dense_out, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_cf_loss, quant_loss.item()

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        total_codebook_perplexity = 0.0
        total_reconstruction_loss = 0.0
        total_codebook_utilization = 0.0
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            
            # get the model output
            out, rq_loss, indices, dense_out = self.model(data, self.labels)
            
            # Calculate reconstruction loss
            if self.model.loss_type == 'mse':
                recon_loss = F.mse_loss(out, data, reduction='mean')
            elif self.model.loss_type == 'l1':
                recon_loss = F.l1_loss(out, data, reduction='mean')
            else:
                recon_loss = F.mse_loss(out, data, reduction='mean')
            
            # Calculate perplexity of codebook indices
            batch_indices = indices.view(-1, indices.shape[-1])  # [batch_size, num_layers]
            num_embeddings = self.model.rq.vq_layers[0].embedding.num_embeddings
            batch_perplexity = self.calculate_perplexity(batch_indices, num_embeddings)
            
            total_codebook_perplexity += batch_perplexity
            total_reconstruction_loss += recon_loss.item()
            
            # Calculate codebook utilization (entropy)
            batch_utilization = self.calculate_codebook_utilization(batch_indices, num_embeddings)
            total_codebook_utilization += batch_utilization
            
            # Get indices for collision rate calculation
            indices_cpu = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices_cpu:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
            
        collision_rate = (num_sample - len(indices_set))/num_sample
        avg_codebook_perplexity = total_codebook_perplexity / len(valid_data)
        avg_reconstruction_loss = total_reconstruction_loss / len(valid_data)
        avg_codebook_utilization = total_codebook_utilization / len(valid_data)
        


        return collision_rate, avg_codebook_perplexity, avg_reconstruction_loss

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cf_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        wandb.log({"train/loss": loss, "train/reconstruction_loss": recon_loss, "train/cf_loss": cf_loss})
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, cf_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, avg_codebook_perplexity, avg_reconstruction_loss = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                # if cur_eval_step >= 10:
                #     print("Finish!")
                #     break

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("codebook_perplexity", "blue")
                    + ": %.4f, "
                    + set_color("reconstruction_loss", "blue")
                    + ": %.4f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate, avg_codebook_perplexity, avg_reconstruction_loss)
                wandb.log({
                    "val/collision_rate": collision_rate, 
                    "val/codebook_perplexity": avg_codebook_perplexity, 
                    "val/reconstruction_loss": avg_reconstruction_loss
                })
                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate




class Unified_Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.user_labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.item_labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}


    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def calculate_perplexity(self, indices, num_embeddings):
        """
        calculate the perplexity of the codebook indices
        
        Args:
            indices: codebook indices [batch_size, num_layers]
            num_embeddings: number of embeddings in each codebook
        
        Returns:
            perplexity: perplexity of the codebook indices
        """
        return calculate_codebook_perplexity(indices, num_embeddings)

    def calculate_codebook_utilization(self, indices, num_embeddings):
        """
        calculate the utilization of the codebook (based on the index entropy)
        
        Args:
            indices: codebook indices [batch_size, num_layers] or [batch_size]
            num_embeddings: number of embeddings in each codebook
        
        Returns:
            utilization: utilization of the codebook (entropy, the larger the better)
        """
        if indices.dim() == 1:
            # single codebook case
            indices = indices.unsqueeze(-1)
        
        batch_size, num_layers = indices.shape
        total_entropy = 0.0
        
        for layer_idx in range(num_layers):
            layer_indices = indices[:, layer_idx]
            
            # calculate the number of occurrences of each index
            index_counts = torch.zeros(num_embeddings, device=indices.device)
            unique_indices, counts = torch.unique(layer_indices, return_counts=True)
            index_counts[unique_indices] = counts.float()
            
            # calculate the probability distribution
            probabilities = index_counts / batch_size
            
            # calculate the entropy (avoid log(0))
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            total_entropy += entropy.item()
        
        # return the average entropy
        return total_entropy / num_layers

    def constrained_km(self, data, n_clusters=10):
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
    
    def vq_init(self):
        self.model.eval()
        # item
        item_data = EmbDataset(self.args.item_data_path)
        item_loader = DataLoader(item_data,num_workers=self.args.num_workers,
                             batch_size=len(item_data), shuffle=True,
                             pin_memory=True)
        print('len(item_data):',len(item_data))
        iter_item_data = tqdm(
                    item_loader,
                    total=len(item_data),
                    ncols=100,
                    desc=set_color(f"Initialization of item vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_item_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data, emb_idx,mode='item')

        # user
        user_data = EmbDataset(self.args.user_data_path)
        user_loader = DataLoader(user_data,num_workers=self.args.num_workers,
                             batch_size=len(user_data), shuffle=True,
                             pin_memory=True)
        print('len(user_data):',len(user_data))
        iter_user_data = tqdm(
                    user_loader,
                    total=len(user_data),
                    ncols=100,
                    desc=set_color(f"Initialization of user vq","pink"),
                    )
        for batch_idx, data in enumerate(iter_user_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.model.vq_initialization(data, emb_idx,mode='user')


    def _train_epoch(self, all_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_item_recon_loss = 0
        total_user_recon_loss = 0
        total_u2i_loss = 0
        total_rq_loss_item = 0
        total_rq_loss_user = 0

        iter_data = tqdm(
                    all_data,
                    total=len(all_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        embs_user  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.User_rq.vq_layers]
        embs_item  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.Item_rq.vq_layers]

        for idx, emb in enumerate(embs_user):
            centers, labels = self.constrained_km(emb)
            self.user_labels[str(idx)] = labels
        for idx, emb in enumerate(embs_item):
            centers, labels = self.constrained_km(emb)
            self.item_labels[str(idx)] = labels


        for batch_idx, data in enumerate(iter_data):
            item_data, user_data, item_cf_emb, user_cf_emb, user_id, item_id = data[0], data[1], data[2], data[3], data[4], data[5] # 这是一组正样本
            item_data = item_data.to(self.device)
            user_data = user_data.to(self.device)
            item_cf_emb = item_cf_emb.to(self.device)
            user_cf_emb = user_cf_emb.to(self.device)
            self.optimizer.zero_grad()
            out_item, out_user, rq_loss_item, rq_loss_user, u2i_loss,indices_item, indices_user = self.model(user_data, self.user_labels,user_cf_emb, item_data, self.item_labels,item_cf_emb,user_id,item_id)

            total_loss, item_rqvae_n_diversity_loss, user_rqvae_n_diversity_loss, rq_loss_item, rq_loss_user, u2i_loss = self.model.compute_loss(out_item, out_user,item_data, user_data, rq_loss_item, rq_loss_user, u2i_loss)
            self._check_nan(total_loss)
            total_loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += total_loss.item()
            total_item_recon_loss += item_rqvae_n_diversity_loss.item()
            total_rq_loss_item += rq_loss_item.item()
            total_rq_loss_user += rq_loss_user.item()
            total_user_recon_loss += user_rqvae_n_diversity_loss.item()
            total_u2i_loss += u2i_loss.item()

        return total_loss, total_item_recon_loss, total_user_recon_loss, total_u2i_loss, total_rq_loss_item, total_rq_loss_user

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        item_indices_set = set()
        user_indices_set = set()

        num_sample = 0
        total_user_codebook_perplexity = 0.0
        total_user_reconstruction_loss = 0.0
        total_user_codebook_utilization = 0.0
        total_item_codebook_perplexity = 0.0
        total_item_reconstruction_loss = 0.0
        total_item_codebook_utilization = 0.0


        embs_user  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.User_rq.vq_layers]
        embs_item  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.Item_rq.vq_layers]

        for idx, emb in enumerate(embs_user):
            centers, labels = self.constrained_km(emb)
            self.user_labels[str(idx)] = labels
        for idx, emb in enumerate(embs_item):
            centers, labels = self.constrained_km(emb)
            self.item_labels[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            item_data, user_data, item_cf_emb, user_cf_emb, user_id, item_id = data[0], data[1], data[2], data[3], data[4], data[5] # 这是一组正样本
            num_sample += len(data)
            item_data = item_data.to(self.device)
            user_data = user_data.to(self.device)
            item_cf_emb = item_cf_emb.to(self.device)
            user_cf_emb = user_cf_emb.to(self.device)
            
            # get the model output
            out_item, out_user, rq_loss_item, rq_loss_user, u2i_loss,indices_item, indices_user = self.model(user_data, self.user_labels,user_cf_emb, item_data, self.item_labels,item_cf_emb,user_id,item_id)

            # calculate the reconstruction loss
            if self.model.loss_type == 'mse':
                item_recon_loss = F.mse_loss(out_item, item_data, reduction='mean')
                user_recon_loss = F.mse_loss(out_user, user_data, reduction='mean')
            elif self.model.loss_type == 'l1':
                item_recon_loss = F.l1_loss(out_item, item_data, reduction='mean')
                user_recon_loss = F.l1_loss(out_user, user_data, reduction='mean')
            else:
                item_recon_loss = F.mse_loss(out_item, item_data, reduction='mean')
                user_recon_loss = F.mse_loss(out_user, user_data, reduction='mean')
            
            # calculate the perplexity of the codebook indices
            batch_indices_item = indices_item.view(-1, indices_item.shape[-1])  # [batch_size, num_layers]
            batch_indices_user = indices_user.view(-1, indices_user.shape[-1])  # [batch_size, num_layers]
            num_embeddings = self.model.Item_rq.vq_layers[0].embedding.num_embeddings
            num_embeddings_user = self.model.User_rq.vq_layers[0].embedding.num_embeddings
            batch_perplexity_item = self.calculate_perplexity(batch_indices_item, num_embeddings)
            batch_perplexity_user = self.calculate_perplexity(batch_indices_user, num_embeddings_user)
            
            total_item_codebook_perplexity += batch_perplexity_item
            total_user_codebook_perplexity += batch_perplexity_user
            total_item_reconstruction_loss += item_recon_loss.item()
            total_user_reconstruction_loss += user_recon_loss.item()
            
            # calculate the utilization of the codebook (entropy)
            batch_utilization_item = self.calculate_codebook_utilization(batch_indices_item, num_embeddings)
            batch_utilization_user = self.calculate_codebook_utilization(batch_indices_user, num_embeddings_user)
            total_item_codebook_utilization += batch_utilization_item
            total_user_codebook_utilization += batch_utilization_user
            
            # get the indices for collision rate calculation
            indices_cpu_item = indices_item.view(-1,indices_item.shape[-1]).cpu().numpy()
            indices_cpu_user = indices_user.view(-1,indices_user.shape[-1]).cpu().numpy()
            for index in indices_cpu_item:
                code = "-".join([str(int(_)) for _ in index])
                item_indices_set.add(code)
            for index in indices_cpu_user:
                code = "-".join([str(int(_)) for _ in index])
                user_indices_set.add(code)
            
        item_collision_rate = (num_sample - len(item_indices_set))/num_sample
        user_collision_rate = (num_sample - len(user_indices_set))/num_sample
        avg_item_codebook_perplexity = total_item_codebook_perplexity / len(valid_data)
        avg_user_codebook_perplexity = total_user_codebook_perplexity / len(valid_data)
        avg_item_reconstruction_loss = total_item_reconstruction_loss / len(valid_data)
        avg_user_reconstruction_loss = total_user_reconstruction_loss / len(valid_data)
        avg_item_codebook_utilization = total_item_codebook_utilization / len(valid_data)
        avg_user_codebook_utilization = total_user_codebook_utilization / len(valid_data)
        


        return item_collision_rate, user_collision_rate, avg_item_codebook_perplexity, avg_user_codebook_perplexity, avg_item_reconstruction_loss, avg_user_reconstruction_loss, avg_item_codebook_utilization, avg_user_codebook_utilization

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, item_recon_loss, user_recon_loss, u2i_loss, rq_loss_item, rq_loss_user):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        wandb.log({"train/loss": loss, "train/item_recon_loss": item_recon_loss, "train/user_recon_loss": user_recon_loss, "train/u2i_loss": u2i_loss, "train/rq_loss_item": rq_loss_item, "train/rq_loss_user": rq_loss_user})
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("item_recon loss", "blue") + ": %.4f" % item_recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("user_recon loss", "blue") + ": %.4f" % user_recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("u2i loss", "blue") + ": %.4f" % u2i_loss
        train_loss_output +=", "
        train_loss_output += set_color("rq_loss_item", "blue") + ": %.4f" % rq_loss_item
        train_loss_output +=", "
        train_loss_output += set_color("rq_loss_user", "blue") + ": %.4f" % rq_loss_user
        return train_loss_output + "]"

    def fit(self, all_data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_item_recon_loss, train_user_recon_loss, train_u2i_loss, train_rq_loss_item, train_rq_loss_user = self._train_epoch(all_data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_item_recon_loss, train_user_recon_loss, train_u2i_loss, train_rq_loss_item, train_rq_loss_user
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                item_collision_rate, user_collision_rate, avg_item_codebook_perplexity, avg_user_codebook_perplexity, avg_item_reconstruction_loss, avg_user_reconstruction_loss, avg_item_codebook_utilization, avg_user_codebook_utilization = self._valid_epoch(all_data)

                if item_collision_rate + user_collision_rate < self.best_collision_rate:
                    self.best_collision_rate = item_collision_rate + user_collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=item_collision_rate + user_collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                # if cur_eval_step >= 10:
                #     print("Finish!")
                #     break

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("item_collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("user_collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("item_codebook_perplexity", "blue")
                    + ": %.4f, "
                    + set_color("user_codebook_perplexity", "blue")
                    + ": %.4f, "
                    + set_color("item_reconstruction_loss", "blue")
                    + ": %.4f, "
                    + set_color("user_reconstruction_loss", "blue")
                    + ": %.4f, "
                    + set_color("item_codebook_utilization", "blue")
                    + ": %.4f, "
                    + set_color("user_codebook_utilization", "blue")
                    + ": %.4f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, item_collision_rate, user_collision_rate, avg_item_codebook_perplexity, avg_user_codebook_perplexity, avg_item_reconstruction_loss, avg_user_reconstruction_loss, avg_item_codebook_utilization, avg_user_codebook_utilization)
                wandb.log({
                    "val/item_collision_rate": item_collision_rate, 
                    "val/user_collision_rate": user_collision_rate, 
                    "val/item_codebook_perplexity": avg_item_codebook_perplexity, 
                    "val/user_codebook_perplexity": avg_user_codebook_perplexity, 
                    "val/item_reconstruction_loss": avg_item_reconstruction_loss, 
                    "val/user_reconstruction_loss": avg_user_reconstruction_loss, 
                    "val/item_codebook_utilization": avg_item_codebook_utilization, 
                    "val/user_codebook_utilization": avg_user_codebook_utilization
                })
                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=item_collision_rate + user_collision_rate)


        return self.best_loss, self.best_collision_rate
