import os
import torch
import sys
import wandb
import argparse
from tqdm import tqdm
import numpy as np


sys.path.append('../..')

def parse_args():
    parser = argparse.ArgumentParser(description='DIN Model Training Script')
    
    # Dataset parameters
    parser.add_argument('--data_set_name', type=str, default='Instruments',
                       help='Dataset name (gowalla, MIND, Amazon_All_Beauty, Instruments)')
    parser.add_argument('--raw_data_file', type=str, 
                       default='data/{}/Musical_Instruments_5.json',
                       help='Raw data file path template')
    parser.add_argument('--train_instances_file', type=str,
                       default='data/{}/item_train_instances',
                       help='Training instances file path template')
    parser.add_argument('--test_instances_file', type=str,
                       default='data/{}/item_test_instances',
                       help='Test instances file path template')
    parser.add_argument('--validation_instances_file', type=str,
                       default='data/{}/item_validation_instances',
                       help='Validation instances file path template')
    parser.add_argument('--item_num_node_num_file', type=str,
                       default='data/{}/item_node_num.txt',
                       help='Item and node count file path template')
    parser.add_argument('--raw_meta_item_file', type=str,
                       default='data/{}/meta_Musical_Instruments.json',
                       help='Raw meta item file path template')
    
    # Model parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cpu, cuda:0, cuda:1, etc.)')
    parser.add_argument('--topk', type=int, default=10, help='Top-K recommendation count')
    parser.add_argument('--emb_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--sum_pooling', action='store_true', help='Whether to use sum pooling')
    parser.add_argument('--sample_negative_num', type=int, default=60, help='Negative sample count')
    parser.add_argument('--feature_groups', type=int, nargs='+', default=[5,4,2,2,1,1,1,1,1,1],
                       help='Feature group configuration')
    
    # Training parameters
    parser.add_argument('--train_sample_seg_cnt', type=int, default=10, 
                       help='Training data segment count')
    parser.add_argument('--parall', type=int, default=10, help='Parallel processing count')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--min_seq_len', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--test_user_num', type=int, default=0, help='Test user count')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Test batch size')
    parser.add_argument('--batch_number', type=int, default=800000, help='Total batch number')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--have_processed_data', action='store_true', 
                       help='Whether data has been processed')
    
    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='DIN', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity')
    parser.add_argument('--wandb_name', type=str, default='DIN', help='wandb run name')
    parser.add_argument('--wandb_dir', type=str, default='./DIN_Model', help='wandb directory')
    
    return parser.parse_args()

def presision(result_list, gt_list, top_k):
    count = 0.0
    for r, g in zip(result_list, gt_list):
        count += len(set(r).intersection(set(g)))
    return count / (top_k * len(result_list))

def recall(result_list, gt_list):
    t = 0.0
    for r, g in zip(result_list, gt_list):
        t += 1.0 * len(set(r).intersection(set(g))) / len(g)
    return t / len(result_list)

def f_measure(result_list, gt_list, top_k, eps=1.0e-9):
    f = 0.0
    for r, g in zip(result_list, gt_list):
        recc = 1.0 * len(set(r).intersection(set(g))) / len(g)
        pres = 1.0 * len(set(r).intersection(set(g))) / top_k
        if recc + pres < eps:
            continue
        f += (2 * recc * pres) / (recc + pres)
    return f / len(result_list)

def novelty(result_list, s_u, top_k):
    count = 0.0
    for r, g in zip(result_list, s_u):
        count += len(set(r) - set(g))
    return count / (top_k * len(result_list))

def hit_ratio(result_list, gt_list):
    intersetct_set = [len(set(r) & set(g)) for r, g in zip(result_list, gt_list)]
    return 1.0 * sum(intersetct_set) / sum([len(gts) for gts in gt_list])

def NDCG_bug(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asarray([1 if r in setgt else 0 for r in re], dtype=float)
        sorted_indicator = indicator[indicator.argsort(-1)[::-1]]
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0 * np.arange(2, len(indicator) + 2))) / \
                 np.sum(sorted_indicator / np.log2(1.0 * np.arange(2, len(indicator) + 2)))
    return t / len(gt_list)

def NDCG_(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asarray([1 if r in setgt else 0 for r in re], dtype=float)
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0 * np.arange(2, len(indicator) + 2))) / \
                 np.sum(sorted_indicator / np.log2(1.0 * np.arange(2, len(sorted_indicator) + 2)))
    return t / len(gt_list)

import math
def NDCG_comicrec(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        recall = 0
        dcg = 0.0
        setgt = set(gt)
        for no, iid in enumerate(re):
            if iid in setgt:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
        idcg = 0.0
        for no in range(recall):
            idcg += 1.0 / math.log(no + 2, 2)
        if recall > 0:
            t += dcg / idcg
    return t / len(gt_list)

def MAP(result_list, gt_list, topk):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asarray([1 if r in setgt else 0 for r in re], dtype=float)
        t += np.mean([indicator[:i].sum(-1) / i for i in range(1, topk + 1)], axis=-1)
    return t / len(gt_list)

def NDCG(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = gt
        indicator = np.asarray([1 if r in setgt else 0 for r in re], dtype=float)
        
        sorted_indicator = np.ones(len(gt))
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0 * np.arange(2, len(indicator) + 2))) / \
                 np.sum(sorted_indicator / np.log2(1.0 * np.arange(2, len(sorted_indicator) + 2)))
    return t / len(gt_list)

def main():
    args = parse_args()
    
    # Set device
    if args.device != 'cpu':
        torch.cuda.set_device(args.device)
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Optimizer configuration
    optimizer = lambda params: torch.optim.Adam(params, lr=args.learning_rate, amsgrad=True)
    
    # Format file paths
    raw_data_file = args.raw_data_file.format(args.data_set_name)
    train_instances_file = args.train_instances_file.format(args.data_set_name)
    test_instances_file = args.test_instances_file.format(args.data_set_name)
    validation_instances_file = args.validation_instances_file.format(args.data_set_name)
    item_num_node_num_file = args.item_num_node_num_file.format(args.data_set_name)
    raw_meta_item_file = args.raw_meta_item_file.format(args.data_set_name)
    
    # wandb initialization
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            dir=args.wandb_dir,
            job_type="training",
            reinit=True
        )
    
    from lib.generate_train_and_test_data import _gen_train_sample, _read, _gen_test_sample
    from lib import generate_train_and_test_data
    import numpy as np
    
    # Data processing
    if not args.have_processed_data:
        behavior_dict, train_sample, test_sample, validation_sample, user_num, item_num, user_ids = generate_train_and_test_data._read(
            raw_data_file, args.test_user_num, raw_meta_item_file)
        
        stat = generate_train_and_test_data._gen_train_sample(
            train_sample, train_instances_file, test_sample=test_sample,
            train_sample_seg_cnt=args.train_sample_seg_cnt,
            parall=args.parall, seq_len=args.seq_len, min_seq_len=args.min_seq_len, 
            user_ids=user_ids, mode='item')
        
        _gen_test_sample(train_sample, test_instances_file, seq_len=args.seq_len, 
                        min_seq_len=args.min_seq_len, mode='item')
        _gen_test_sample(validation_sample, validation_instances_file, seq_len=args.seq_len, 
                        min_seq_len=args.min_seq_len, mode='item')
        
        del behavior_dict, train_sample, test_sample, stat
        np.savetxt(item_num_node_num_file, np.array([user_num, item_num]), fmt='%d', delimiter=',')
    else:
        [user_num, item_num] = np.loadtxt(item_num_node_num_file, dtype=np.int32, delimiter=',')
    
    print('user num is {}, item is {}'.format(user_num, item_num))
    
    from lib import DINTrain
    
    train_model = DINTrain(
        item_num=item_num,
        sample_negative_num=args.sample_negative_num,
        emb_dim=args.emb_dim,
        device=device,
        sum_pooling=args.sum_pooling,
        feature_groups=args.feature_groups,
        optimizer=optimizer
    )
    print(train_model.DINModel)
    
    from lib.generate_training_batches import Train_instance
    train_instances = Train_instance(parall=args.parall)
    
    his_maxtix_path = 'data/{}/item_his_maxtix.pt'.format(args.data_set_name)
    labels_path = 'data/{}/item_labels.pt'.format(args.data_set_name)
    training_data, training_labels = train_instances.get_training_data(
        train_instances_file, args.train_sample_seg_cnt, item_num, his_maxtix_path, labels_path)
    
    test_batch_generator = train_instances.test_batches(test_instances_file, item_num, batchsize=args.test_batch_size)
    validation_batch_generator = train_instances.validation_batches(validation_instances_file, item_num, batchsize=args.test_batch_size)
    test_instances = train_instances.read_test_instances_file(test_instances_file, item_num)
    
    # Training history
    loss_history, dev_precision_history, dev_recall_history, dev_f_measure_history, dev_novelty_history, dev_ndcg_history, policy_acc = [], [], [], [], [], [], []
    test_precision_history, test_recall_history, test_f_measure_history, test_novelty_history, test_ndcg_history = [], [], [], [], []
    total_precision_history, total_recall_history, total_f_measure_history, total_novelty_history, total_ndcg_history, total_hit_history = [], [], [], [], [], []
    
    train_model.DINModel.train()
    
    for (batch_x, batch_y) in train_instances.generate_training_records(training_data, training_labels, batch_size=args.batch_size):
        loss = train_model.update_DIN(batch_x, batch_y)
        loss_history.append(loss.item())
        
        if args.use_wandb:
            wandb.log({"loss": loss.item()})
        
        if train_model.batch_num % 1000 == 0:  # evaluate once
            train_model.DINModel.eval()
            gt_history = train_instances.test_labels
            all_items = torch.arange(item_num, device=device).view(-1, 1)
            preference_matrix = torch.full((len(test_instances), item_num), -1.0e9, dtype=torch.float32)
            batch_size = 2000
            print(test_instances.shape)
            f_num = test_instances.shape[1]
            
            for i, user in enumerate(tqdm(test_instances)):
                start_id = 0
                while start_id < item_num:
                    part_labels = all_items[start_id:start_id + batch_size, :]
                    with torch.no_grad():
                        preference_matrix[i, start_id:start_id + batch_size] = train_model.calculate_preference(
                            user.to(device).expand(len(part_labels), f_num), part_labels).view(1, -1).cpu()
                    start_id = start_id + batch_size
            
            resutl_history = preference_matrix.argsort(dim=-1)[:, -args.topk:].numpy()
            resutl_history = resutl_history[:, ::-1]
            
            total_precision_history.append(presision(resutl_history, gt_history, args.topk))
            total_recall_history.append(recall(resutl_history, gt_history))
            total_f_measure_history.append(f_measure(resutl_history, gt_history, args.topk))
            total_novelty_history.append(novelty(resutl_history, test_instances.tolist(), args.topk))
            total_ndcg_history.append(NDCG(resutl_history, gt_history))
            total_hit_history.append(hit_ratio(resutl_history, gt_history))
            
            print('precision: {}'.format(total_precision_history[-1]))
            print('recall: {}'.format(total_recall_history[-1]))
            print('f-score: {}'.format(total_f_measure_history[-1]))
            print('ndcg: {}'.format(total_ndcg_history[-1]))
            print('hit_rate: {}'.format(total_hit_history[-1]))
            
            if args.use_wandb:
                wandb.log({
                    "precision": total_precision_history[-1],
                    "recall": total_recall_history[-1],
                    "f-score": total_f_measure_history[-1],
                    "ndcg": total_ndcg_history[-1],
                    "hit_rate": total_hit_history[-1]
                })
            
            train_model.DINModel.train()
        
        if train_model.batch_num % 1000 == 0:  # save model once every 1000 batches
            DIN_Model_path = 'data/{}/item_DIN_MODEL/DIN_MODEL_{}.pt'.format(args.data_set_name, (train_model.batch_num))
            os.makedirs(os.path.dirname(DIN_Model_path), exist_ok=True)
            torch.save(train_model.DINModel, DIN_Model_path)
            print('saved')
        
        if train_model.batch_num % 100 == 0:  # print every 100 batches
            print("step=%i, mean_loss=%.3f, time=%.3f" %
                  (len(loss_history), np.mean(loss_history[-100:]), 1.0))
            if args.use_wandb:
                wandb.log({"mean_loss": np.mean(loss_history[-100:])})
        
        if train_model.batch_num > args.batch_number:
            break
    
    # Final evaluation
    train_model.DINModel.eval()
    gt_history = train_instances.test_labels
    all_items = torch.arange(item_num, device=device).view(-1, 1)
    preference_matrix = torch.full((len(test_instances), item_num), -1.0e9, dtype=torch.float32)
    batch_size = 2000
    f_num = test_instances.shape[1]
    
    for i, user in enumerate(test_instances):
        start_id = 0
        while start_id < item_num:
            part_labels = all_items[start_id:start_id + batch_size, :]
            with torch.no_grad():
                preference_matrix[i, start_id:start_id + batch_size] = train_model.calculate_preference(
                    user.to(device).expand(len(part_labels), f_num), part_labels).view(1, -1).cpu()
            start_id = start_id + batch_size
    
    resutl_history = preference_matrix.argsort(dim=-1)[:, -args.topk:].numpy()
    total_precision_history.append(presision(resutl_history, gt_history, args.topk))
    total_recall_history.append(recall(resutl_history, gt_history))
    total_f_measure_history.append(f_measure(resutl_history, gt_history, args.topk))
    train_model.DINModel.train()
    
    DIN_Model_path = 'data/{}/DIN_MODEL.pt'.format(args.data_set_name)
    torch.save(train_model.DINModel, DIN_Model_path)
    print(total_precision_history[-1], total_recall_history[-1], total_f_measure_history[-1], total_novelty_history[-1])
    
    sorted_test_users_path = 'data/{}/sorted_test_users.txt'.format(args.data_set_name)
    np.savetxt(sorted_test_users_path, preference_matrix.argsort(dim=-1).numpy(), delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
