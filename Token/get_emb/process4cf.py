import json
import torch
import sys
import argparse
from tqdm import tqdm

sys.path.append('../..')

def parse_args():
    parser = argparse.ArgumentParser(description='Recommendation System Data Processing Script')
    
    # Dataset related parameters
    parser.add_argument('--data_set_name', type=str, default='Instruments', 
                       help='Dataset name (gowalla, MIND, Amazon_All_Beauty, Instruments)')
    parser.add_argument('--raw_data_file', type=str, 
                       default='data/Instruments/Musical_Instruments_5.json',
                       help='Raw data file path')
    parser.add_argument('--user_id_file', type=str,
                       default='data/Instruments/Instruments.user2id',
                       help='User ID mapping file path')
    parser.add_argument('--item_id_file', type=str,
                       default='data/Instruments/Instruments.item2id',
                       help='Item ID mapping file path')
    parser.add_argument('--inter_file', type=str,
                       default='data/Instruments/Instruments.inter.json',
                       help='Interaction data file path')
    parser.add_argument('--item_file', type=str,
                       default='data/Instruments/Instruments.item.json',
                       help='Item information file path')
    
    # Output file paths
    parser.add_argument('--train_instances_file', type=str,
                       default='recommend/{}/train_instances',
                       help='Training instances file path template')
    parser.add_argument('--test_instances_file', type=str,
                       default='recommend/{}/test_instances',
                       help='Test instances file path template')
    parser.add_argument('--validation_instances_file', type=str,
                       default='recommend/{}/validation_instances',
                       help='Validation instances file path template')
    parser.add_argument('--item_num_node_num_file', type=str,
                       default='recommend/{}/item_node_num.txt',
                       help='Item and node count file path template')
    
    # Processing parameters
    parser.add_argument('--train_sample_seg_cnt', type=int, default=10, 
                       help='Training data segment count')
    parser.add_argument('--parall', type=int, default=10, help='Parallel processing count')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--min_seq_len', type=int, default=5, help='Minimum sequence length')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Format file paths
    train_instances_file = args.train_instances_file.format(args.data_set_name)
    test_instances_file = args.test_instances_file.format(args.data_set_name)
    validation_instances_file = args.validation_instances_file.format(args.data_set_name)
    item_num_node_num_file = args.item_num_node_num_file.format(args.data_set_name)
    
    from lib.generate_train_and_test_data import _gen_train_sample, _read, _gen_test_sample
    from lib import generate_train_and_test_data
    import numpy as np
    
    # Read user and item mappings
    user_dict = {}
    item_dict = {}
    
    with open(args.user_id_file, 'r') as f:
        for line in f:
            user_dict[line.split('\t')[0]] = int(line.split('\t')[1])
    user_ids = list(user_dict.keys())
    
    with open(args.item_id_file, 'r') as f:
        for line in f:
            item_dict[line.split('\t')[0]] = int(line.split('\t')[1])
    
    with open(args.inter_file, 'r') as f:
        user_behav = json.load(f)
    
    # Generate training samples
    stat = generate_train_and_test_data._gen_train_sample(
        args.train_sample_seg_cnt, 
        train_instances_file, 
        parall=args.parall, 
        seq_len=args.seq_len, 
        min_seq_len=args.min_seq_len,
        user_his_behav=user_behav
    )
    
    # Generate test samples
    _gen_test_sample(
        test_instances_file, 
        seq_len=args.seq_len, 
        min_seq_len=args.min_seq_len,
        user_his_behav=user_behav
    )
    
    # Generate validation samples
    _gen_test_sample(
        validation_instances_file, 
        seq_len=args.seq_len, 
        min_seq_len=args.min_seq_len,
        user_his_behav=user_behav
    )
    
    # Clean memory and save statistics
    del stat
    np.savetxt(
        item_num_node_num_file,
        np.array([len(user_dict), len(item_dict)]),
        fmt='%d',
        delimiter=','
    )
    
    print(f"Data processing completed!")
    print(f"User count: {len(user_dict)}")
    print(f"Item count: {len(item_dict)}")

if __name__ == "__main__":
    main()



