import json
import torch
import numpy as np
import argparse
import sys
sys.path.append('Align3GR/DIN/lib')
sys.path.append('Align3GR/DIN')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate User CF Embeddings from Item Embeddings')
    
    # File paths
    parser.add_argument('--cf_emb_file', type=str, default="data/Instruments/item_DIN_MODEL/DIN_MODEL_4000.pt",
                       help='Path to the CF model file')
    parser.add_argument('--inters_file', type=str, default="data/Instruments/Instruments.inter.json",
                       help='Path to the interactions file')
    parser.add_argument('--save_file', type=str, default="data/Instruments/item_DIN_MODEL/user_cf_emb.npy",
                       help='Path to save the user CF embeddings')
    
    # Pooling method
    parser.add_argument('--pooling_method', type=str, default='attention', 
                       choices=['mean', 'sum', 'attention', 'temporal'],
                       help='Pooling method to aggregate item embeddings')
    
    # Attention parameters
    parser.add_argument('--attention_decay', type=float, default=0.9,
                       help='Decay factor for temporal weighted pooling')
    
    return parser.parse_args()

def attention_pooling(embeddings, query=None):
    if query is None:
        query = np.mean(embeddings, axis=0)
    
    scores = np.dot(embeddings, query) / np.sqrt(query.shape[0])
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    user_emb = np.sum(embeddings * attention_weights[:, np.newaxis], axis=0)
    return user_emb

def temporal_weighted_pooling(embeddings, decay=0.9):
    weights = [decay ** i for i in range(len(embeddings))]
    weights = np.array(weights) / np.sum(weights)
    return np.average(embeddings, axis=0, weights=weights)

def main():
    args = parse_args()
    
    try:
        import lib
        print("Successfully imported lib module")
    except ImportError as e:
        print(f"Warning: Could not import lib module: {e}")
    
    # Load data
    inters = json.load(open(args.inters_file, 'r'))
    cf_model = torch.load(args.cf_emb_file)
    
    # Get item embeddings (excluding the last item which is usually padding)
    item_cf_emb = cf_model.item_embedding.embed.weight.data[:-1,:].cpu().numpy()
    
    results = []
    
    for user_id, items in inters.items():
        user_cf_emb_list = []
        for item in items:
            user_cf_emb_list.append(item_cf_emb[item])
        
        if len(user_cf_emb_list) == 0:
            continue
        
        user_cf_emb_list = np.array(user_cf_emb_list)
        
        # Apply selected pooling method
        if args.pooling_method == 'mean':
            user_cf_emb = np.mean(user_cf_emb_list, axis=0)
        elif args.pooling_method == 'sum':
            user_cf_emb = np.sum(user_cf_emb_list, axis=0)
        elif args.pooling_method == 'attention':
            user_cf_emb = attention_pooling(user_cf_emb_list)
        elif args.pooling_method == 'temporal':
            user_cf_emb = temporal_weighted_pooling(user_cf_emb_list, args.attention_decay)
        else:
            raise ValueError(f"Unknown pooling method: {args.pooling_method}")
        
        results.append(user_cf_emb)
    
    results = np.array(results)
    np.save(args.save_file, results)
    
    print(f"User CF embeddings generated successfully!")
    print(f"Method used: {args.pooling_method}")
    print(f"Number of users: {len(results)}")
    print(f"Embedding dimension: {results.shape[1]}")
    print(f"Saved to: {args.save_file}")

if __name__ == "__main__":
    main()

