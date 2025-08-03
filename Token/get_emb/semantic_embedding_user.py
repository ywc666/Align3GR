import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import set_device, load_json, load_plm, clean_text
from sentence_transformers import SentenceTransformer



def load_data(args):

    user2feature_path = os.path.join(args.root, f'{args.dataset}.user.json')
    user2feature = load_json(user2feature_path)

    return user2feature

def generate_text(user2feature, features):
    user_text_list = []

    # Data structure: user2feature is a dictionary containing 'user_explicit_preference' and 'user_vague_intention'
    # We only need to process 'user_explicit_preference'
    if 'user_explicit_preference' in user2feature:
        user_preferences = user2feature['user_explicit_preference']
        
        for user_id, preference_texts in user_preferences.items():
            text = []
            for pref_text in preference_texts:
                cleaned_text = clean_text(pref_text)
                if cleaned_text:
                    text.append(cleaned_text)
            
            if text:  # Only add users with text
                user_text_list.append([int(user_id), text])

    return user_text_list

def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)

    user2feature = load_data(args)
    # load user text and clean - directly process user preference and intention data
    user_text_list = generate_text(user2feature, [])
    # return: list of (user_ID, cleaned_user_text)
    return user_text_list

def generate_user_embedding(args, user_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    if not user_text_list:
        print("No user text data found!")
        return

    users, texts = zip(*user_text_list)
    print(f'Processing text data for {len(users)} users')
    
    embeddings = []
    valid_user_ids = []
    
    for i, (user_id, text_list) in enumerate(zip(users, texts)):
        if (i+1) % 1000 == 0:
            print(f"Processing progress: {i+1}/{len(users)}")
        
        if not text_list:  # Skip users without text
            continue
            
        # Combine all user text into a complete sentence
        combined_text = ' '.join(text_list)
        
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            words = combined_text.split(' ')
            new_words = []
            for word in words:
                rd = random.random()
                if rd > word_drop_ratio:
                    new_words.append(word)
            combined_text = ' '.join(new_words)
        
        if args.plm_name == 't5':
            embeddings.append(model.encode(combined_text))
            valid_user_ids.append(user_id)
        else:
            # Encode the combined text
            encoded_sentences = tokenizer([combined_text], max_length=args.max_sent_len,
                                        truncation=True, return_tensors='pt', padding="longest").to(args.device)
            
            with torch.no_grad():  # Model forward pass, get hidden states
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)
            
            # Process hidden states to get mean_output: compress into a fixed-dimensional overall semantic vector
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            
            embeddings.append(mean_output)
            valid_user_ids.append(user_id)

    if not embeddings:

        return
        
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # Save embedding vectors and user ID mapping
    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-user-pref" + ".npy")
    np.save(file, embeddings)
    
    # Save user ID mapping
    user_mapping_file = os.path.join(args.root, args.dataset + '.user_mapping.json')
    user_mapping = {str(user_id): idx for idx, user_id in enumerate(valid_user_ids)}
    with open(user_mapping_file, 'w') as f:
        json.dump(user_mapping, f, indent=2)
    
    print(f'Saved embeddings to: {file}')
    print(f'Saved user mapping to: {user_mapping_file}')
    print(f'Generated embeddings for {len(valid_user_ids)} users')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='t5',help='llama / t5')
    # parser.add_argument('--plm_checkpoint', type=str,default='LLM-Research/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--plm_checkpoint_llama', type=str,default='huggyllama/llama-7b')
    parser.add_argument('--plm_checkpoint_t5', type=str,default='sentence-transformers/sentence-t5-base')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device

    user_text_list = preprocess_text(args)
    if args.plm_name == 't5':
        plm_model = SentenceTransformer(args.plm_checkpoint_t5)
        plm_tokenizer = None
    else:
        plm_tokenizer, plm_model = load_plm(args.plm_checkpoint_llama)
        if plm_tokenizer.pad_token_id is None:
            plm_tokenizer.pad_token_id = 0
        # plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(device)

    generate_user_embedding(args, user_text_list,plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio)


