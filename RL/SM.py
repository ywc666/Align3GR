import argparse
import json
import os
from typing import List

from tqdm import tqdm


def prefix_ngram_score(seq1, seq2):
    score = 0
    # seq1 = seq1[0].strip('<').strip('>').split('><')
    # seq2 = seq2[0].strip('<').strip('>').split('><')
    for a, b in zip(seq1, seq2):
        if a == b:
            score += 1
    return score


def construct_progressive_rejected(
                                   candidate_scids: List[List[str]], 
                                   user_SCID: str,
                                   k_like=[5], k_neutral=[3,4], k_dislike=[1,2], top_k=5,
                                   sentiment_labels=None):
    buckets = {'like': [], 'neutral': [], 'dislike': []}
    for cand in candidate_scids:
        if user_SCID+'review'+cand not in sentiment_labels:
            buckets['dislike'].append(cand)
        else:
            if int(sentiment_labels[user_SCID+'review'+cand]['response']) in k_like:
                buckets['like'].append(cand)
            elif int(sentiment_labels[user_SCID+'review'+cand]['response']) in k_neutral:
                buckets['neutral'].append(cand)
            elif int(sentiment_labels[user_SCID+'review'+cand]['response']) in k_dislike:
                buckets['dislike'].append(cand)
        
    neutral_like = []
    dislike_like = []
    for like in buckets['like']:
        for neutral in buckets['neutral']:
            neutral_like.append((like,neutral))
    
    for like in buckets['like']:
        for dislike in buckets['dislike']:
            dislike_like.append((like,dislike))
    
    return neutral_like, dislike_like


def parse_args():
    parser = argparse.ArgumentParser(description="SM-based Progressive SCID Rejection Sampling")
    parser.add_argument('--data_path', type=str, default='Align3GR/RL/Instruments_hard/output/predictions.jsonl',
                        help='Path to the SFT outputs with prompt/label/candidates')
    parser.add_argument('--sentiment_path', type=str, default='Align3GR/data/Instruments/Instruments.RF.json',
                        help='Path to the sentiment labels')
    parser.add_argument('--user2id_path', type=str, default='Align3GR/data/Instruments/Instruments.user2id',
                        help='Path to the user2id')
    parser.add_argument('--save_path', type=str, default='./SM_progressive_data.jsonl',help='Path to save the structured progressive data')
    parser.add_argument('--top_k', type=int, default=5, help='Number of rejected responses to keep per level')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load SP-DPO outputs
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # load sentiment labels
    with open(args.sentiment_path, 'r', encoding='utf-8') as f:
        sentiment_labels = json.load(f)

    neutral_result = []
    dislike_result = []

    # like_save_path = args.save_path.replace('.jsonl', '_like.jsonl')
    neutral_save_path = args.save_path.replace('.jsonl', '_neutral.jsonl')
    dislike_save_path = args.save_path.replace('.jsonl', '_dislike.jsonl')
    
    # if os.path.exists(like_save_path):
    #     os.remove(like_save_path)
    if os.path.exists(neutral_save_path):
        os.remove(neutral_save_path)
    if os.path.exists(dislike_save_path):
        os.remove(dislike_save_path)

    for entry in tqdm(data):
        query = entry['query']              # user ID
        user_SCID = query.split('The user ')[1].split(' has interacted')[0]
        label = entry['target']                  # ground-truth SCID (tokenized list)
        response = entry['response']        # list of generated SCIDs (list of tokenized lists)

        neutral_like, dislike_like = construct_progressive_rejected(
            response,user_SCID,
            k_like=[5], k_neutral=[3,4], k_dislike=[1,2],
            top_k=args.top_k,
            sentiment_labels=sentiment_labels
        )
        
        with open(neutral_save_path, 'a', encoding='utf-8') as f_neutral:
            for like, neutral in neutral_like:
                now = {
                    'query': query,
                    'response': like,
                    'rejected': neutral,
                }
                f_neutral.write(json.dumps(now,ensure_ascii=False) + '\n')

        with open(dislike_save_path, 'a', encoding='utf-8') as f_dislike:
            for like, dislike in dislike_like:
                now = {
                    'query': query,
                    'response': like,
                    'rejected': dislike,
                }
                f_dislike.write(json.dumps(now,ensure_ascii=False) + '\n')


