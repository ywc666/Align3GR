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


def construct_progressive_rejected(label_scid: List[str],
                                   candidate_scids: List[List[str]],
                                   k_easy=0, k_medium=1, k_hard=2, top_k=5):
    buckets = {'easy': [], 'medium': [], 'hard': []}
    for cand in candidate_scids:
        seq1 = label_scid[0].strip('<').strip('>').split('><')
        seq2 = cand.strip('<').strip('>').split('><')
        if not len(seq1) == len(seq2):
            continue
        score = prefix_ngram_score(seq1, seq2)
        if score == k_easy:
            buckets['easy'].append(cand)
        elif score == k_medium:
            buckets['medium'].append(cand)
        elif score >= k_hard:
            buckets['hard'].append(cand)
    # Top-k sampling
    for key in buckets:
        buckets[key] = buckets[key]
    return buckets


def parse_args():
    parser = argparse.ArgumentParser(description="PNM-based Progressive SCID Rejection Sampling")
    parser.add_argument('--data_path', type=str, default='Align3GR/RL/Instruments_easy/output/predictions.jsonl',
                        help='Path to the SFT outputs with prompt/label/candidates')
    parser.add_argument('--save_path', type=str, default='./PNM_progressive_data.jsonl',help='Path to save the structured progressive data')
    parser.add_argument('--top_k', type=int, default=5, help='Number of rejected responses to keep per level')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load SFT outputs
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    result = []

    easy_save_path = args.save_path.replace('.jsonl', '_easy.jsonl')
    medium_save_path = args.save_path.replace('.jsonl', '_medium.jsonl')
    hard_save_path = args.save_path.replace('.jsonl', '_hard.jsonl')
    
    if os.path.exists(easy_save_path):
        os.remove(easy_save_path)
    if os.path.exists(medium_save_path):
        os.remove(medium_save_path)
    if os.path.exists(hard_save_path):
        os.remove(hard_save_path)

    for entry in tqdm(data):
        query = entry['query']              # user ID
        label = entry['target']                  # ground-truth SCID (tokenized list)
        response = entry['response']        # list of generated SCIDs (list of tokenized lists)

        progressive_rejects = construct_progressive_rejected(
            label, response,
            k_easy=0, k_medium=1, k_hard=2,
            top_k=args.top_k
        )

        result.append({
            'label': label,
            'rejected_easy': progressive_rejects['easy'],
            'rejected_medium': progressive_rejects['medium'],
            'rejected_hard': progressive_rejects['hard'],
        })

        print('easy:',len(progressive_rejects['easy']),'medium:',len(progressive_rejects['medium']),'hard:',len(progressive_rejects['hard']))
        prompt =  "The user {user_id} has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
        # easy rejected SCIDs
        # Save each level of rejected SCIDs to separate files


        with open(easy_save_path, 'a', encoding='utf-8') as f_easy:
            now = {
                'query': query,
                'response': label,
                'rejected': progressive_rejects['easy'],
            }
            for scid in progressive_rejects['easy']:
                f_easy.write(json.dumps(now,ensure_ascii=False) + '\n')

        # medium rejected SCIDs
        with open(medium_save_path, 'a', encoding='utf-8') as f_medium:
            now = {
                'query': query,
                'response': label,
                'rejected': progressive_rejects['medium'],
            }
            for scid in progressive_rejects['medium']:
                f_medium.write(json.dumps(now,ensure_ascii=False) + '\n')


        # hard rejected SCIDs
        with open(hard_save_path, 'a', encoding='utf-8') as f_hard:
            now = {
                'query': query,
                'response': label,
                'rejected': progressive_rejects['hard'],
            }
            for scid in progressive_rejects['hard']:
                f_hard.write(json.dumps(now,ensure_ascii=False) + '\n')

    # # Save result
    # with open(args.save_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)

    # print(f"Saved progressive rejected SCIDs to {args.save_path}")
