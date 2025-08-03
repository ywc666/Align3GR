import torch

import transformers
from typing import List
from datasets import load_dataset
import json
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from fire import Fire
from tqdm import tqdm
import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k] 
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)

    return results,predictions

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):

    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit



device_map = "auto"
def evaluate(
    model,
    tokenizer,
    val_data,
    batch_size: int = 32,
    num_beams: int = 20,
    save_predictions: str = "",
    metrics: str = "hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
):
    def output_generate(prompts,temperature = 0,):
        # print([len(prompt) for prompt in prompts])
        inputs = tokenizer(prompts,return_tensors="pt",truncation=True,padding=True,max_length=1024).to(model.device)
        generation_config = GenerationConfig(
            # temperature = temperature,
            do_sample = False,
        )
        output = model.generate(
            **inputs,
            pad_token_id = tokenizer.pad_token_id,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_scores = True,
            num_beams=num_beams,
            max_new_tokens = 20
        )
        output_ids = output.sequences
        scores = output.sequences_scores
        output = tokenizer.batch_decode(output_ids,skip_special_tokens=True)
        output = [_.strip() for _ in output]
        return output,scores

    all_results = []
    metrics = metrics.split(",")
    targets = []
    inputs = []
    cans = []
    for elm in val_data:
        prompt = elm["query"]
        target = elm["response"]
        targets.append(target)
        inputs.append(prompt)
        cans.append(elm["rejected"])

    batch_num = (len(inputs)-1)// batch_size + 1
    score = 0
    valid = 0
    metrics_results = {}
    total = 0
    predictions_list = []
    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        output,scores = output_generate(batch_inputs)
        topk_res,predictions = get_topk_results(output, scores, targets, num_beams)
        if save_predictions:
            predictions_list.append({
                "query": output[0].split("\n\n### Response")[0],
                "response": predictions,
                "target": targets
            })

        total += batch_size
        batch_metrics_res = get_metrics_results(topk_res, metrics)
        for m, res in batch_metrics_res.items():
            if m not in metrics_results:
                metrics_results[m] = res
            else:
                metrics_results[m] += res

        if (i + 1) % 50 == 0:
            temp = {}
            for m in metrics_results:
                temp[m] = metrics_results[m] / total
            print(temp)

            if save_predictions:
                with open(save_predictions, 'w') as f:
                    for item in predictions_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total

    all_results.append(metrics_results)
    print("======================================================")
    print("Prompt results: ", metrics_results)
    print("======================================================")
    print("")

    return all_results