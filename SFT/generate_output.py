import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import sys

import torch
import transformers
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from collator import TestCollator
from my_prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results



def test(args):
    set_seed(args.seed)
    print(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained('Align3GR/SFT/ckpt/Instruments/v2/checkpoint-256')

    if args.lora:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}, 
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,              
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            attn_implementation="flash_attention_2",
        )
    # assert model.config.vocab_size == len(tokenizer)

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    print(args.test_prompt_ids, prompt_ids)

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    prefix_allowed_tokens_ = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=2, pin_memory=True)

    print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():

        for prompt_id in prompt_ids:

            print("Start prompt: ", prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0
            predictions_list = []
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                bs = len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            # prefix_allowed_tokens_fn = prefix_allowed_tokens_,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams - 1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res,predictions = get_topk_results(output, scores, targets, num_beams, all_items=all_items if args.filter_items else None)
                if args.save_predictions:
                    predictions_list.append({
                        "query": output[0].split("\n\n### Response")[0],
                        "response": predictions,
                        "target": targets
                    })

                total += bs
                batch_metrics_res = get_metrics_results(topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 50 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
                    
                    if args.save_predictions:
                        with open(args.save_predictions, 'w') as f:
                            for item in predictions_list:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results
   
    if not os.path.exists(args.results_file):
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print("Save file: ", args.results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument("--save_predictions", default='./predictions.jsonl')
    args = parser.parse_args()
    test(args)
