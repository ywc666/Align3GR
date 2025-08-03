import json
import os
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import transformers
from evaluate_batch import evaluate
from peft import PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator
import fire

def process_data(examples, neg_num=20):
    dic = {"prompt":[], "chosen":[]}
    for i in range(1, neg_num+1):
        dic[f"rejected{i}"] = []
    
    # when using batched=True, examples is a Dataset object
    # can access column data via dictionary
    queries = examples["query"]  # string list
    responses = examples["response"]  # list of lists
    rejecteds = examples["rejected"]  # list of lists
    
    for i in range(len(queries)):
        prompt = queries[i]  # single string
        response = responses[i]  # single list
        rejected = rejecteds[i]  # single list
        
        # process chosen (correct answer)
        chosen = response if isinstance(response, list) else response
        
        # process rejected (wrong answer)
        rejected_items = rejected if isinstance(rejected, list) else [rejected]
        
        dic["prompt"].append(prompt)
        dic["chosen"].append(chosen)
        
        # ensure enough rejected samples
        available_rejected = rejected_items[:neg_num]
        while len(available_rejected) < neg_num:
            available_rejected.append(rejected_items[0] if rejected_items else chosen)
        
        for j in range(neg_num):
            dic[f"rejected{j+1}"].append(available_rejected[j])
    return dic

def inference( dataset="",
               batch_size: int = 0,
               resume_from_checkpoint: str = "",
               base_model = "",
               neg_num = 20,
               results_file = "",
               num_beams = 200,
               ):
    base_model = base_model
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        # load_in_8bit=True,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    if "Llama-3" in base_model:
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(resume_from_checkpoint)
        
    tokenizer.pad_token_id = (0)
    
    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    model.resize_token_embeddings(len(tokenizer))

    if resume_from_checkpoint != "":
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    model.eval()


    # tokenizer.padding_side = "left"
    

    
    def convert_dict_to_prompt(d:dict):
        return d["prompt"]
    
    def generate_and_tokenize_prompt(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        dic = data_point
        dic["prompt"] = prompt[:-1]
        return dic
    

    data_files = {
        "test": "test.json",
    }


    data = load_dataset("json", data_files=data_files)

    data.cleanup_cache_files()
    print(data)

    columns = data["test"].column_names
    test_data = data["test"].map(process_data(neg_num=neg_num), remove_columns=columns, num_proc=8, batched=True)


    metrics = "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
    results = evaluate(model, tokenizer, test_data, batch_size=batch_size, num_beams=num_beams, save_predictions=results_file, metrics=metrics)
    mean_results = {}
    min_results = {}
    max_results = {}    
    for m in results[0]:
        all_res = [_[m] for _ in results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)


    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = results
    
    if not os.path.exists(results_file):
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print("Save file: ", results_file)

if __name__ == "__main__":
    fire.Fire(inference)
