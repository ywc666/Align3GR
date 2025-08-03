import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# from trl import DPOTrainer
from trainer.softmax_dpo_trainer import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

random.seed(1958)
def train(
    #train
    output_dir="Align3GR/DPO/Instruments_easy/output",
    logging_dir="Align3GR/DPO/Instruments_easy/output/logs",
    model_name ='/home/ecs-user/nas_original_data/csh/MODEL/LLM-Research/llama-2-7b',
    prompt_path = "Align3GR/DPO/prompt/movie.txt",
    dataset="lastfm",
    resume_from_checkpoint: str = "Align3GR/SFT/ckpt/Instruments/v2/checkpoint-256",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "DPO",
    wandb_name: str = "dpo-instruments-easy",   # the name of the wandb run
    # training hyperparameters
    beta: float = 1,
    neg_num: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 0.1, 
    data_train = "Align3GR/DPO/sample_data/Instrument_easy/data_easy_train.jsonl",
    data_val = "Align3GR/DPO/sample_data/Instrument_easy/data_easy_val.jsonl",
    deepspeed="config/ds_z2_fp16.json",  # Updated path to DeepSpeed config file
):
    
    data_files = {
        "train": data_train,
        "validation": data_val,
    }

    def process_data(examples):
        dic = {"prompt":[], "chosen":[]}
        for i in range(1, neg_num+1):
            dic[f"rejected{i}"] = []
        
        queries = examples["query"]
        responses = examples["response"]
        rejecteds = examples["rejected"]
        
        for i in range(len(queries)):
            prompt = queries[i]
            response = responses[i]
            rejected = rejecteds[i]
            
            # chosen
            chosen = response if isinstance(response, list) else response
            
            # rejected
            rejected_items = rejected if isinstance(rejected, list) else [rejected]
            
            dic["prompt"].append(prompt)
            dic["chosen"].append(chosen)
            
            available_rejected = rejected_items[:neg_num]
            while len(available_rejected) < neg_num:
                available_rejected.append(rejected_items[0] if rejected_items else chosen)
            
            for j in range(neg_num):
                dic[f"rejected{j+1}"].append(available_rejected[j])
        return dic

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)
    print(train_data)

    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = LlamaTokenizer.from_pretrained(resume_from_checkpoint)
    tokenizer.pad_token_id = (0)
    
    base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
    # print_trainable_parameters(base_model)
    base_model.print_trainable_parameters()

    model_ref = LlamaForCausalLM.from_pretrained(model_name,
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    model_ref.resize_token_embeddings(len(tokenizer)) # 这里需要resize tokenizer
    reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    reference_model.print_trainable_parameters()


    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        load_best_model_at_end=True,
        logging_steps=1,
        rpo_alpha=1,
        output_dir=output_dir,
        # report_to = "wandb",
        report_to = "none",
        run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
        label_names=[],
        # deepspeed=deepspeed,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)