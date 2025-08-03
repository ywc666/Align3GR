import argparse
import os
import sys
from typing import List
# import wandb
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# from modeling_letter import LETTER
# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import transformers


from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from collator import Collator



def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}

    config = LlamaConfig.from_pretrained(args.base_model)
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0
    # 加载数据集
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        device_map={"": local_rank},  # 手动绑定设备，避免 auto
    )
    model.set_hyper(args.temperature)

    # model = model.to(torch.device(f"cuda:{local_rank}"))
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)


    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        args.resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")

    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in config.modules_to_save):
            p.requires_grad = False

    if local_rank == 0:
        model.print_trainable_parameters()


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            # report_to=['wandb'],
            report_to='none',
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=True,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            # report_to=None,
            eval_delay=1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    DATASET='Instruments'
    BASE_MODEL='huggyllama/llama2-7b' # LLaMA
    DATA_PATH='../data'
    OUTPUT_DIR=f'./ckpt/{DATASET}/'
    # parser.add_argument("--base_model", type=str, default=BASE_MODEL, help='base model')
    # parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help='output directory')
    # parser.add_argument("--data_path", type=str, default=DATA_PATH, help='data path')
    # parser.add_argument("--dataset", type=str, default=DATASET, help='dataset')
    # --per_device_batch_size 16 \
    # --learning_rate 1e-4 \
    # --epochs 4 \
    # --tasks seqrec \
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    train(args)
