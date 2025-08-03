
# Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Project Overview

Align³GR is an innovative framework designed to address semantic and behavioral misalignment challenges when applying Large Language Models (LLMs) to recommender systems. Our approach bridges this gap by unifying alignment across three levels:

- **Token-level Alignment**: Dual tokenization fusing user-item semantic and collaborative signals
- **Behavior Modeling-level Alignment**: Enhanced behavior modeling with bidirectional semantic alignment
- **Preference-level Alignment**: Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO)

### 🎯 Key Features

- **Multi-level Alignment**: Unified alignment across token, behavior, and preference levels
- **Progressive Training**: Innovative DPO strategy combining self-play and real-world feedback
- **Efficient Training**: Support for LoRA fine-tuning and DeepSpeed distributed training
- **Modular Design**: Clear module separation for easy extension and maintenance

## 🏗️ Project Structure

```
Align3GR/
├── README.md                    # Project documentation
├── data/                        # Data files
│   ├── Instruments/           # Instruments dataset
├── SFT/                        # Supervised Fine-Tuning module
│   ├── lora_finetune.py       # LoRA fine-tuning implementation
│   ├── mydata.py              # Data processing
│   ├── my_prompt.py           # Prompt templates
│   ├── collator.py            # Data collator
│   ├── evaluate.py            # Evaluation script
│   ├── generate_output.py     # Output generation
│   ├── utils.py               # Utility functions
│   ├── run_train.sh           # Training script
│   └── generate.sh            # Generation script
├── RL/                         # Reinforcement Learning module
│   ├── softmax_dpo.py         # DPO training implementation
│   ├── PNM.py                 # Progressive negative sample generation
│   ├── SM.py                  # Sentiment negative sample matching
│   ├── data4RF.py            # RL data preparation
│   ├── inference.py           # Inference script
│   ├── evaluate_batch.py      # Batch evaluation
│   ├── run_RF-dpo_training.sh # RF-DPO training script
│   ├── run_SP-dpo_training.sh # SP-DPO training script
│   ├── inference.sh           # Inference script
│   └── trainer/               # Trainer module
├── Token/                      # Token Alignment module
│   ├── main.py               # Main training script
│   ├── trainer.py            # Trainer implementation
│   ├── datasets.py           # Dataset classes
│   ├── generate_indices.py   # Index generation
│   ├── tokenize.sh           # Tokenization script
│   ├── train_tokenizer.sh    # Tokenizer training
│   └── models/               # Model definitions
└── config/                   # Configuration files
    ├── ds_z2_bf16.json      # DeepSpeed configurations
    ├── ds_z2_fp16.json
    ├── ds_z3_bf16.json
    ├── ds_z3_fp16.json
    ├── ds_z3_bf16_save16bit.json
    └── ds_z3_fp16_save16bit.json
```



## 📚 Usage Guide

### 1. Token Level Alignment (Tokenization)

```bash
bash Token/get_emb/process4cf.sh
bash Token/get_emb/train_din.sh

python Token/get_emb/semantic_embedding_item.py --plm_name t5 --plm_checkpoint_t5 sentence-transformers/sentence-t5-base --gpu_id 0
python Token/get_emb/semantic_embedding_user.py --plm_name t5 --plm_checkpoint_t5 sentence-transformers/sentence-t5-base --gpu_id 0

bash Token/get_emb/getting_user_cf.sh

bash Token/train_tokenizer.sh

bash Token/tokenize.sh
```

### 2. Behaviors Modeling Level Alignment (Multi-task SFT)

```bash
bash SFT/run_train.sh

bash SFT/generate_output.sh
```

### 3. Preference Level Alignment (Preference-based RL)

```bash

python RL/PNM.py 

bash RL/run_SP-dpo_training.sh

bash RL/inference_SP.sh

bash RL/data4RF.sh

python RL/SM.py

bash RL/run_RF-dpo_training.sh

bash RL/inference_RF.sh

```


## ⚙️ Configuration

The project provides various DeepSpeed configuration options:

- `ds_z2_*.json`: ZeRO-2 configuration for medium-scale training
- `ds_z3_*.json`: ZeRO-3 configuration for large-scale training
- `*_fp16.json`: FP16 precision for memory efficiency
- `*_bf16.json`: BF16 precision for better numerical stability

# Align3GR
