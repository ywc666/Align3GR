# 语义encoder
import json
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel


def load_model_and_tokenizer(args):

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    if args.plm_name.lower() == 'llama':
        model_path = args.plm_checkpoint_llama
        print(f"Loading LLaMA model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
            
    elif args.plm_name.lower() == 't5':
        model_path = args.plm_checkpoint_t5
        print(f"Loading T5 model from {model_path}")
        model = SentenceTransformer(model_path)
        tokenizer = None
            
    else:
        raise ValueError(f"Unsupported model type: {args.plm_name}. Supported types: 'llama', 't5'")
    

    model.eval()
    model = model.to(device)
    
    return tokenizer, model, device


def load_data(args):
    meta_data_file = f'data/Instruments/meta_Musical_Instruments.json' # the data is large, so we don't store it in the repo
    item_ids_file = f'data/Instruments/Instruments.item2id'
    
    item_ids = {}
    with open(item_ids_file, 'r') as f:
        for line in f:
            item_id, item_name = line.strip().split('\t')
            item_ids[item_name] = int(item_id)

    item_dict = {}
    
    bnu = 0
    with open(meta_data_file, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['asin'] in item_ids.keys():
                bnu += 1
                description = ''
                for des in data['description']:
                    description += des + ','
                item_dict[item_ids[data['asin']]] = data['title'] + '\n' + description + '\n' + data['brand'] + '\n' + data['price']
    

    item_dict = sorted(item_dict.items(), key=lambda x: x[0])
    item_dict = dict(item_dict)
    
    return item_dict

def generate_item_embedding(item_dict, tokenizer, model, device, model_type, max_sent_len=2048):

    print(f'Generate Text Embedding using {model_type.upper()}...')
    

    items = sorted(item_dict.keys())
    order_texts = []
    for item in items:
        order_texts.append([item_dict[item]])  # each item's text as a field
    
    embeddings = []
    batch_size = 1
    
    for start in tqdm(range(0, len(order_texts), batch_size), desc="Processing items"):
        if (start + 1) % 100 == 0:
            print(f"==> {start + 1}")
            
        field_texts = order_texts[start:start + batch_size]
        field_texts = list(zip(*field_texts)) 
        
        field_embeddings = []
        for sentences in field_texts: 
            sentences = list(sentences)
            

            if model_type.lower() == 'llama':

                encoded_sentences = tokenizer(
                    sentences, 
                    max_length=max_sent_len, 
                    truncation=True, 
                    return_tensors='pt', 
                    padding="longest"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=encoded_sentences.input_ids, 
                        attention_mask=encoded_sentences.attention_mask
                    )
                

                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()

            elif model_type.lower() == 't5':
                mean_output = model.encode(sentences)
                mean_output = torch.FloatTensor(mean_output)
            field_embeddings.append(mean_output)

        field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
        embeddings.append(field_mean_embedding)
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    
    return embeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='llama', help='llama / t5')
    parser.add_argument('--plm_checkpoint_llama', type=str, default='huggyllama/llama-7b')
    parser.add_argument('--plm_checkpoint_t5', type=str, default='sentence-transformers/sentence-t5-base')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


def main():
    args = parse_args()
    

    tokenizer, model, device = load_model_and_tokenizer(args)


    item_dict = load_data(args)
    

    embeddings = generate_item_embedding(item_dict, tokenizer, model, device, args.plm_name, args.max_sent_len)
    

    output_filename = f'{args.dataset}/item_embeddings_{args.plm_name}.npy'
    np.save(output_filename, embeddings)
    print(f"Embeddings saved to: {output_filename}")

if __name__ == "__main__":
    main()









