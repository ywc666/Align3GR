import argparse
import json
import os

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from tqdm import tqdm



PROMPT_TEMPLATE = """You are a sentiment analysis expert. 
Given a user's profile, the item title, and the user's review of the item, assign a sentiment score from 1 to 5. 
Consider the user’s background, the context of the item, and the tone of the review. 
Return **only the score**. 

Scoring guide:
1 = dislike  
2 = slightly dislike  
3 = neutral  
4 = slightly like  
5 = like  

Here are examples:
User Name: francisco  
Item Title: T-shirt  
Review: I dislike it  
Score: 1

User Name: alice  
Item Title: Laptop  
Review: It works fine for me  
Score: 4

User Name: bob  
Item Title: Coffee Mug  
Review: Just okay  
Score: 3

Now analyze the following:
User Name: {user_name}  
Item Title: {item_title}  
Review: {review}  
Score:"""


# prompt = PROMPT_TEMPLATE.format(**inputs)
# pipeline_ins = pipeline(task=Tasks.text_generation, model='/home/ecs-user/nas_original_data/csh/MODEL/iic/nlp_ecomgpt_multilingual-7B-ecom',model_revision='master', external_engine_for_llm=False)
# print(pipeline_ins(prompt))

# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import base64

class DeepSeek:
    def __init__(self):
        # self.client = OpenAI(api_key="sk-769fbd3a8cea40dc93cfcac16100f652", base_url="https://api.deepseek.com")
        self.client = OpenAI(api_key="", base_url="https://api.modelscope.cn/v1")

    def get_response(self, image_path,user_prompt):
        if image_path:
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",  #  "https://kexue.fm/usr/uploads/2022/07/3685027055.jpeg"
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        response = self.client.chat.completions.create(
            # model="deepseek-chat",
            model="emogpt-32b-instruct",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class EcomGPT:
    def __init__(self):
        self.pipeline = pipeline(task=Tasks.text_generation, model='iic/nlp_ecomgpt_multilingual-7B-ecom',model_revision='master', external_engine_for_llm=False)

    def get_response(self, image_path,user_prompt):
        if image_path:
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",  #  "https://kexue.fm/usr/uploads/2022/07/3685027055.jpeg"
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        response = self.pipeline(messages)
        return response


def main(args):
    model = EcomGPT() 
    # model = DeepSeek()

    # read reviews
    review_data_path = args.review_data_path
    item2id_path = args.item2id_path
    user2id_path = args.user2id_path
    user2SCID_path = args.user2SCID_path
    item2SCID_path = args.item2SCID_path
    item_title_path = args.item_title_path
    item2id = {}
    id2item = {}
    user2id = {}
    id2user = {}
    res = {}
    with open(item_title_path, 'r') as f:
        item_title = json.load(f)
    with open(review_data_path, 'r') as f:
        review_data = [json.loads(line) for line in f]
    with open(item2id_path, 'r') as f:
        for line in f:
            item_name, item_id = line.strip().split('\t')
            item2id[item_name] = item_id
            id2item[item_id] = item_name
    with open(user2id_path, 'r') as f:
        for line in f:
            user_name, user_id = line.strip().split('\t')
            user2id[user_name] = user_id
            id2user[user_id] = user_name
    with open(user2SCID_path, 'r') as f:
        user2SCID = json.load(f)
    with open(item2SCID_path, 'r') as f:
        item2SCID = json.load(f)
    if os.path.exists(item_title_path.replace('.item.json', '.RF.json')):
        # 读取已经有的数据
        with open(item_title_path.replace('.item.json', '.RF.json'), 'r') as f:
            res = json.load(f)

    for review in tqdm(review_data):
        if 'reviewText' not in review:
            continue
        if 'reviewerName' not in review:
            continue
        if 'asin' not in review:
            continue
        if 'title' not in  item_title[item_id]:
            continue
        user_name = review['reviewerID']
        item_name = review['asin']
        if user_name not in user2id or item_name not in item2id:
            continue
        user_id = user2id[user_name]
        item_id = item2id[item_name]
        if user_id+'review'+item_id in res:
            continue
        review_text = review['reviewText']
        inputs = {
            'user_name': review['reviewerName'],
            'item_title': item_title[item_id]['title'],
            'review': review_text,
        }
        user_prompt = PROMPT_TEMPLATE.format(**inputs)
        
        # print(model.get_response(None,user_prompt))
        response = model.get_response(None,user_prompt)
        user_scid =  "".join([prefix for prefix in user2SCID[user_id]])
        item_scid =  "".join([prefix for prefix in item2SCID[item_id]])
        res[user_scid+'review'+item_scid] = {
            'user_name': review['reviewerName'],
            'item_title': item_title[item_id]['title'],
            'review': review_text,
            'response': response,
        }
        if len(res) % 20 == 0:
            with open(item_title_path.replace('.item.json', '.RF.json'), 'w') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_data_path", type=str, default='Align3GR/data/Instruments/Musical_Instruments_5.json')
    parser.add_argument("--item2id_path", type=str, default='Align3GR/data/Instruments/Instruments.item2id')
    parser.add_argument("--user2id_path", type=str, default='Align3GR/data/Instruments/Instruments.user2id')
    parser.add_argument("--item_title_path", type=str, default='Align3GR/data/Instruments/Instruments.item.json')
    parser.add_argument("--user2SCID_path", type=str, default='Align3GR/data/Instruments/Instruments.user.index.json')
    parser.add_argument("--item2SCID_path", type=str, default='Align3GR/data/Instruments/Instruments.item.index.json')
    args = parser.parse_args()
    main(args)

    # user_prompt = "You are a sentiment analysis expert. Given a user\'s profile, the item title, and the user\'s review of the item, assign a sentiment score from 1 to 5. Consider the user’s background, the context of the item, and the tone of the review. Return only the score."

