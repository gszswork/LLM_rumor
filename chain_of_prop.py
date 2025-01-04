# Reproduce experiments for the ArXiv-2024 paper: <Can Large Language Models Detect Rumors on Social Media? >
# 1. Vanilla prompt: news content + "Verify the credibility of the news."
# 2. Rational prompt: news content + "Based on the writing style and the commensense knowledge, estimate the credibility of the news."
# 3. Vanilla prompt with comment: "Based on the comments, verify the authenticity of the news"
# 4. Conflicting prompt with comment: "Based on the comments, analyze whether there are any rebuttals or conflicts, and accordingly verify the authenticity of the news. "
# 5. Chain of prompt: 100 comments each prompt, combination of rational prompt and conflicting prompt with comments. 

import numpy as np
import json
import os
from openai import OpenAI
from sklearn.metrics import classification_report
import argparse 
client = OpenAI()
import logging 
from tqdm import tqdm
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def prune_long_text(text, max_token_length=2048):
    tokenized_text = tokenizer(text, max_length=max_token_length, truncation=True, return_tensors="pt")
    truncated_text = tokenizer.decode(tokenized_text["input_ids"][0])
    return truncated_text



from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


# completion_with_backoff(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Once upon a time,"}])


parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--dataset', type=str, default='Weibo', help='Dataset name')
parser.add_argument('--k', type=int, default=50, help='Number of comments in single prompt.')
args = parser.parse_args()

logging.basicConfig(filename="output_logs."+args.dataset+".txt", level=logging.INFO, format="%(asctime)s - %(message)s")

path_of_data = {'Twitter': './_2_chain_of_prop_data/twitter_dict.json', 'Weibo': './_2_chain_of_prop_data/weibo_dict.json'}
path_to_return = {'Twitter': './_2_cop_twitter_ret', 'Weibo': './_2_cop_weibo_ret'}

path = path_of_data[args.dataset]
with open(path, 'r') as f:
    data = json.load(f)
returned_path = path_to_return[args.dataset]
if not os.path.exists(returned_path):
      os.mkdir(returned_path)

# chain of prompt
def get_prompt(news, comments): 
    prompt_texts = 'There is a piece of news: '+ news + 'There are comments for the news: ' + comments + 'You need to do: \
    (1) Based on the writing style and the commonsense knowledge, estimate the credibility of the news. \
    (2) Based on the comments, analyze whether there are any rebuttals or conflicts, and then accordingly verify the authenticity of the news. \
        Based on above results, please choose the answer from the following options: A. Fake, B. Real.'    
    return prompt_texts

def merge_list(list, idx1, idx2):
    sub_list = list[idx1: idx2+1]
    merged_string = ', '.join(f'"{s}"' for s in sub_list)
    return merged_string

for id, val in tqdm(data.items()):
    if os.path.exists(os.path.join(returned_path, id+'.json')):
        print("file ", id, 'already checked, skipped.')
        continue

    news_content = val['news']
    comment_list = val['comments'][:300]

    if len(comment_list) == 0:
        continue
    
    returned_msg = None
    for i in range(0, len(comment_list), args.k):
        if i == 0:
            merged_string = merge_list(comment_list, i, max(i+args.k, len(comment_list)))
            merged_string = prune_long_text(merged_string)
            completion = completion_with_backoff(
                model = 'gpt-3.5-turbo',
                messages=[
                    {'role': 'user', 
                    "content": get_prompt(news_content, merged_string)}
                ]
            )
        else: 
            merged_string = merge_list(comment_list, i, max(i+args.k, len(comment_list)))
            merged_string = prune_long_text(merged_string)
            completion = completion_with_backoff(
                model = 'gpt-3.5-turbo',
                messages=[
                    {'role': 'developer', 'content': 'This is the analysis of a piece of news\'s veracity in last round: ' + returned_msg.content + '.\
                        Your task is to predict the veracity of the news given the information in last round and this round as follows: '}, 
                    {'role': 'user', 
                    "content": get_prompt(news_content, merged_string)}
                ]
            )

        returned_msg = completion.choices[0].message

    logging.info(id)
    logging.info(returned_msg.content)

    with open(os.path.join(returned_path, id+'.json'), 'w') as json_file:
        json.dump(returned_msg.content, json_file)


