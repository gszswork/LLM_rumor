# The main code to run pure prompt engineering to debunk fake news. 
#  (1) Pure Prompting
#  (2) CoT (chain-of-thought)
# Reference: 
import os
import openai
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()
from prompts import PROMPTS
from load_data import *
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

returned_path = './liar_ret'

liar_new = load_liar_new()
liar_ids = liar_new.keys()

@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

for id in tqdm(liar_ids):
    if os.path.exists(os.path.join(returned_path, str(id)+'.json')):
        print("file ", id, 'already checked, skipped.')
        continue
    p = PROMPTS['CoT_prompt'].format(news_title=liar_new[id]['title'],
                                     news_article=liar_new[id]['article'],
                                     date=liar_new[id]['date'])
    # print(p)

        # Make the API call to OpenAI's GPT-4 model
    completion = completion_with_backoff(
        model = 'o3-mini',
        messages=[
            {'role': 'user', 
            "content": p}
        ]
    )

    # Extract the model's response
    returned_msg = completion.choices[0].message
    print(returned_msg)
    with open(os.path.join(returned_path, str(id)+'.json'), 'w') as json_file:
        json.dump(returned_msg.content, json_file)



