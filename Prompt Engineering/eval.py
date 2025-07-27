from sklearn.metrics import classification_report
from load_data import load_liar_new
import os
import json
import re
# liar-labels 

liar_new = load_liar_new()

true_labels = ['true', 'mostly-true']
false_labels = ['pants-fire', 'false', 'barely-true', 'half-true']

label_count = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 0,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0
}

gtruth, pred = [], []

for id, val in liar_new.items():
    label_count[val['label']] += 1
    if val['label'] in true_labels:
        gtruth.append(0)
    elif val['label'] in false_labels:
        gtruth.append(1)
    else:
        assert 'Missing label error.'

assert len(gtruth)==len(liar_new)
#  Ground truth label count. 
print("Label counts:")
for label, count in label_count.items():
    print(f"{label}: {count}")


def fix_broken_json(json_str):
    # Fix unescaped double quotes inside values
    json_str = re.sub(r'(?<!\\)"([^"]*?)"([^"]*?)"([^"]*?)(?<!\\)"', r'"\1\2\3"', json_str)
    return json_str

def load_json_safe(json_str):
    try:
        return json.loads(json_str)  # Try loading JSON
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Attempting to fix...")
        
        fixed_json_str = fix_broken_json(json_str)
        
        try:
            return json.loads(fixed_json_str)  # Try again after fixing
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON: {e}")
            return None  # Return None if still broken

# gpt_3.5_turbo -> liar_new
# ret_path = './gpt_3.5_turbo_liar_new_rets'
# gpt_o3_mini -> liar new
ret_path = './gpt_o3_mini_liar_new_rets'

for id in liar_new.keys():

    with open(os.path.join(ret_path, str(id)+'.json'), 'r', encoding='utf-8') as f:
        #print(id)
        raw_content = f.read()
        json_file = json.loads(raw_content).strip()
        #print(json_file)
        if json_file[0] != '{':
            json_file = "{" + json_file.split('{')[1]
        json_file = json.loads(json_file, strict=False)
        #print(type(json_file))
        #print(type(line), type(json_file), json_file)
        #print(json_file['prediction'], type(json_file['prediction']))
        
        if json_file['prediction'] == 'True':
            pred.append(0)
        else:
            pred.append(1)
            # print(id, 'label inconsistency', json_file['prediction'])
            
print(len(gtruth), len(pred))
assert len(gtruth) == len(pred)
print(classification_report(gtruth, pred, digits=4))