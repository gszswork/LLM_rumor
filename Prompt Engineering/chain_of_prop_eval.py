import re
import json
import argparse 
import os
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--dataset', type=str, default='Weibo', help='Dataset name')
parser.add_argument('--k', type=int, default=50, help='Number of comments in single prompt.')
args = parser.parse_args()

path_of_data = {'Twitter': './_2_chain_of_prop_data/twitter_dict.json', 'Weibo': './_2_chain_of_prop_data/weibo_dict.json'}
path_to_return = {'Twitter': './_2_cop_twitter_ret', 'Weibo': './_2_cop_weibo_ret'}
path_of_label = {'Twitter': './_2_chain_of_prop_data/twitter_label_dict.json', 'Weibo': './_2_chain_of_prop_data/weibo_label_dict.json'}

data_path = path_of_data[args.dataset]
returned_path = path_to_return[args.dataset]
with open(data_path, 'r') as f:
    data = json.load(f)

data_ids = data.keys()

with open(path_of_label[args.dataset]) as f:
    data_label = json.load(f)

pred_list, truth_list = [], []

for id in data_ids:
    pred_path = os.path.join(returned_path, id+'.json')
    with open(pred_path, 'r') as f:
        paragraph = json.load(f)
        print(paragraph)


    # Regular expression pattern
    pattern_fake = r'\bA\.\ Fake\b'
    pattern_true = r'\bB\.\ Real\b'

    # Check for matches
    if re.search(pattern_fake, paragraph):
        pred_list.append(1)
        truth_list.append(data_label[id])
        print("fake")
    if re.search(pattern_true, paragraph):
        print("real")
        pred_list.append(0)
        truth_list.append(data_label[id])
    

print(len(data_ids), len(pred_list))
print(classification_report(truth_list, pred_list, digits=4))
