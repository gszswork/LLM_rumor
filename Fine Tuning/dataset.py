import os
import json
import torch
from datasets import Dataset, DatasetDict
from random import shuffle

def load_data(datasetname, path):
    if datasetname == 'mc-fake':
        train_dict, valid_dict, test_dict = load_mc_fake(datasetname, path)
    else: 
        NotImplementedError
    #print(type(train_dict['label'][0]))
    #print(train_dict['label'])
    hf_dataset = DatasetDict({
        'train': Dataset.from_dict(train_dict),
        'valid': Dataset.from_dict(valid_dict),
        'test': Dataset.from_dict(test_dict)
    })
    
    return hf_dataset


def load_mc_fake(datasetname, path):
    data_json = json.load(open(os.path.join(path, datasetname + ".json"), "r"))
    length_of_data = len(data_json)

    # check the existence of the split
    if os.path.exists(os.path.join(path, datasetname + "_train.json")):
        train_ids = json.load(open(os.path.join(path, datasetname + "_train.json"), "r"))
        valid_ids = json.load(open(os.path.join(path, datasetname + "_valid.json"), "r"))
        test_ids = json.load(open(os.path.join(path, datasetname + "_test.json"), "r"))
    else:
        all_ids = list(range(length_of_data))
        shuffle(all_ids)
        train_ids = all_ids[:int(length_of_data * 0.7)]
        valid_ids = all_ids[int(length_of_data * 0.7):int(length_of_data * 0.8)]
        test_ids = all_ids[int(length_of_data * 0.8):]
        # save the ids
        with open(os.path.join(path, datasetname + "_train.json"), "w") as f:
            json.dump(train_ids, f)
        with open(os.path.join(path, datasetname + "_valid.json"), "w") as f:
            json.dump(valid_ids, f)
        with open(os.path.join(path, datasetname + "_test.json"), "w") as f:
            json.dump(test_ids, f)

    # 7:1:2 split
    train_data = {'text': [data_json[str(i)]['title'] + ' (Date: '+ data_json[str(i)]['date'] +') '+ data_json[str(i)]['text'] for i in train_ids],
                  'label': [data_json[str(i)]['label'] for i in train_ids]}

    valid_data = {'text': [data_json[str(i)]['title'] + ' (Date: '+ data_json[str(i)]['date'] +') '+ data_json[str(i)]['text'] for i in valid_ids],
                  'label': [data_json[str(i)]['label'] for i in valid_ids]}

    test_data = {'text': [data_json[str(i)]['title'] + ' (Date: '+ data_json[str(i)]['date'] +') '+ data_json[str(i)]['text'] for i in test_ids],
                 'label': [data_json[str(i)]['label'] for i in test_ids]}
    
    return train_data, valid_data, test_data


# if __name__ == '__main__':
#     mc_fake_dataset = load_data("mc-fake", "./data")
#     mc_fake_train = mc_fake_dataset['train']
#     print(mc_fake_train[0])
    