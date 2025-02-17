import torch
import os
import argparse
import loralib as lora
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset import load_dataset
import os
from peft import LoraConfig, get_peft_model, TaskType



def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a language model for classification using LoRA')
    parser.add_argument('--base_model', type=str, default="crumb/nano-mistral", help='Base model to fine-tune (default: crumb/nano-mistral)')
    parser.add_argument('--dataset_name', type=str, default="mc-fake", help='Name of the dataset to use (default: mc-fake)')
    parser.add_argument('--cache_dir', type=str, default="./model_cache", help='Directory to cache the downloaded models (default: ./model_cache)')
    return parser.parse_args()
args = parse_args()

dataset = load_dataset(args.dataset_name, "./data")

# Load model and tokenizer for sequence classification
tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.base_model,num_labels=2,cache_dir=args.cache_dir)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None
    )
print(dataset["train"].column_names)
# Process dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  
    lora_dropout=0.1,  
    bias="none",  
    task_type=TaskType.SEQ_CLS  # Sequence classification
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_finetuned_llama_cls",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    push_to_hub=False,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

checkpoint_path = './checkpoints/lora_params'+args.base_model+'.pt'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(lora.lora_state_dict(model), checkpoint_path)
