from transformers import AutoTokenizer
from dataset import load_dataset
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from peft import LoraConfig, get_peft_model, TaskType
from model import LlamaForClassification
from transformers import TrainingArguments, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser example")
    
    parser.add_argument("--dataset", type=str, default="mc-fake", help="Dataset name")
    # Add arguments
    # parser.add_argument("--input", type=str, required=True, help="Path to input file")
    # parser.add_argument("--output", type=str, default="output.txt", help="Path to output file")
    # parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization")
    # parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()
args = parse_args()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(args.dataset)
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels into tensors
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Load model
num_labels = 2  # MC-Fake News has 2 categories
model = LlamaForClassification(MODEL_NAME, num_labels).to("cuda")

# Apply LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  
    lora_dropout=0.1,  
    bias="none",  
    task_type=TaskType.SEQ_CLS  # Sequence classification
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()


# TODO: Load the trained model and apply to the test set. 
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction

