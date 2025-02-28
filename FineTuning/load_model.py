from transformers import AutoTokenizer
from dataset import load_dataset
import argparse
import torch
from transformers import AutoModelForCausalLM
from torch import nn

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
