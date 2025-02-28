import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn




class LlamaForClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )
        self.classifier = nn.Linear(self.llm.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.llm.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # Extract first token's hidden state (CLS-like token for classification)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}