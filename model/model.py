import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

class QModel(nn.Module):
    def __init__(self, model_name, tokenizer_name):
        super(QModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(self, input_ids, max_length=50):
        return self.model.generate(input_ids=input_ids, max_length=max_length)