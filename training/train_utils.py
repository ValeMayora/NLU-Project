
import torch
import torch.nn.functional as F
import yaml

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compute_perplexity(logits, targets):
    # logits: [batch * seq_len, vocab_size]
    # targets: [batch * seq_len]
    loss = F.cross_entropy(logits, targets, ignore_index=0)  # ignore padding
    return torch.exp(loss).item()
