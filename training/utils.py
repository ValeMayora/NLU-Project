import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# ---------- #
# 1. Read corpus and add <eos> to every sentence
# ---------- #

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# ---------- #
# 2. Lang class: vocab <-> ids
# ---------- #

class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

# ---------- #
# 3. Dataset class
# ---------- #

class PennTreeBank(Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            tokens = sentence.split()
            self.source.append(tokens[:-1])
            self.target.append(tokens[1:])

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {"source": src, "target": trg}

    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print(f"OOV found! Token: {x}")
                    break  # handle OOV case safely if needed
            res.append(tmp_seq)
        return res

# ---------- #
# 4. Collate function for dynamic padding
# ---------- #

def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 1
        padded = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded[i, :end] = seq
        return padded, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0]:
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item

# ---------- #
# 5. Dataloader creation function
# ---------- #

def get_ptb_dataloaders(data_dir, batch_size=64):
    train_raw = read_file(f"{data_dir}/ptb.train.txt")
    valid_raw = read_file(f"{data_dir}/ptb.valid.txt")
    test_raw  = read_file(f"{data_dir}/ptb.test.txt")

    lang = Lang(train_raw, special_tokens=["<pad>", "<eos>"])
    pad_token = lang.word2id["<pad>"]

    train_dataset = PennTreeBank(train_raw, lang)
    valid_dataset = PennTreeBank(valid_raw, lang)
    test_dataset  = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=partial(collate_fn, pad_token=pad_token), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2*batch_size,
                              collate_fn=partial(collate_fn, pad_token=pad_token))
    test_loader  = DataLoader(test_dataset, batch_size=2*batch_size,
                              collate_fn=partial(collate_fn, pad_token=pad_token))

    return train_loader, valid_loader, test_loader, lang
