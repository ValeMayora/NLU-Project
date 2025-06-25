import torch.nn as nn
import torch.nn.functional as F

class LM_LSTM_A(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, num_layers, use_dropout, dropout=0.0,
                  pad_index=0):
        super(LM_LSTM_A, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_index)


        self.use_dropout = use_dropout

        if self.use_dropout:
            self.emb_dropout = nn.Dropout(dropout)
            self.out_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_size, hid_size, num_layers, batch_first=True)
        
        self.output = nn.Linear(hid_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)

        if self.use_dropout: 
            x = self.emb_dropout(x)

        output, _ = self.lstm(x)
        if self.use_dropout:
            output = self.drop_lstm(output)

        logits = self.output(output)
        logits = logits.permute(0, 2, 1)
        return logits
