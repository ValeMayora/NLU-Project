import torch
import torch.nn as nn

class VanillaRNNLanguageModel(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(VanillaRNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)  # padding_idx=0 for <pad> token
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        emb = self.embedding(x)
        if self.dropout:
            emb = self.dropout(emb)

        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)

        return output

