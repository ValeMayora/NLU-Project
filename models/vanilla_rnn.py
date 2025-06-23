import torch
import torch.nn as nn

class VanillaRNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(VanillaRNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0 for <pad> token
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        x: [batch_size, seq_length]
        hidden: [num_layers, batch_size, hidden_dim]
        """
        embed = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embed, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

