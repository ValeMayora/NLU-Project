
import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        use_dropout_embed=False,
        use_dropout_output=False,
        dropout_embed=0.0,
        dropout_output=0.0
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_dropout_embed = use_dropout_embed
        self.use_dropout_output = use_dropout_output

        self.dropout_embed = nn.Dropout(dropout_embed) if use_dropout_embed else nn.Identity()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout_output = nn.Dropout(dropout_output) if use_dropout_output else nn.Identity()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        x = self.dropout_embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout_output(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))
