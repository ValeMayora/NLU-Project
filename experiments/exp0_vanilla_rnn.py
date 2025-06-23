
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ptb_dataset import get_ptb_dataloaders
from models.vanilla_rnn import VanillaRNNLanguageModel
from training.train_utils import compute_perplexity

# Set seed
torch.manual_seed(42)

# Load config
with open("configs/vanilla_rnn.yaml", "r") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
train_cfg = config["training"]
data_cfg = config["data"]
output_cfg = config["output"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, valid_loader, test_loader, vocab = get_ptb_dataloaders(
    data_dir="data/PennTreeBank",
    batch_size=train_cfg["batch_size"],
    seq_len=data_cfg["sequence_length"]
)

# Update vocab size dynamically
vocab_size = len(vocab)
model_cfg["vocab_size"] = vocab_size

# Initialize model
model = VanillaRNNLanguageModel(
    vocab_size=model_cfg["vocab_size"],
    embedding_dim=model_cfg["embedding_dim"],
    hidden_dim=model_cfg["hidden_dim"],
    num_layers=model_cfg["num_layers"],
    dropout=model_cfg.get("dropout", 0.0)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=train_cfg["learning_rate"])

# Create output dirs
os.makedirs(output_cfg["save_path"], exist_ok=True)
os.makedirs(output_cfg["log_path"], exist_ok=True)
log_file = os.path.join(output_cfg["log_path"], "exp0_vanilla_train_log.txt")


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.squeeze(0).to(device), targets.squeeze(0).to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            num_tokens = torch.count_nonzero(targets).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train():
    best_val_ppl = float('inf')
    patience = 3
    patience_counter = 0

    train_ppls = []
    val_ppls = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.squeeze(0).to(device), targets.squeeze(0).to(device)

            hidden = model.init_hidden(inputs.size(0), device)
            hidden = hidden.detach()

            optimizer.zero_grad()
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        val_loss, val_ppl = evaluate(model, valid_loader)

        epoch_duration = time.time() - epoch_start_time
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # Logging
        log_msg = (
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f}, PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f}, PPL: {val_ppl:.2f} | "
            f"Time: {epoch_duration:.2f}s"
        )
        #log_msg = f"Epoch {epoch:02d} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            model_path = os.path.join(output_cfg["save_path"], "vanilla_rnn_best.pt")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Final test evaluation
    test_loss, test_ppl = evaluate(model, test_loader)
    final_msg = f"\nFinal Test Loss: {test_loss:.4f}, Test Perplexity: {test_ppl:.2f}"
    print(final_msg)
    with open(log_file, "a") as f:
        f.write(final_msg + "\n")

    # Plot and save perplexity curves
    plt.plot(train_ppls, label='Train PPL')
    plt.plot(val_ppls, label='Val PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Vanilla RNN Perplexity')
    plt.savefig(os.path.join(output_cfg["log_path"], "ppl_curve.png"))
    plt.close()


if __name__ == "__main__":
    print("Starting Vanilla RNN training...")
    train()

