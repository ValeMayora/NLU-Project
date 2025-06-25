import math
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Weight Initialization ===
def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.RNN, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    ih_chunks = torch.chunk(param, 4, 0)
                    for chunk in ih_chunks:
                        nn.init.xavier_uniform_(chunk)
                elif 'weight_hh' in name:
                    hh_chunks = torch.chunk(param, 4, 0)
                    for chunk in hh_chunks:
                        nn.init.orthogonal_(chunk)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

# === Training Step ===
def train_loop(data_loader, model, optimizer, criterion, clip=5):
    model.train()
    loss_array = []
    token_counts = []

    for sample in data_loader:
        optimizer.zero_grad()
        source = sample['source'].to(DEVICE)
        target = sample['target'].to(DEVICE)

        output = model(source)
        loss = criterion(output,target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_array.append(loss.item() * sample['number_tokens'])
        token_counts.append(sample['number_tokens'])

    return sum(loss_array) / sum(token_counts)

# === Evaluation Step ===
def eval_loop(data_loader, model, criterion):
    model.eval()
    loss_array = []
    token_counts = []

    with torch.no_grad():
        for sample in data_loader:
            source = sample['source'].to(DEVICE)
            target = sample['target'].to(DEVICE)

            output = model(source)
            loss = criterion(output, target)

            loss_array.append(loss.item())
            token_counts.append(sample['number_tokens'])

    avg_loss = sum(loss_array) / sum(token_counts)
    ppl = math.exp(sum(loss_array) / sum(token_counts))  
    return ppl, avg_loss

# === Full Training Procedure ===
def train_model(model, train_loader, dev_loader, test_loader,
                criterion_train, criterion_eval, optimizer, config):
    
    results = {
        "best_model": None,
        "losses_train": [],
        "losses_dev": [],
        "ppls_dev": [],
        "sampled_epochs": [],
        "best_ppl": math.inf,
        "final_ppl": None
    }

    patience = config["patience_init"]
    n_epochs = config["n_epochs"]
    clip = config["clip"]
    pbar = tqdm(range(1, n_epochs + 1), desc="Training", leave=True)

    for epoch in pbar:
        start_time = time.time()

        train_loss = train_loop(train_loader, model, optimizer, criterion_train, clip)
        ppl_dev, loss_dev = eval_loop(dev_loader, model, criterion_eval)

        end_time = time.time()
        epoch_duration = end_time - start_time

        results["sampled_epochs"].append(epoch)
        results["losses_train"].append(train_loss)
        results["losses_dev"].append(loss_dev)
        results["ppls_dev"].append(ppl_dev)

        # Proper logging without interfering with the tqdm bar
        log_msg = (
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {loss_dev:.4f} | "
            f"Dev PPL: {ppl_dev:.2f} | "
            f"Time: {epoch_duration:.2f}s"
        )
        tqdm.write(log_msg)

                # with open(log_file, "a") as f:
        #     f.write(log_msg + "\n")
        #
        #
        # Early stopping
        if ppl_dev < results["best_ppl"]:
            results["best_ppl"] = ppl_dev
            results["best_model"] = copy.deepcopy(model).to("cpu")
            patience = config["patience_init"]
        else:
            patience -= 1
            if patience <= 0:
                tqdm.write("Early stopping.")
                break

    pbar.close()

    # Final test
    results["best_model"].to(DEVICE)
    results["final_ppl"], _ = eval_loop(test_loader, results["best_model"], criterion_eval)
    tqdm.write(f"Final Test PPL: {results['final_ppl']:.2f}")
    return results
