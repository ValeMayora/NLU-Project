import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
import matplotlib.pyplot as plt
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vanilla_rnn import VanillaRNNLanguageModel
from training.train_utils import init_weights, train_model, eval_loop
from training.utils import load_config, get_ptb_dataloaders

def main():
    # Load Config
    config = load_config("configs/vanilla_rnn.yaml")

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    # Load Data
    train_loader, dev_loader, test_loader, lang = get_ptb_dataloaders(
        data_dir="data/PennTreeBank",
        batch_size=train_cfg["batch_size"],
    )
    vocab_size = len(lang.word2id)
    model_cfg["vocab_size"] = vocab_size

    # Initialize Model
    model = VanillaRNNLanguageModel(
        output_size=model_cfg["vocab_size"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"]
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(model)

    init_weights(model)

    # Optimizer & Loss
    pad_index = lang.word2id["<pad>"]
    optimizer = optim.SGD(model.parameters(), lr=train_cfg["learning_rate"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        criterion_train=criterion_train,
        criterion_eval=criterion_eval,
        optimizer=optimizer,
        config=train_cfg
    )

    # Save best model
    os.makedirs(output_cfg["save_path"], exist_ok=True)
    model_save_path = os.path.join(output_cfg["save_path"], "best_model_vanilla.pt")
    torch.save(results["best_model"].state_dict(), model_save_path)

    # Save logs
    os.makedirs(output_cfg["log_path"], exist_ok=True)
    metrics_path = os.path.join(output_cfg["log_path"], "metrics_vanilla.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train_loss": results["losses_train"],
            "dev_loss": results["losses_dev"],
            "dev_ppl": results["ppls_dev"],
            "final_test_ppl": results["final_ppl"]
        }, f, indent=2)

    print(f"Best validation PPL: {results['best_ppl']:.2f}")
    print(f"Final test PPL: {results['final_ppl']:.2f}")

    # Plot perplexity curves
    plt.plot(results["ppls_dev"], label='Dev PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Vanilla RNN Perplexity')
    plt.legend()
    plt.savefig(os.path.join(output_cfg["log_path"], "ppl_curve_vanilla.png"))
    plt.close()

if __name__ == "__main__":
    print("Starting Vanilla RNN training...")
    main()

