import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm_A import LM_LSTM_A
from training.train_utils import init_weights, train_model, eval_loop
from training.utils import load_config, get_ptb_dataloaders#, build_vocab, set_seed

def main():
    # Load Config
    config = load_config("configs/lstm_config.yaml")

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    # Load Data
    train_loader, dev_loader, test_loader, lang = get_ptb_dataloaders(
    data_dir="data/PennTreeBank",
    batch_size=train_cfg["batch_size"],
)
    vocab = len(lang.word2id)

    

    # Model 
    model = LM_LSTM_A(
        emb_size=model_cfg["emb_size"],
        hid_size=model_cfg["hid_size"],
        vocab_size=vocab,
        num_layers=model_cfg["num_layers"],
        use_dropout=model_cfg["use_dropout"],
        dropout=model_cfg["dropout"],
        pad_index=lang.word2id["<pad>"]
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(model)

    init_weights(model)

    # Optimizer & Loss
    optimizer = optim.SGD(model.parameters(), lr=train_cfg["learning_rate"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
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

    # Save Best Model
    os.makedirs("results", exist_ok=True)
    model_save_path = os.path.join(output_cfg["save_path"], "best_model_lstm.pt")
    torch.save(results["best_model"].state_dict(), model_save_path)

    # Save Logs
    with open("results/metrics_lstmSGD.json", "w") as f:
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
    plt.savefig(os.path.join(output_cfg["log_path"], "ppl_curve_lstm.png"))
    plt.close()


if __name__ == "__main__":
    ("Starting LSTM trining...")
    main()
