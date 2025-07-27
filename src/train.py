import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.Pas_Dataset import PasDataset
from src.models.POLYNET import POLYNET
import matplotlib.pyplot as plt
import random
import pandas as pd

def evaluate(model, loader, device):
    print("Evaluating model...")
    model.eval()
    ys, ps = [], []
    losses = []
    lossf = nn.BCELoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).float()
            preds = model(xb)
            ys.extend(yb.cpu().numpy())
            ps.extend(preds.cpu().numpy())
            losses.append(lossf(preds, yb).item() * xb.size(0))
    avg_loss = sum(losses) / len(ys) if ys else 0.0
    return roc_auc_score(ys, ps), average_precision_score(ys, ps), avg_loss


def train_and_evaluate(
    train_files, val_files, test_files,
    batch_size, lr, epochs
):
    print("Loading data and preparing datasets...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets and loaders
    train_ds = PasDataset(train_files, [1, 0])
    val_ds = PasDataset(val_files, [1, 0])
    test_ds = PasDataset(test_files, [1, 0])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Initializing model and optimizer...")
    model = POLYNET().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCELoss()

    train_losses = []
    val_losses = []

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} - Training...")
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad()
            preds = model(xb)
            loss = lossf(preds, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} - Validating...")
        val_auc, val_auprc, val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_AUROC {val_auc:.4f} | val_AUPRC {val_auprc:.4f}")

    print("Training complete. Plotting loss curves...")
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/loss_curve.png')
    plt.close()
    print("Loss curves saved to models/loss_curve.png")

    print("Evaluating on test set...")
    test_auc, test_auprc, _ = evaluate(model, test_loader, device)
    print("Saving trained model to models/POLYNET.pt...")
    torch.save(model.state_dict(), "models/POLYNET.pt")
    print("Model saved.")
    return test_auc, test_auprc, train_losses, val_losses, val_auc

hyper_param_space = {
        "batch_size": [64,128],
        "lr": [1e-3, 1e-4, 1e-2],
        "epochs": [6,8,10]
    }
def random_hyper_params():
    return {param: random.choice(values) for param, values in hyper_param_space.items()}
if __name__ == "__main__":
    print("Starting POLYNET training script...")
    train_files = ["src/data/processed/pos_201_train.fa", "src/data/processed/neg_201_train.fa"]
    val_files   = ["src/data/processed/pos_201_val.fa",   "src/data/processed/neg_201_val.fa"]
    test_files  = ["src/data/processed/pos_201_test.fa",  "src/data/processed/neg_201_test.fa"]
    model_outputs = []
    for i in range(10):
        print(f"Running experiment {i+1}...")
        hyper_params = random_hyper_params()
        test_auc, test_auprc, train_losses, val_losses, val_auc = train_and_evaluate(
            train_files, val_files, test_files, hyper_params["batch_size"], hyper_params["lr"], hyper_params["epochs"]
        )
        result = {
            'hyper_params': hyper_params,
            'test_auc': test_auc,
            'test_auprc': test_auprc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_auc': val_auc
        }
        model_outputs.append(result)
    df = pd.DataFrame(model_outputs)
    df.to_csv('models/model_outputs.csv', index=False)
    print("Model outputs saved to models/model_outputs.csv")
    
        







