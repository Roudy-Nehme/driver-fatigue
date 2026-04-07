import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FatigueSequenceDataset, pad_collate_fn
from src.model import TemporalGRUClassifier
from src.feature_utils import FEATURE_NAMES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(manifest_csv, batch_size=16, num_workers=2):
    train_dataset = FatigueSequenceDataset(manifest_csv, split="train")
    val_dataset = FatigueSequenceDataset(manifest_csv, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def batch_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    return acc


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for x, lengths, y in progress_bar:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x, lengths)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        acc = batch_accuracy(logits.detach(), y)

        total_loss += loss.item()
        total_acc += acc

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    for x, lengths, y in progress_bar:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        acc = batch_accuracy(logits, y)

        total_loss += loss.item()
        total_acc += acc

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, val_loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "feature_names": FEATURE_NAMES,
        },
        save_path,
    )


def main():
    config = {
        "seed": 42,
        "manifest_csv": "/content/driver_fatigue_project/driver_fatigue_colab/data/manifest.csv",
        "batch_size": 16,
        "num_workers": 2,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "bidirectional": False,
        "learning_rate": 1e-3,
        "epochs": 10,
        "checkpoint_path": "/content/driver_fatigue_project/driver_fatigue_colab/checkpoints/best_model.pt",
    }

    set_seed(config["seed"])
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(
        manifest_csv=config["manifest_csv"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = TemporalGRUClassifier(
        input_size=len(FEATURE_NAMES),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_loss,
                save_path=config["checkpoint_path"],
            )
            print(f"Saved best model to: {config['checkpoint_path']}")


if __name__ == "__main__":
    main()
