import sys
import os
import argparse

# Add the current project's src directory to the front of the path
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_dir = os.path.dirname(current_dir)  # src directory
sys.path.insert(0, src_dir)

import pydra
from pydra import REQUIRED, Config

import wandb
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

from infra.action_encoder import Encoder
from data.replay_buffer import ReplayBuffer

# -----------------------------------------------------------------------------------
# Load datasets once at module load
# -----------------------------------------------------------------------------------

def load_datasets(window_len, transform):
    train_dataset = ReplayBuffer(
        root_dir=f"/home/ubuntu/slippify/data_window_{window_len}/train/",
        transform=transform,
        window_len=window_len
    )
    val_dataset = ReplayBuffer(
        root_dir=f"/home/ubuntu/slippify/data_window_{window_len}/val/",
        transform=transform,
        window_len=window_len
    )
    test_dataset = ReplayBuffer(
        root_dir=f"/home/ubuntu/slippify/data_window_{window_len}/test/",
        transform=transform,
        window_len=window_len
    )
    return train_dataset, val_dataset, test_dataset


def normalize_datasets_attr(datasets, attr):
    attr_tensor = getattr(datasets[0], attr)
    for i in range(1, len(datasets)):
        attr_tensor = torch.concat((attr_tensor, getattr(datasets[i], attr)), dim=0)

    mean = torch.mean(attr_tensor, dim=0)
    std = torch.std(attr_tensor, dim=0)
    print(f"Data mean: {mean}, Data std: {std}")
    
    for dataset in datasets:
        setattr(dataset, attr, (getattr(dataset, attr) - mean) / (std + 1e-8))


# -----------------------------------------------------------------------------------
# Configuration for Pydra (hyperparameters and fixed values)
# -----------------------------------------------------------------------------------

class TrainActionEncoderConfig(Config):
    def __init__(self):
        self.name = "train_action_encoder"
        self.transform = "default"
        self.dropout = 0.1
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epochs = 50
        self.z_dim = 12  # fixed, not swept
        self.window_len: int = None  # must be provided

    def __repr__(self):
        return f"TrainActionEncoderConfig({self.to_dict()})"

# -----------------------------------------------------------------------------------
# Training function used by both single-run and sweep
# Dataloaders are assumed to be global variables initialized before calling this
# -----------------------------------------------------------------------------------

def train_with_wandb(config):
    assert config.window_len is not None
    print(config)

    datasets = load_datasets(config.window_len, config.transform)
    normalize_datasets_attr(datasets, "actions")
    train_dataset, val_dataset, test_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(window_len=config.window_len, z_dim=config.z_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            (observations, actions, next_observations), frames = batch
            frames = frames.squeeze(0).to(device)
            observations = actions.squeeze(1).to(device)

            y = model(frames)
            loss = F.mse_loss(y, observations, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    (observations, actions, next_observations), frames = batch
                    frames = frames.squeeze(0).to(device)
                    observations = actions.squeeze(1).to(device)

                    y = model(frames)
                    val_loss += F.mse_loss(y, observations, reduction="mean").item()

            avg_val_loss = val_loss / len(val_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
            print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.6f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            (observations, actions, next_observations), frames = batch
            frames = frames.squeeze(0).to(device)
            observations = actions.squeeze(1).to(device)

            y = model(frames)
            test_loss += F.mse_loss(y, observations, reduction="mean").item()

    avg_test_loss = test_loss / len(test_loader)
    wandb.log({"test_loss": avg_test_loss})
    print(f"Test Loss: {avg_test_loss:.6f}")

    torch.save(model.state_dict(), f"{config.name}.pth")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(f"{config.name}.pth")
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"Model saved to {config.name}.pth")

# -----------------------------------------------------------------------------------
# Main entry point for non-sweep runs
# -----------------------------------------------------------------------------------

@pydra.main(base=TrainActionEncoderConfig)
def main(config: TrainActionEncoderConfig):
    # Initialize WandB
    wandb.init(
        project="slippi-frame-autoencoder",
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "z_dim": config.z_dim,
            "dropout": config.dropout,
            "window_len": config.window_len,
            "transform": config.transform,
            "name": config.name,
        }
    )
    train_with_wandb(wandb.config)

# -----------------------------------------------------------------------------------
# Sweep-specific training wrapper
#-----------------------------------------------------------------------------------

def sweep_train():
    # Called by wandb.agent; wandb.init must be called first
    wandb.init()
    config = wandb.config
    print(config)
    train_with_wandb(config)

# -----------------------------------------------------------------------------------
# Script entry: handle sweep vs single run
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Action Encoder with optional WandB sweep")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Launch a WandB hyperparameter sweep"
    )
    parser.add_argument(
        "--window_len", type=int,
        help="Window length for the encoder (do not change in sweep)"
    )
    parser.add_argument(
        "--transform", type=str, default="default",
        help="Transform string for the ReplayBuffer"
    )
    args, unknown = parser.parse_known_args()

    if args.sweep:
        sweep_config = {
            "method": "random",
            "metric": {"name": "val_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"value": 1e-3},
                "batch_size": {"value": 64},
                "dropout": {"value": 0.1},
                # z_dim fixed, not swept
                "z_dim": {"value": 12},
                "epochs": {"value": 30},
                "window_len": {"values": [3, 5, 7, 9]},
                "transform": {"value": args.transform},
                "name": {"value": "train_action_encoder"}
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="slippify-action-encoder_fixed-norm")
        wandb.agent(sweep_id, function=sweep_train)
    else:
        # Single run: use Pydra to parse and launch
        main()
