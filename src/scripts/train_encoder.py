# Add the current project's src directory to the front of the path
import sys
import os

# Get the absolute path to the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_dir = os.path.dirname(current_dir)  # src directory
sys.path.insert(0, src_dir)

# Now import the modules
from infra.model import Encoder
from data.replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torchvision

columns = ["p1_position_x", "p1_position_y", "p1_percent", "p1_facing", "p1_action", "p2_position_x", "p2_position_y", "p2_percent", "p2_facing", "p2_action"]

def main():
    # Initialize wandb
    wandb.init(
        project="slippi-frame-autoencoder",
        config={
            "learning_rate": 1e-3,
            "batch_size": 128,
            "epochs": 30,
            "architecture": "Encoder",
            "dataset": "slippi_frames",
            "image_size": 64,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    dataset = ReplayBuffer(root_dir="/home/ubuntu/project/slippi-converter/data/test/", window_size=1)
    dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # Log model architecture
    wandb.watch(model, log="all")

    for epoch in tqdm(range(wandb.config.epochs), desc="Epochs"):
        model.train()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            (observations, actions, next_observations), frames = batch

            frames = torch.stack(frames).squeeze(0).to(device)
            observations = observations.detach().clone().requires_grad_(True).squeeze(1).to(device)

            y = model(frames)

            loss = F.mse_loss(y, observations, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
        })

        print(f"Epoch {epoch}, Loss: {loss.item()}")


         # every few epochs, visualize reconstructions:
        if epoch % 10 == 0:
            with torch.no_grad():
                x0 = frames[:8]                                     # last batch
                y0 = model(x0)
                y0 = y0.cpu()
                x0 = x0.cpu()
                grid = torchvision.utils.make_grid(x0, nrow=8)
                torchvision.utils.save_image(grid, f"tmp/reconstructions_{epoch}.png")
                
                # Log reconstruction images to wandb
                wandb.log({
                    "frames": wandb.Image(grid, caption=f"Epoch {epoch} - Frames"),
                })

                print(y0)

    torch.save(model.state_dict(), "model.pth")
    
    # Save model as wandb artifact
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)
    
    print("Model saved to model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()