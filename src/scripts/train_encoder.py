from infra.model import FrameAE
from infra.dataset import FrameDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from tqdm import tqdm
import wandb

def main():
    # Initialize wandb
    wandb.init(
        project="slippi-frame-autoencoder",
        config={
            "learning_rate": 1e-3,
            "batch_size": 128,
            "epochs": 100,
            "architecture": "FrameAE",
            "dataset": "slippi_frames",
            "image_size": 64,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameAE().to(device)
    dataset = FrameDataset("data/test/frames/20200226T211056")
    dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # Log model architecture
    wandb.watch(model, log="all")

    for epoch in tqdm(range(wandb.config.epochs), desc="Epochs"):
        model.train()
        total_elbo = 0
        total_mse = 0
        total_kl = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            x = batch.to(device)
            recon, mu, logv, loss, mse, kl = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_elbo += loss.item()
            total_mse += mse.item()
            total_kl += kl.item()

        # Calculate average metrics for the epoch
        avg_elbo = total_elbo / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/elbo": avg_elbo,
            "train/mse": avg_mse,
            "train/kl": avg_kl,
        })

        print(f"Epoch {epoch}, ELBO: {avg_elbo}, MSE: {avg_mse}, KL: {avg_kl}")

         # every few epochs, visualize reconstructions:
        if epoch % 10 == 0:
            with torch.no_grad():
                x0 = x[:8]                                     # last batch
                r0 = model.dec(model.reparam(*model.enc(x0)))
                r0 = r0.cpu()
                x0 = x0.cpu()
                grid = torchvision.utils.make_grid(torch.cat([x0, r0]), nrow=8)
                torchvision.utils.save_image(grid, f"tmp/reconstructions_{epoch}.png")
                
                # Log reconstruction images to wandb
                wandb.log({
                    "reconstructions": wandb.Image(grid, caption=f"Epoch {epoch} - Original (top) vs Reconstructed (bottom)")
                })

    torch.save(model.state_dict(), "model.pth")
    
    # Save model as wandb artifact
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)
    
    print("Model saved to model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()