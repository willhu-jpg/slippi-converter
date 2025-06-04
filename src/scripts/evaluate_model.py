#!/usr/bin/env python3
"""
Model Evaluation Script for Slippi Frame Models

This script loads a trained model and evaluates it on the test dataset.
Supports both Encoder and FrameAE models.

Usage:
    python src/scripts/evaluate_model.py --model_path models/train_encoder.pth --model_type encoder
    python src/scripts/evaluate_model.py --model_path models/train_AE.pth --model_type frameae
"""

import sys
import os
import argparse
from pathlib import Path

# Add the current project's src directory to the front of the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

import wandb

# Import models and data
from infra.model import Encoder
from infra.biggerModel import BigEncoder
from infra.AE import FrameAE
from data.replay_buffer import ReplayBuffer

# Import visualization utilities if they exist
try:
    from utils.visualize import visualize_frame, compare_frames
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Warning: Visualization utilities not found. Skipping visualizations.")

class ModelEvaluator:
    """Evaluates trained models on test data"""
    
    def __init__(self, model_path, model_type, device=None):
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Loaded {model_type} model from {model_path}")
        print(f"Using device: {self.device}")
        
    def _load_model(self):
        """Load the appropriate model based on type"""
        if self.model_type == "encoder":
            model = Encoder(z_dim=10, dropout=0.0)  # No dropout for evaluation
        elif self.model_type == "bigger":
            model = BigEncoder(z_dim=10, dropout=0.0)
        elif self.model_type == "frameae":
            model = FrameAE()  # Remove coord_dim parameter since it's not used in current implementation
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load state dict with error handling for potential tensor sharing issues
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "more than one element of the written-to tensor refers to a single memory location" in str(e):
                print("Warning: Detected tensor sharing issue in saved model. Attempting to fix...")
                # Create a new state dict with cloned tensors
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        fixed_state_dict[key] = value.clone()
                    else:
                        fixed_state_dict[key] = value
                model.load_state_dict(fixed_state_dict)
                print("Successfully loaded model with fixed state dict.")
            else:
                raise e
        
        model.to(self.device)
        
        return model
    
    def to_serializable(self, obj):
        if isinstance(obj, (np.generic,)):       # np.float32, np.int64, …
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        raise TypeError(f"{type(obj)} not serializable")
    
    def evaluate_encoder(self, test_dataloader):
        """Evaluate encoder model (predicts observations from frames)"""
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        print("Evaluating Encoder model...")
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                (observations, actions, next_observations), frames = batch
                
                # Handle tensor shapes
                if isinstance(frames, list):
                    frames = torch.stack(frames).to(self.device)
                else:
                    frames = frames.to(self.device)
                
                if isinstance(observations, list):
                    observations = torch.stack(observations).to(self.device)
                else:
                    observations = observations.to(self.device)
                
                # Remove extra dimensions
                if len(frames.shape) > 4:
                    frames = frames.squeeze()
                if len(observations.shape) > 2:
                    observations = observations.squeeze()
                
                # Forward pass
                predictions = self.model(frames)
                
                # Compute loss
                loss = F.mse_loss(predictions, observations, reduction='mean')
                total_loss += loss.item()
                
                # Store for analysis
                all_predictions.append(self.to_serializable(predictions))
                all_targets.append(self.to_serializable(observations))
                num_batches += 1
        
        # Aggregate results
        avg_loss = total_loss / num_batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute additional metrics
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        # Per-feature analysis
        feature_names = ['P1_X', 'P1_Y', 'P1_%', 'P1_Face', 'P1_Action',
                        'P2_X', 'P2_Y', 'P2_%', 'P2_Face', 'P2_Action']
        
        per_feature_mae = np.mean(np.abs(all_predictions - all_targets), axis=0)
        per_feature_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))
        
        results = {
            'model_type': 'encoder',
            'avg_loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'per_feature_mae': dict(zip(feature_names, per_feature_mae.tolist())),
            'per_feature_rmse': dict(zip(feature_names, per_feature_rmse.tolist())),
            'num_samples': len(all_predictions)
        }
        
        return results, all_predictions, all_targets

    def annotate_frames(self, coords, frames, save_path):
        """Annotate frames with coordinates from SpatialSoftArgmax"""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Ensure save directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert tensors to numpy if needed
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        
        batch_size = frames.shape[0]
        num_channels = 16  # Your encoder has 16 channels
        
        for i in range(min(4, batch_size)):  # Only annotate first 4 frames
            frame = frames[i]  # Shape: (C, H, W)
            coord = coords[i]  # Shape: (32,) -> [x_1, ..., x_16, y_1, ..., y_16]
            
            # Convert frame to displayable format
            if frame.shape[0] == 3:  # RGB: (C, H, W) -> (H, W, C)
                frame_display = frame.transpose(1, 2, 0)
                # Normalize to [0,1] if values are in [0,255]
                if frame_display.max() > 1.0:
                    frame_display = frame_display / 255.0
                frame_display = np.clip(frame_display, 0, 1)
            else:  # Grayscale or single channel
                frame_display = frame[0] if frame.ndim == 3 else frame
            
            # Extract x and y coordinates
            x_coords = coord[:num_channels]  # First 16 values
            y_coords = coord[num_channels:]  # Last 16 values
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Display frame
            if frame_display.ndim == 3:
                ax.imshow(frame_display)
            else:
                ax.imshow(frame_display, cmap='gray')
            
            # Plot coordinates as colored dots
            colors = plt.cm.tab20(np.linspace(0, 1, num_channels))
            
            for j in range(num_channels):
                x, y = x_coords[j], y_coords[j]
                x = x / 109 * 240
                y = y / 109 * 240
                # Add scatter point with channel number
                ax.scatter(x, y, c=[colors[j]], s=150, alpha=0.8, edgecolors='white', linewidth=2)
                ax.text(x+3, y+3, f'{j}', color='white', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            ax.set_title(f'Frame {i} with Spatial Attention Coordinates\n'
                        f'({num_channels} channels, coordinates in feature space)')
            ax.set_xlim(0, frame_display.shape[1])
            ax.set_ylim(frame_display.shape[0], 0)  # Invert y-axis for image coordinates
            
            # Save individual frame
            batch_num = save_path.split('_')[-1].split('.')[0] if '_' in save_path else '0'
            frame_save_path = save_dir / f"annotated_frame_batch_{batch_num}_sample_{i}.png"
            plt.savefig(frame_save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Saved annotated frame to {frame_save_path}")

    def evaluate_frameae(self, test_dataloader):
        """Evaluate FrameAE model (autoencoder)"""
        total_loss = 0.0
        recon_losses = []
        num_batches = 0
        
        print("Evaluating FrameAE model...")
        
        # Reset model state for fresh evaluation
        if hasattr(self.model, 'reset_temporal_state'):
            self.model.reset_temporal_state()
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                _, frames = batch
                
                # Handle tensor shapes
                if isinstance(frames, list):
                    frames = torch.stack(frames).to(self.device)
                else:
                    frames = frames.to(self.device)
                
                if len(frames.shape) > 4:
                    frames = frames.squeeze()
                
                # Forward pass
                coords, recon, recon_truth, loss = self.model(frames)

                save_path = f"evaluation_results/coords_batch_{num_batches}.png"
                self.annotate_frames(coords, frames, save_path)
                
                total_loss += loss.item()
                
                # Compute reconstruction loss separately
                recon_loss = F.mse_loss(recon, recon_truth, reduction='mean')
                recon_losses.append(recon_loss.item())
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon_loss = np.mean(recon_losses)
        
        results = {
            'model_type': 'frameae',
            'avg_total_loss': avg_loss,
            'avg_recon_loss': avg_recon_loss,
            'num_samples': num_batches * test_dataloader.batch_size
        }
        
        return results, None, None
    
    def visualize_results(self, test_dataloader, save_dir="evaluation_results"):
        """Create visualizations of model performance"""
        if not HAS_VIZ:
            print("Skipping visualizations (utils not available)")
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"Creating visualizations in {save_dir}...")
        
        with torch.no_grad():
            # Get a few batches for visualization
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx >= 3:  # Only visualize first 3 batches
                    break
                
                if self.model_type == "encoder":
                    (observations, actions, next_observations), frames = batch
                    
                    # Process tensors
                    if isinstance(frames, list):
                        frames = torch.stack(frames).to(self.device)
                    else:
                        frames = frames.to(self.device)
                    
                    if len(frames.shape) > 4:
                        frames = frames.squeeze()
                    
                    # Visualize frames
                    for i in range(min(4, frames.shape[0])):
                        visualize_frame(frames[i], 
                                      title=f"Test Frame {batch_idx}_{i}",
                                      save_path=save_dir / f"frame_{batch_idx}_{i}.png",
                                      show=False)
                
                elif self.model_type == "frameae":
                    _, frames = batch
                    
                    if isinstance(frames, list):
                        frames = torch.stack(frames).to(self.device)
                    else:
                        frames = frames.to(self.device)
                    
                    if len(frames.shape) > 4:
                        frames = frames.squeeze()
                    
                    # Get reconstructions
                    coords, recon, recon_truth, loss = self.model(frames)
                    
                    # Visualize original vs reconstruction vs truth
                    for i in range(min(4, frames.shape[0])):
                        # Original frame
                        visualize_frame(frames[i], 
                                      title=f"Original Frame {batch_idx}_{i}",
                                      save_path=save_dir / f"original_{batch_idx}_{i}.png",
                                      show=False)
                        
                        # Reconstruction
                        plt.figure(figsize=(10, 4))
                        
                        plt.subplot(1, 2, 1)
                        plt.imshow(recon[i].cpu().numpy(), cmap='gray')
                        plt.title('Reconstruction')
                        plt.axis('off')
                        
                        plt.subplot(1, 2, 2)
                        plt.imshow(recon_truth[i].cpu().numpy(), cmap='gray')
                        plt.title('Ground Truth')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(save_dir / f"reconstruction_{batch_idx}_{i}.png", 
                                  bbox_inches='tight', dpi=150)
                        plt.close()
    
    def run_evaluation(self, test_data_path, batch_size=128, save_results=True):
        """Run complete evaluation pipeline"""
        print(f"Loading test data from {test_data_path}...")
        
        # Load test dataset with appropriate transforms based on model type
        if self.model_type == "frameae":
            # Use AE_transform to match training data preprocessing
            test_dataset = ReplayBuffer(root_dir=test_data_path, transforms=["AE_transform"])
        else:
            # For encoder models, use default transforms
            test_dataset = ReplayBuffer(root_dir=test_data_path, transforms=["default"])
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Test dataset: {len(test_dataset)} samples")
        
        # Run evaluation based on model type
        if self.model_type == "encoder":
            results, predictions, targets = self.evaluate_encoder(test_dataloader)
        elif self.model_type == "frameae":
            results, predictions, targets = self.evaluate_frameae(test_dataloader)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Add metadata
        results.update({
            'model_path': str(self.model_path),
            'test_data_path': test_data_path,
            'batch_size': batch_size,
            'device': str(self.device),
            'evaluation_time': datetime.now().isoformat()
        })
        
        # Print results
        self._print_results(results)
        
        # Save results
        if save_results:
            results_file = f"evaluation_results_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_file}")
        
        # Create visualizations
        self.visualize_results(test_dataloader)
        
        return results
    
    def _print_results(self, results):
        """Print evaluation results in a nice format"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if results['model_type'] == 'encoder':
            print(f"Model Type: Encoder")
            print(f"Average Loss (MSE): {results['avg_loss']:.6f}")
            print(f"Mean Absolute Error: {results['mae']:.6f}")
            print(f"Root Mean Square Error: {results['rmse']:.6f}")
            print(f"Number of samples: {results['num_samples']}")
            
            print("\nPer-feature MAE:")
            for feature, mae in results['per_feature_mae'].items():
                print(f"  {feature}: {mae:.6f}")
                
        elif results['model_type'] == 'frameae':
            print(f"Model Type: FrameAE")
            print(f"Average Total Loss: {results['avg_total_loss']:.2f}")
            print(f"Average Reconstruction Loss: {results['avg_recon_loss']:.6f}")
            print(f"Number of samples: {results['num_samples']}")
        
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Slippi models')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model (.pth file)')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['encoder', 'frameae'],
                      help='Type of model to evaluate')
    parser.add_argument('--test_data', type=str, 
                      default='/home/ubuntu/project/slippify/data_split/test/',
                      help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu, auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(args.model_path, args.model_type, device)
    results = evaluator.run_evaluation(args.test_data, args.batch_size)
    
    return results

if __name__ == "__main__":
    main()