import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# This is a copy of the replay buffer from cs224r-smash-sim.

class ReplayBuffer(Dataset):
    """
    A replay buffer for pre-processed Melee pickle files
    """
    def __init__(self, root_dir: str, max_size: int = 1000000, transform: str = "default"):
        self.max_size = max_size
        self.root_dir = root_dir
        self.pkl_dir = root_dir + "pkl/"
        self.pkl_files = sorted(glob(str(Path(self.pkl_dir) / "*.pkl")))
        self.frame_dir = root_dir + "frames/"

        if transform == "default":
            self.transform = T.Compose([
                T.Resize((64,64)),
                T.ToTensor(),                   # [0,1]
                T.Normalize(0.5, 0.5),          # -> [–1,1]
            ])
        elif transform == "jitter":
            self.transform = T.Compose([
                T.Resize((64,64)),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.ToTensor(),                   # [0,1]
                T.Normalize(0.5, 0.5),          # -> [–1,1]
            ])
        elif transform == "AE_transform":
            self.transform = T.Compose([
                T.Resize((240,240)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),          # -> [–1,1]
            ])
        
        self.reset()
        self.add_directory(self.pkl_dir)

    def reset(self):
        """Reset the buffer"""
        self.observations = []  # Game state observations
        self.actions = []      # Player actions
        self.next_observations = []  # Next frame observations
        self.offsets = []
        self.file_ids = []     # Track which file this state originates
        self.frame_idx = []
        self.current_size = 0

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        # Construct a window of observations and actions
        observation = self.observations[idx]
        action = self.actions[idx]
        next_observation = self.next_observations[idx]

        file_id = self.file_ids[idx]
        frame_idx = self.frame_idx[idx]
        file_path = Path(self.frame_dir) / file_id / f"frame_{frame_idx:04d}.jpg"
        frame = self.transform(Image.open(file_path).convert("RGB"))

        return (observation, action, next_observation), frame

    def _create_observation(self, data: Dict, frame_idx: int) -> np.ndarray:
        """
        Create an observation vector from the frame data
        
        Args:
            data: Dictionary containing frame data
            frame_idx: Index of the frame to process
            
        Returns:
            Numpy array containing the observation
        """
        # Extract relevant features for the observation
        obs = np.array([
            data['p1_position_x'][frame_idx],
            data['p1_position_y'][frame_idx],
            data['p1_percent'][frame_idx],
            data['p1_facing'][frame_idx],
            data['p1_action'][frame_idx],
            data['p2_position_x'][frame_idx],
            data['p2_position_y'][frame_idx],
            data['p2_percent'][frame_idx],
            data['p2_facing'][frame_idx],
            data['p2_action'][frame_idx]
        ], dtype=np.float32)
        
        return obs

    def _create_action(self, data: Dict, frame_idx: int) -> np.ndarray:
        """
        Create an action vector from the frame data
        
        Args:
            data: Dictionary containing frame data
            frame_idx: Index of the frame to process
            
        Returns:
            Numpy array containing the action
        """
        # Extract all control inputs for player 1
        action = np.array([
            data['p1_main_stick_x'][frame_idx],    # Main stick X
            data['p1_main_stick_y'][frame_idx],    # Main stick Y
            data['p1_c_stick_x'][frame_idx],       # C-stick X
            data['p1_c_stick_y'][frame_idx],       # C-stick Y
            data['p1_l_shoulder'][frame_idx],      # L trigger
            data['p1_r_shoulder'][frame_idx],      # R trigger
            data['p1_button_a'][frame_idx],        # A button
            data['p1_button_b'][frame_idx],        # B button
            data['p1_button_x'][frame_idx],        # X button
            data['p1_button_y'][frame_idx],        # Y button
            data['p1_button_z'][frame_idx],        # Z button
            data['p1_button_d_up'][frame_idx]     # D up button
        ], dtype=np.float32)
        
        return action

    def add_pkl_file(self, pkl_path: str) -> None:
        """
        Add a pickle file to the buffer
        
        Args:
            pkl_path: Path to the .pkl file
        """
        # print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Get number of frames in this file
        num_frames = len(data['frame'])
        prev_length = len(self.observations)

        first_frame = data['frame'][0]

        # Add frames to buffer, excluding the last frame since we need next_observation
        for i in range(num_frames - 1):
            if self.current_size >= self.max_size:
                return
                
            # Create observation and action vectors
            observation = self._create_observation(data, i)
            action = self._create_action(data, i)
            next_observation = self._create_observation(data, i + 1)
            
            # Add to buffer
            self.observations.append(observation)
            self.actions.append(action)
            self.next_observations.append(next_observation)
            self.file_ids.append(Path(pkl_path).stem)
            self.offsets.append(prev_length)
            self.frame_idx.append(data['frame'][i] - first_frame + 1)
            self.current_size += 1

    def add_directory(self, directory: str) -> None:
        """
        Add all pickle files from a directory
        
        Args:
            directory: Path to directory containing .pkl files
        """
        pkl_files = sorted(glob(str(Path(directory) / "*.pkl")))
        for pkl_file in pkl_files:
            self.add_pkl_file(pkl_file)

        self.observations = np.array(self.observations)
        self.next_observations = np.array(self.next_observations)
        self.mean_observations = np.mean(self.observations, axis=0)
        self.std_observations = np.std(self.observations, axis=0)

        self.observations = (self.observations - self.mean_observations) / (self.std_observations + 1e-8)
        self.next_observations = (self.next_observations - self.mean_observations) / (self.std_observations + 1e-8)

        print(f"Loaded {self.current_size} total frames")