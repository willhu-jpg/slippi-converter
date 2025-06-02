import os
import pickle
import argparse
import numpy as np
import shutil


SPLIT_PERCENT = 0.05  # ~5% for val and test each


def process_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    # Ensure 'frame' key exists and is a NumPy array
    if 'frame' not in obj or not isinstance(obj['frame'], np.ndarray):
        raise RuntimeError(f"Expected 'frame' key with NumPy array in {os.path.basename(pkl_path)}")

    return obj


def sample_indices(total_frames):
    # Compute interval step
    step = max(1, int(1 / SPLIT_PERCENT))

    # Validation indices: start at 0
    val_idxs = np.arange(0, total_frames, step)

    # Test indices: offset by half step
    offset = step // 2
    test_idxs = np.arange(offset, total_frames, step)

    # Ensure first frame (0) is included in both
    if 0 not in val_idxs:
        val_idxs = np.insert(val_idxs, 0, 0)

    if 0 not in test_idxs:
        test_idxs = np.insert(test_idxs, 0, 0)

    # Remove duplicates and keep sorted
    val_idxs = np.unique(val_idxs)
    test_idxs = np.unique(test_idxs)

    # Train indices: all others
    all_idxs = np.arange(total_frames)
    train_idxs = np.setdiff1d(all_idxs, np.union1d(val_idxs, test_idxs))

    # Ensure first frame in train
    if 0 not in train_idxs:
        train_idxs = np.insert(train_idxs, 0, 0)

    train_idxs = np.unique(train_idxs)

    return train_idxs, val_idxs, test_idxs


def prepare_output_dirs(output_dir, sets, base_name):
    for split in sets:
        # Create frames and pkl subdirs
        frames_sub = os.path.join(output_dir, split, "frames", base_name)
        pkl_sub = os.path.join(output_dir, split, "pkl")
        os.makedirs(frames_sub, exist_ok=True)
        os.makedirs(pkl_sub, exist_ok=True)


def copy_frames(frames_src_dir, frames_dst_dir, idxs):
    for idx in idxs:
        # Compute 1-indexed filename by subtracting first_frame and adding 1
        idx = idx + 1
        file_name = f"frame_{idx:04d}.jpg"
        src_path = os.path.join(frames_src_dir, file_name)
        dst_path = os.path.join(frames_dst_dir, file_name)
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Frame image not found: {src_path}")
        shutil.copyfile(src_path, dst_path)


def save_subset_pkl(obj, indices, dst_pkl_path):
    subset = {}
    for key, arr in obj.items():
        if isinstance(arr, np.ndarray):
            subset[key] = arr[indices]
        else:
            subset[key] = arr

    with open(dst_pkl_path, 'wb') as f:
        pickle.dump(subset, f)


def load_and_split_data(data_dir, output_dir):
    pkl_dir = os.path.join(data_dir, "pkl")
    frames_dir = os.path.join(data_dir, "frames")

    for filename in os.listdir(pkl_dir):
        if not filename.endswith(".pkl"):
            continue

        base_name = os.path.splitext(filename)[0]
        pkl_path = os.path.join(pkl_dir, filename)
        obj = process_pkl_file(pkl_path)

        # Total frames in pkl
        total = obj['frame'].shape[0]

        # Sample indices for train, val, test
        train_idxs, val_idxs, test_idxs = sample_indices(total)

        # Prepare output directories
        sets = ['train', 'val', 'test']
        prepare_output_dirs(output_dir, sets, base_name)

        # Paths
        frames_src_dir = os.path.join(frames_dir, base_name)

        # Copy frames and save pkl for each split
        for split, idxs in zip(
            sets,
            [train_idxs, val_idxs, test_idxs],
        ):
            frames_dst_dir = os.path.join(output_dir, split, "frames", base_name)
            pkl_dst_path = os.path.join(output_dir, split, "pkl", f"{base_name}.pkl")

            copy_frames(frames_src_dir, frames_dst_dir, idxs)
            save_subset_pkl(obj, idxs, pkl_dst_path)

        print(
            f"Split completed for '{base_name}': train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets.")
    parser.add_argument("data_dir", type=str, help="Path to data directory with frames, pkl, slp folders")
    parser.add_argument("output_dir", type=str, help="Path to output directory for splits")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    load_and_split_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
