import os
import pickle
import argparse
import numpy as np
import shutil


SPLIT_PERCENT = 0.15  # ~5% coverage by chunks for val and test
CHUNK_SIZE = 5  # number of contiguous frames per sample chunk


def process_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    # Ensure 'frame' key exists and is a NumPy array
    if 'frame' not in obj or not isinstance(obj['frame'], np.ndarray):
        raise RuntimeError(f"Expected 'frame' key with NumPy array in {os.path.basename(pkl_path)}")

    return obj


def sample_indices(total_frames):
    # Determine radius around a center for each chunk
    chunk = CHUNK_SIZE
    radius = chunk // 2  # floor division => 2 for CHUNK_SIZE=5

    # Calculate step between chunk centers so that total frames covered ~ SPLIT_PERCENT * total_frames
    step = max(1, int(chunk / SPLIT_PERCENT))

    # Validation centers: start at radius (to allow chunk to begin at 0)
    val_centers = np.arange(radius, total_frames, step)
    # Test centers: offset by half step for interleaving
    offset = step // 2
    test_centers = np.arange(radius + offset, total_frames, step)

    # Clip any centers beyond range
    val_centers = val_centers[val_centers < total_frames]
    test_centers = test_centers[test_centers < total_frames]

    # Build blackout indices by expanding each center into a contiguous chunk
    blackout_idxs = []
    for c in np.concatenate([val_centers, test_centers]):
        start = max(0, c - radius)
        end = min(total_frames - 1, c + radius)
        blackout_idxs.extend(range(start, end + 1))
    blackout_idxs = np.unique(blackout_idxs)

    # All frame indices
    all_idxs = np.arange(total_frames)
    # Train indices are those not in blackout set
    train_idxs = np.setdiff1d(all_idxs, blackout_idxs)

    # Val and test indices are only the centers
    val_idxs = np.array(val_centers, dtype=int)
    test_idxs = np.array(test_centers, dtype=int)

    return train_idxs, val_idxs, test_idxs


def prepare_output_dirs(output_dir, sets, base_name):
    for split in sets:
        # Create frames and pkl subdirectories
        frames_sub = os.path.join(output_dir, split, "frames", base_name)
        pkl_sub = os.path.join(output_dir, split, "pkl")
        os.makedirs(frames_sub, exist_ok=True)
        os.makedirs(pkl_sub, exist_ok=True)


def copy_frames(frames_src_dir, frames_dst_dir, idxs):
    for idx in idxs:
        # Compute 1-indexed filename: add 1 to the zero-based idx
        file_idx = idx + 1
        file_name = f"frame_{file_idx:04d}.jpg"
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

        # Total frames in the pkl file
        total = obj['frame'].shape[0]

        # Sample train/val/test indices with contiguous chunks for val and test
        train_idxs, val_idxs, test_idxs = sample_indices(total)

        # Prepare output directories
        sets = ['train', 'val', 'test']
        prepare_output_dirs(output_dir, sets, base_name)

        # Source directory containing frame images for this base_name
        frames_src_dir = os.path.join(frames_dir, base_name)

        # Copy frames and save the subset PKL for each split
        for split, idxs in zip(sets, [train_idxs, val_idxs, test_idxs]):
            frames_dst_dir = os.path.join(output_dir, split, "frames", base_name)
            pkl_dst_path = os.path.join(output_dir, split, "pkl", f"{base_name}.pkl")

            copy_frames(frames_src_dir, frames_dst_dir, idxs)
            save_subset_pkl(obj, idxs, pkl_dst_path)

        print(
            f"Split completed for '{base_name}': train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets with blackout chunks.")
    parser.add_argument("data_dir", type=str, help="Path to data directory with frames and pkl folders")
    parser.add_argument("output_dir", type=str, help="Path to output directory for splits")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    load_and_split_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
