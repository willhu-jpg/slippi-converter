import os
import pickle
import argparse
import numpy as np
import shutil

def process_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    # Ensure 'frame' key exists and is a NumPy array
    if 'frame' not in obj or not isinstance(obj['frame'], np.ndarray):
        raise RuntimeError(f"Expected 'frame' key with NumPy array in {os.path.basename(pkl_path)}")

    return obj


def sample_indices(total_frames, window, keep_window, split_ratio):
    radius = window // 2
    step = max(1, int(window / split_ratio))
    offset = step // 2

    # Sample at certain step, with test offset half a step
    center_sets = [
        np.arange(radius, total_frames, step),
        np.arange(radius + offset, total_frames, step)
    ]

    # Clip so we don't have any windows off the end
    for i, centers in enumerate(center_sets):
        center_sets[i] = centers[centers + radius <= total_frames]

    # Gather window ranges surrounding the centers
    window_sets = []
    for centers in center_sets:
        windows = [np.arange(centers[i] - radius, centers[i] + radius + 1)
                   for i in range(len(centers))]
        windows = np.array(windows).reshape(len(windows) * window)
        window_sets.append(windows)

    # Train is all remaining frames
    all_frames = np.arange(total_frames)
    blacked_out = np.sort(np.concatenate(window_sets, axis=0))
    train = np.setdiff1d(all_frames, blacked_out)
    val, test = center_sets
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    if keep_window:
        val, test = window_sets
        
        # Check that train windows are at least window size
        cur_len = 1
        for i in range(1, len(train)):
            if train[i] != train[i - 1] + 1:
                assert cur_len >= window
                cur_len = 0
            cur_len += 1
        assert cur_len > window

    return train, val, test


def prepare_output_dirs(output_dir, sets, base_name):
    for split in sets:
        # Create frames and pkl subdirectories
        frames_sub = os.path.join(output_dir, split, "frames", base_name)
        pkl_sub = os.path.join(output_dir, split, "pkl")
        os.makedirs(frames_sub, exist_ok=True)
        os.makedirs(pkl_sub, exist_ok=True)


def copy_frames(frames_src_dir, frames_dst_dir, idxs):
    for i, idx in enumerate(idxs):
        # Compute 1-indexed filename: add 1 to the zero-based idx
        file_idx = idx + 1
        src_file_name = f"frame_{file_idx:04d}.jpg"
        dst_file_name = f"frame_{i:04d}.jpg"

        src_path = os.path.join(frames_src_dir, src_file_name)
        dst_path = os.path.join(frames_dst_dir, dst_file_name)

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


def load_and_split_data(data_dir, output_dir, window, keep_window, split_ratio):
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
        train_idxs, val_idxs, test_idxs = sample_indices(total, window, keep_window, split_ratio)

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
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--keep_window", action="store_true")
    parser.add_argument("--split_ratio", type=float, default=0.15)
    args = parser.parse_args()

    # Window length must be odd to center frame
    assert args.window & 1 == 1

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    load_and_split_data(args.data_dir, args.output_dir, 
                        args.window, args.keep_window, args.split_ratio)


if __name__ == "__main__":
    main()
