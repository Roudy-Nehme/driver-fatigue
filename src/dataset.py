import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FatigueSequenceDataset(Dataset):
    """
    Expected CSV columns:
    - feature_path
    - label
    - split

    Example:
    feature_path,label,split
    /content/.../video_001.npy,0,train
    /content/.../video_002.npy,1,val
    """

    def __init__(self, csv_path, split):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"feature_path", "label", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = row["feature_path"]
        label = int(row["label"])

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = np.load(feature_path)

        if features.ndim != 2:
            raise ValueError(
                f"Expected feature array shape [T, F], got {features.shape} for {feature_path}"
            )

        x = torch.tensor(features, dtype=torch.float32)
        length = torch.tensor(features.shape[0], dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)

        return x, length, y


def pad_collate_fn(batch):
    """
    Returns:
    - padded_x: [B, T_max, F]
    - lengths: [B]
    - labels: [B]
    """
    sequences, lengths, labels = zip(*batch)

    batch_size = len(sequences)
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]

    padded_x = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded_x[i, :seq_len, :] = seq

    lengths = torch.stack(lengths)
    labels = torch.stack(labels)

    return padded_x, lengths, labels
