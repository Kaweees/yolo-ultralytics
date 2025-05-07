import os
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class VoxCelebDataset(Dataset):
    def __init__(self, train_csv_path: str, transform=None):
        """
        Args:
            train_csv_path (str): Path to the dataset CSV file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df: pd.DataFrame = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_path"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform = torchaudio.load(self.paths[idx])

        waveform_length = waveform.shape[-1]

        sample = {
            "waveform": waveform,
            "path": self.paths[idx],
            "mapped_id": self.labels[idx],
            "lens": waveform_length,
        }

        return sample


# Path to your VoxCeleb dataset directory
data_dir = "path/to/voxceleb.csv"

# Create the dataset
voxceleb_dataset = VoxCelebDataset(data_dir)

# Split the dataset into train, validation, and test sets
# 80% for training, 10% for validation, 10% for testing
train_size = int(0.8 * len(voxceleb_dataset))
val_size = int(0.1 * len(voxceleb_dataset))
test_size = len(voxceleb_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    voxceleb_dataset, [train_size, val_size, test_size]
)

# Create DataLoaders for each split
batch_size = 16
voxceleb_dataloader_train = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
voxceleb_dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
voxceleb_dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
