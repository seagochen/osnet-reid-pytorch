"""
ReID dataset loading from CSV.

CSV format:
    img_path,person_id,camera_id
    train/0001_c1s1_001.jpg,1,1
    train/0001_c2s2_002.jpg,1,2
    ...

camera_id column is optional.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ReIDDataset(Dataset):
    """
    ReID dataset that loads images and identity labels from a CSV file.

    Returns (image, person_id, camera_id) tuples.
    """

    def __init__(self, csv_file, dataset_root, transform=None):
        """
        Args:
            csv_file: Path to CSV file
            dataset_root: Root directory for images
            transform: torchvision transforms
        """
        self.data = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.transform = transform

        # Remap person_id to contiguous 0-indexed labels
        unique_pids = sorted(self.data['person_id'].unique())
        self.pid_to_label = {pid: label for label, pid in enumerate(unique_pids)}
        self.num_pids = len(unique_pids)

        # Check for camera_id column
        self.has_camid = 'camera_id' in self.data.columns

        # Build pid -> list of indices for sampler
        self.pid_index = {}
        for idx, row in self.data.iterrows():
            label = self.pid_to_label[row['person_id']]
            if label not in self.pid_index:
                self.pid_index[label] = []
            self.pid_index[label].append(idx)

        print(f"ReIDDataset: {len(self.data)} images, {self.num_pids} identities")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = os.path.join(self.dataset_root, row['img_path'])
        pid = self.pid_to_label[row['person_id']]
        camid = int(row['camera_id']) if self.has_camid else 0

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, pid, camid
