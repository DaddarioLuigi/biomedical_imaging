import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
This class provides samples for training a 3D segmentation model.
Given a list of case folders and a data loader, it fetches (CT, mask) pairs.
It can apply simple 3D data augmentation like flips, rotations, and intensity noise.
"""
class VolumeDataset(Dataset):
    def __init__(
        self,
        case_paths,
        data_loader,             
        augment=True,
        p_flip=0.5,
        p_rot90=0.5,
        p_jitter=0.0,           # default off
        jitter_sigma=0.03,      # std of gaussian noise
        seed=42
    ):
        self.case_paths = list(case_paths)
        self.dl = data_loader
        self.augment = bool(augment)
        self.p_flip = float(p_flip)
        self.p_rot90 = float(p_rot90)
        self.p_jitter = float(p_jitter)
        self.jitter_sigma = float(jitter_sigma)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        cdir = self.case_paths[idx]
        x, y = self.dl.load_case(cdir)  # (D,H,W,1)
        if self.augment:
            x, y = self._apply_augment(x, y)
        # convert to torch tensors with channel-first
        x = torch.from_numpy(np.transpose(x, (3, 0, 1, 2)))  # (1,D,H,W)
        y = torch.from_numpy(np.transpose(y, (3, 0, 1, 2)))  # (1,D,H,W)
        return x.float(), y.float()

    def _apply_augment(self, x, y):
        # random flips along D/H/W
        if self.rng.random() < self.p_flip:
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=0); y = np.flip(y, axis=0)  # D
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=1); y = np.flip(y, axis=1)  # H
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2); y = np.flip(y, axis=2)  # W

        # 90-degree rotation around a random pair of axes among (D,H), (H,W), (D,W)
        if self.rng.random() < self.p_rot90:
            pairs = [(0, 1), (1, 2), (0, 2)]
            ax = pairs[self.rng.integers(0, len(pairs))]
            k = int(self.rng.integers(1, 4))  # 90,180,270
            x = np.rot90(x, k=k, axes=ax)
            y = np.rot90(y, k=k, axes=ax)

        # intensity jitter on CT only
        if self.p_jitter > 0 and self.rng.random() < self.p_jitter:
            noise = self.rng.normal(0.0, self.jitter_sigma, size=x.shape).astype(np.float32)
            x = x + noise
            x = np.clip(x, 0.0, 1.0)
        return x, y


def create_dataloader(dataset: Dataset, batch_size: int = 2, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 