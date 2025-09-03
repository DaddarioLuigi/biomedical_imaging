# src/sequence.py
# Keras Sequence minimale per 3D lung segmentation
# - legge path da split['train'|'val'|'test']
# - usa DataLoader.load_case(case_dir)
# - applica flip e rotazioni a 90° (stesse su CT e mask)
# - opzionale jitter d'intensità

import numpy as np
from tensorflow.keras.utils import Sequence

class VolumeSequence(Sequence):
    def __init__(
        self,
        case_paths,
        data_loader,             # istanza di DataLoader
        batch_size=2,
        shuffle=True,
        augment=True,
        p_flip=0.5,
        p_rot90=0.5,
        p_jitter=0.0,           # default off
        jitter_sigma=0.03,      # std rumore gaussiano (su CT normalizzata [0,1])
        seed=42
    ):
        self.case_paths = list(case_paths)
        self.dl = data_loader
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.p_flip = float(p_flip)
        self.p_rot90 = float(p_rot90)
        self.p_jitter = float(p_jitter)
        self.jitter_sigma = float(jitter_sigma)
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    # ---------------------------
    # Keras Sequence API
    # ---------------------------
    def __len__(self):
        return int(np.ceil(len(self.case_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.case_paths)

    def __getitem__(self, idx):
        batch_paths = self.case_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        xs, ys = [], []
        for cdir in batch_paths:
            x, y = self.dl.load_case(cdir)  # (D,H,W,1)
            if self.augment:
                x, y = self._apply_augment(x, y)
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0).astype(np.float32)  # (N,D,H,W,1)
        Y = np.stack(ys, axis=0).astype(np.float32)  # (N,D,H,W,1)
        return X, Y

    # ---------------------------
    # Augmentations (semplici)
    # ---------------------------
    def _apply_augment(self, x, y):
        # flip casuale lungo ciascun asse D/H/W
        if self.rng.random() < self.p_flip:
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=0); y = np.flip(y, axis=0)  # D
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=1); y = np.flip(y, axis=1)  # H
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2); y = np.flip(y, axis=2)  # W

        # rotazione a 90° attorno a una coppia di assi (tra D,H,W)
        if self.rng.random() < self.p_rot90:
            # scegli una coppia tra (D,H), (H,W), (D,W)
            pairs = [(0,1), (1,2), (0,2)]
            ax = pairs[self.rng.integers(0, len(pairs))]
            k = int(self.rng.integers(1, 4))  # 90,180,270
            x = np.rot90(x, k=k, axes=ax)
            y = np.rot90(y, k=k, axes=ax)

        # jitter d'intensità (solo CT, non la mask)
        if self.p_jitter > 0 and self.rng.random() < self.p_jitter:
            noise = self.rng.normal(0.0, self.jitter_sigma, size=x.shape).astype(np.float32)
            x = x + noise
            # CT è in [0,1]: riclip
            x = np.clip(x, 0.0, 1.0)

        return x, y
