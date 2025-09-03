# src/data_loader.py
import os
import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir, wmin=-1000, wmax=300, add_channel=True):
        self.data_dir = data_dir
        self.wmin = wmin
        self.wmax = wmax
        self.add_channel = add_channel
        self.cases = self._list_cases()

    def _list_cases(self):
        case_dirs = sorted(
            d for d in glob.glob(os.path.join(self.data_dir, "*"))
            if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "ct.nii.gz"))
            and os.path.exists(os.path.join(d, "lung.nii.gz"))
        )
        if not case_dirs:
            raise FileNotFoundError(f"Nessun caso valido in {self.data_dir}")
        return case_dirs

    def _apply_hu_window(self, x):
        """Clipping HU diretto nella classe"""
        return np.clip(x, self.wmin, self.wmax)

    def _normalize_minmax(self, x):
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    def load_case(self, case_dir):
        ct_p = os.path.join(case_dir, "ct.nii.gz")
        mk_p = os.path.join(case_dir, "lung.nii.gz")

        ct = nib.load(ct_p).get_fdata().astype(np.float32)
        mk = nib.load(mk_p).get_fdata().astype(np.float32)

        if ct.shape != mk.shape:
            raise ValueError(f"Shape mismatch: {ct.shape} vs {mk.shape}")

        # HU window + normalizzazione
        ct = self._apply_hu_window(ct)
        ct = self._normalize_minmax(ct)

        # maschera binaria
        mk = (mk > 0).astype(np.float32)

        if self.add_channel:
            ct = ct[..., None]
            mk = mk[..., None]

        return ct, mk

    def split(self, ratios=(0.7, 0.15, 0.15), seed=42):
        r_train, r_val, r_test = ratios
        if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
            raise ValueError("Le proporzioni devono sommare a 1.0")

        train, temp = train_test_split(self.cases, test_size=(1 - r_train),
                                       random_state=seed, shuffle=True)
        rel_val = r_val / (r_val + r_test)
        val, test = train_test_split(temp, test_size=(1 - rel_val),
                                     random_state=seed, shuffle=True)
        return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
