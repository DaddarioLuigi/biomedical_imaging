import os
import argparse
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from src.inference import run_inference


def _load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data, nii


def _pick_slice_indices(ct: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    d, h, w = ct.shape[:3]
    if mask is None or mask.sum() == 0:
        return d // 2, h // 2, w // 2
    # Use center of mass to choose representative slices
    coords = np.argwhere(mask > 0)
    cz, cy, cx = np.mean(coords, axis=0)
    return int(round(cz)), int(round(cy)), int(round(cx))


def plot_overlays(
    ct: np.ndarray,
    mask_gt: Optional[np.ndarray] = None,
    mask_pred: Optional[np.ndarray] = None,
    title: str = "",
    out_path: Optional[str] = None,
    window: Tuple[float, float] = (-1000, 1000)
):
    assert ct.ndim in (3, 4), f"CT ndim must be 3 or 4, got {ct.ndim}"
    if ct.ndim == 4 and ct.shape[-1] == 1:
        ct = np.squeeze(ct, axis=-1)

    if mask_gt is not None and mask_gt.ndim == 4 and mask_gt.shape[-1] == 1:
        mask_gt = np.squeeze(mask_gt, axis=-1)
    if mask_pred is not None and mask_pred.ndim == 4 and mask_pred.shape[-1] == 1:
        mask_pred = np.squeeze(mask_pred, axis=-1)

    z_idx, y_idx, x_idx = _pick_slice_indices(ct, mask_gt if mask_gt is not None else mask_pred)

    vmin, vmax = window
    fig, ax = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    fig.suptitle(title)

    # Sagittal (YZ at x)
    ax[0].imshow(ct[x_idx, :, :].T, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    if mask_gt is not None:
        ax[0].imshow(np.ma.masked_where(mask_gt[x_idx, :, :].T == 0, mask_gt[x_idx, :, :].T), cmap="Reds", alpha=0.35, origin="lower")
    if mask_pred is not None:
        ax[0].imshow(np.ma.masked_where(mask_pred[x_idx, :, :].T == 0, mask_pred[x_idx, :, :].T), cmap="Blues", alpha=0.35, origin="lower")
    ax[0].set_title("Sagittal")
    ax[0].axis("off")

    # Coronal (XZ at y)
    ax[1].imshow(ct[:, y_idx, :].T, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    if mask_gt is not None:
        ax[1].imshow(np.ma.masked_where(mask_gt[:, y_idx, :].T == 0, mask_gt[:, y_idx, :].T), cmap="Reds", alpha=0.35, origin="lower")
    if mask_pred is not None:
        ax[1].imshow(np.ma.masked_where(mask_pred[:, y_idx, :].T == 0, mask_pred[:, y_idx, :].T), cmap="Blues", alpha=0.35, origin="lower")
    ax[1].set_title("Coronal")
    ax[1].axis("off")

    # Axial (XY at z)
    ax[2].imshow(ct[:, :, z_idx], cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    if mask_gt is not None:
        ax[2].imshow(np.ma.masked_where(mask_gt[:, :, z_idx] == 0, mask_gt[:, :, z_idx]), cmap="Reds", alpha=0.35, origin="lower")
    if mask_pred is not None:
        ax[2].imshow(np.ma.masked_where(mask_pred[:, :, z_idx] == 0, mask_pred[:, :, z_idx]), cmap="Blues", alpha=0.35, origin="lower")
    ax[2].set_title("Axial")
    ax[2].axis("off")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_3d_mask(mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0), out_path: Optional[str] = None):
    assert mask.ndim in (3, 4)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)

    verts, faces, normals, values = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.15)
    mesh.set_edgecolor('none')
    mesh.set_facecolor('tab:blue')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, mask.shape[0] * spacing[0])
    ax.set_ylim(0, mask.shape[1] * spacing[1])
    ax.set_zlim(0, mask.shape[2] * spacing[2])
    ax.set_title("Mask 3D Isosurface")
    ax.set_box_aspect([mask.shape[0] * spacing[0], mask.shape[1] * spacing[1], mask.shape[2] * spacing[2]])
    ax.axis('off')

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def visualize_case(case_dir: str, model_path: Optional[str] = None, out_prefix: Optional[str] = None,
                    wmin: int = -1000, wmax: int = 1000, thr: float = 0.5):
    ct_path = os.path.join(case_dir, "ct.nii.gz")
    gt_path = os.path.join(case_dir, "lung.nii.gz")

    ct, ct_ref = _load_nifti(ct_path)
    gt, _ = _load_nifti(gt_path)

    # Normalize like inference does (window and min-max)
    ct_clip = np.clip(ct, wmin, wmax)
    mn, mx = float(ct_clip.min()), float(ct_clip.max())
    ct_norm = (ct_clip - mn) / (mx - mn + 1e-8)

    pred = None
    if model_path is not None and len(model_path) > 0:
        tmp_prob = os.path.join("results", "tmp_prob.nii.gz")
        tmp_mask = os.path.join("results", "tmp_mask.nii.gz")
        os.makedirs("results", exist_ok=True)
        run_inference(
            model_path=model_path,
            input_path=case_dir,
            out_mask_path=tmp_mask,
            wmin=wmin,
            wmax=wmax,
            threshold=thr,
            save_prob_path=tmp_prob,
        )
        pred, _ = _load_nifti(tmp_mask)

    title = os.path.basename(case_dir)
    out_img = f"{out_prefix}_slices.png" if out_prefix else None
    plot_overlays(ct_norm, mask_gt=(gt > 0).astype(np.float32), mask_pred=(pred > 0).astype(np.float32) if pred is not None else None,
                  title=title, out_path=out_img, window=(0.0, 1.0))

    # 3D view (GT or prediction if present)
    spacing = ct_ref.header.get_zooms()[:3]
    mask_for_3d = (pred > 0.5).astype(np.float32) if pred is not None else (gt > 0.5).astype(np.float32)
    out_3d = f"{out_prefix}_3d.png" if out_prefix else None
    plot_3d_mask(mask_for_3d, spacing=spacing, out_path=out_3d)


def main():
    ap = argparse.ArgumentParser(description="Visualize 3D CT with masks (2D overlays + 3D isosurface)")
    ap.add_argument("--case_dir", required=True, help="Directory containing ct.nii.gz and lung.nii.gz")
    ap.add_argument("--model", default=None, help="Optional path to .pt model to produce prediction overlay")
    ap.add_argument("--out_prefix", default=None, help="Optional output prefix for saving figures (without extension)")
    ap.add_argument("--wmin", type=int, default=-1000)
    ap.add_argument("--wmax", type=int, default=1000)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    visualize_case(args.case_dir, args.model, args.out_prefix, args.wmin, args.wmax, args.thr)


if __name__ == "__main__":
    main() 