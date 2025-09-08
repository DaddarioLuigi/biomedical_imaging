import os
import argparse
import numpy as np
import nibabel as nib
import torch

from src.model import build_unet_3d


def _apply_hu_window(x, wmin=-1000, wmax=300):
    return np.clip(x.astype(np.float32), wmin, wmax)


def _normalize_minmax(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


"""
If input_path is a folder, look for ct.nii.gz inside it.
On the contrary, assume it is a CT NIfTI file.
Returns (ct_array, nib_ref_path)
"""
def _load_ct_from_case(input_path):
    if os.path.isdir(input_path):
        ct_path = os.path.join(input_path, "ct.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"ct.nii.gz not found in {input_path}")
        ref_path = ct_path
    else:
        ref_path = input_path
    ct = nib.load(ref_path).get_fdata().astype(np.float32)
    return ct, ref_path


def _save_nifti_like(reference_path, data, out_path, dtype=np.float32):
    ref = nib.load(reference_path)
    nii = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine=ref.affine, header=ref.header)
    nib.save(nii, out_path)


def run_inference(
    model_path: str,
    input_path: str,
    out_mask_path: str,
    wmin: int = -1000,
    wmax: int = 300,
    threshold: float = 0.5,
    save_prob_path: str | None = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1)Load CT
    ct, ref_path = _load_ct_from_case(input_path)

    # 2)Preprocessing (same as training)
    ct = _apply_hu_window(ct, wmin=wmin, wmax=wmax)
    ct = _normalize_minmax(ct)
    if ct.ndim != 3:
        raise ValueError(f"CT expected 3D, got shape {ct.shape}")
    x = np.expand_dims(ct, axis=(0, -1))  # (1,D,H,W,1)
    x = np.transpose(x, (0, 4, 1, 2, 3))  # (1,1,D,H,W)

    # 3)Load model
    model = build_unet_3d()
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # 4)Prediction
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().to(device)
        y_prob = model(x_tensor).cpu().numpy()[0, 0]  # (D,H,W)

    # 5)Threshold and save
    y_bin = (y_prob >= float(threshold)).astype(np.uint8)

    _save_nifti_like(ref_path, y_bin, out_mask_path, dtype=np.uint8)

    if save_prob_path is not None and len(save_prob_path) > 0:
        _save_nifti_like(ref_path, y_prob, save_prob_path, dtype=np.float32)

    print(f"Saved mask: {out_mask_path}")
    if save_prob_path:
        print(f"Saved prob:  {save_prob_path}")


def main():
    ap = argparse.ArgumentParser(description="Inference 3D lung segmentation")
    ap.add_argument("--model", required=True, help="Path to model .pt checkpoint (best_model.pt)")
    ap.add_argument("--input", required=True, help="Patient folder (with ct.nii.gz) or CT .nii/.nii.gz file")
    ap.add_argument("--output_mask", required=True, help="Output path for binary mask NIfTI")
    ap.add_argument("--output_prob", default=None, help="(Optional) Output path for probability NIfTI float")
    ap.add_argument("--wmin", type=int, default=-1000, help="HU window min")
    ap.add_argument("--wmax", type=int, default=300, help="HU window max")
    ap.add_argument("--thr", type=float, default=0.5, help="Binarization threshold")
    args = ap.parse_args()

    run_inference(
        model_path=args.model,
        input_path=args.input,
        out_mask_path=args.output_mask,
        wmin=args.wmin,
        wmax=args.wmax,
        threshold=args.thr,
        save_prob_path=args.output_prob
    )


if __name__ == "__main__":
    main()
