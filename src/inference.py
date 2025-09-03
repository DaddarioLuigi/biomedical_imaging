# src/inference.py
# Inference minimale: CT (NIfTI) -> mask (NIfTI)
# Usa stesso preprocessing del training: HU window + min-max [0,1]

import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf

def _apply_hu_window(x, wmin=-1000, wmax=300):
    return np.clip(x.astype(np.float32), wmin, wmax)

def _normalize_minmax(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def _load_ct_from_case(input_path):
    """
    Se input_path è una cartella, cerca ct.nii.gz al suo interno.
    Altrimenti assume sia un file NIfTI di CT.
    Restituisce (ct_array, nib_ref_path)
    """
    if os.path.isdir(input_path):
        ct_path = os.path.join(input_path, "ct.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"ct.nii.gz non trovato in {input_path}")
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
    # 1) carica CT
    ct, ref_path = _load_ct_from_case(input_path)

    # 2) preprocessing come nel training
    ct = _apply_hu_window(ct, wmin=wmin, wmax=wmax)
    ct = _normalize_minmax(ct)
    if ct.ndim != 3:
        raise ValueError(f"CT atteso 3D, trovato shape {ct.shape}")
    ct = ct[..., None]  # (D,H,W,1)
    x = np.expand_dims(ct, axis=0)  # (1,D,H,W,1)

    # 3) carica modello
    # NOTA: il modello è stato compilato con dice/iou custom;
    # per sola inferenza non servono custom_objects, ma se servissero:
    # from src.model import dice_coefficient, iou_coefficient, dice_loss
    # custom = {'dice_coefficient': dice_coefficient, 'iou_coefficient': iou_coefficient, 'dice_loss': dice_loss}
    # model = tf.keras.models.load_model(model_path, custom_objects=custom)
    model = tf.keras.models.load_model(model_path, compile=False)

    # 4) predici
    y_prob = model.predict(x, verbose=0)[0]  # (D,H,W,1)
    y_prob = np.squeeze(y_prob, axis=-1)     # (D,H,W)

    # 5) soglia e salva
    y_bin = (y_prob >= float(threshold)).astype(np.uint8)

    # salva maschera binaria
    _save_nifti_like(ref_path, y_bin, out_mask_path, dtype=np.uint8)

    # opzionale: salva la probabilità
    if save_prob_path is not None and len(save_prob_path) > 0:
        _save_nifti_like(ref_path, y_prob, save_prob_path, dtype=np.float32)

    print(f"[OK] Salvato mask: {out_mask_path}")
    if save_prob_path:
        print(f"[OK] Salvato prob:  {save_prob_path}")

def main():
    ap = argparse.ArgumentParser(description="Inferenza 3D lung segmentation")
    ap.add_argument("--model", required=True, help="Percorso al modello .keras (best_model.keras)")
    ap.add_argument("--input", required=True, help="Cartella paziente (con ct.nii.gz) oppure file CT .nii/.nii.gz")
    ap.add_argument("--output_mask", required=True, help="Percorso di output per la maschera binaria NIfTI")
    ap.add_argument("--output_prob", default=None, help="(Opzionale) Percorso di output per la probabilità NIfTI float")
    ap.add_argument("--wmin", type=int, default=-1000, help="HU window min")
    ap.add_argument("--wmax", type=int, default=300, help="HU window max")
    ap.add_argument("--thr", type=float, default=0.5, help="Soglia di binarizzazione")
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
