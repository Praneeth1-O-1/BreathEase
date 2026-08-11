"""
Patient-level train / validation / test split generator.

Run once before training:
    python split.py

Behaviour
---------
1. If the official NIH split files (train_val_list.txt, test_list.txt)
   are present in DATA_DIR, they define the train_val vs. test boundary,
   ensuring results are directly comparable to CheXNet and published benchmarks.

2. If the official files are absent (e.g. local sample), a deterministic
   patient-level split is generated (70 / 10 / 20).

3. If the CSV has no 'Patient ID' column (minimal sample CSV), an
   image-level split is used with a printed warning.

Splits are saved as CSV files and reused on subsequent calls.
The test split is never re-derived at runtime — evaluate.py loads
splits/test.csv directly to guarantee no leakage between runs.
"""

import os

import numpy as np
import pandas as pd

import config


# ── Split ratios used when official files are absent ─────────────────────────
_VAL_FRACTION  = 0.125   # ~12.5% of train_val patients → ~10% of total
_TEST_FRACTION = 0.200   # 20% of all patients


# ============================================================
# Public API
# ============================================================

def create_splits(force: bool = False):
    """
    Build patient-level splits and save them to config.SPLIT_DIR.

    Parameters
    ----------
    force : bool
        If True, regenerate and overwrite existing split files.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each DataFrame contains rows from the main metadata CSV.
    """
    os.makedirs(config.SPLIT_DIR, exist_ok=True)

    train_path = os.path.join(config.SPLIT_DIR, "train.csv")
    val_path   = os.path.join(config.SPLIT_DIR, "val.csv")
    test_path  = os.path.join(config.SPLIT_DIR, "test.csv")

    # ── Return cached splits if they already exist ───────────────────────────
    if not force and all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("[split] Existing splits found — loading from disk.")
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    print(f"[split] Generating splits from : {config.CSV_FILE}")

    df = pd.read_csv(config.CSV_FILE)
    print(f"[split] Total rows in CSV      : {len(df)}")

    # ── Step 1: Determine train_val vs. test ─────────────────────────────────
    official_present = (
        os.path.exists(config.TRAIN_VAL_LIST)
        and os.path.exists(config.TEST_LIST)
    )

    if official_present:
        print("[split] Official NIH split files found — using canonical test set.")
        with open(config.TRAIN_VAL_LIST) as f:
            train_val_names = set(f.read().splitlines())
        with open(config.TEST_LIST) as f:
            test_names = set(f.read().splitlines())

        train_val_df = df[df["Image Index"].isin(train_val_names)].copy()
        test_df      = df[df["Image Index"].isin(test_names)].copy()

    else:
        print("[split] Official split files not found — generating patient-level split.")
        train_val_df, test_df = _patient_or_image_split(
            df, test_fraction=_TEST_FRACTION, seed=config.SEED
        )

    print(f"[split] Train+val : {len(train_val_df)} rows")
    print(f"[split] Test      : {len(test_df)} rows")

    # ── Step 2: Carve validation from train_val ──────────────────────────────
    train_df, val_df = _patient_or_image_split(
        train_val_df, test_fraction=_VAL_FRACTION, seed=config.SEED + 1
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n[split] Splits saved:")
    print(f"  Train : {len(train_df):>6} images  →  {train_path}")
    print(f"  Val   : {len(val_df):>6} images  →  {val_path}")
    print(f"  Test  : {len(test_df):>6} images  →  {test_path}")

    _verify_no_overlap(train_df, val_df, test_df)

    return train_df, val_df, test_df


# ============================================================
# Internal helpers
# ============================================================

def _patient_or_image_split(df, test_fraction, seed):
    """
    Split df into (main, held_out) at the patient level when possible.
    Falls back to image-level with a warning if 'Patient ID' is absent.

    Returns (main_df, held_out_df) — held_out is the smaller portion.
    """
    if "Patient ID" in df.columns:
        return _patient_level_split(df, test_fraction, seed)
    else:
        print(
            "[split] WARNING: 'Patient ID' not found in CSV. "
            "Using image-level split — results may be inflated if the same "
            "patient appears in both train and test."
        )
        return _image_level_split(df, test_fraction, seed)


def _patient_level_split(df, held_out_fraction, seed):
    """Split df so that no patient appears in both partitions."""
    patient_ids = df["Patient ID"].unique()
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(patient_ids)

    n_held_out   = max(1, int(held_out_fraction * len(patient_ids)))
    held_out_ids = set(patient_ids[:n_held_out])
    main_ids     = set(patient_ids[n_held_out:])

    held_out_df = df[df["Patient ID"].isin(held_out_ids)].copy()
    main_df     = df[df["Patient ID"].isin(main_ids)].copy()

    return main_df, held_out_df


def _image_level_split(df, held_out_fraction, seed):
    """Fallback: split at the image level with a fixed seed."""
    rng      = np.random.default_rng(seed=seed)
    indices  = rng.permutation(len(df))
    n_held   = max(1, int(held_out_fraction * len(df)))

    held_out_df = df.iloc[indices[:n_held]].copy()
    main_df     = df.iloc[indices[n_held:]].copy()

    return main_df, held_out_df


def _verify_no_overlap(train_df, val_df, test_df):
    """Assert that no image filename appears in more than one split."""
    train_names = set(train_df["Image Index"])
    val_names   = set(val_df["Image Index"])
    test_names  = set(test_df["Image Index"])

    tv_overlap = train_names & val_names
    tt_overlap = train_names & test_names
    vt_overlap = val_names   & test_names

    if tv_overlap or tt_overlap or vt_overlap:
        raise RuntimeError(
            f"[split] OVERLAP DETECTED — "
            f"train∩val={len(tv_overlap)}, "
            f"train∩test={len(tt_overlap)}, "
            f"val∩test={len(vt_overlap)}"
        )

    print("[split] ✓ No image overlap between train / val / test.")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    create_splits(force=False)
