"""Download HAM10000 from Kaggle, parse metadata, and write group-aware splits.

Run from the project root:
    python src/prepare_data.py

Outputs data/splits/{train,val,test}.csv with columns (image_path, label).
Images are NOT copied; the CSVs point to the files in the kagglehub cache.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the dataset via kagglehub."""
    import kagglehub

    # kagglehub caches downloads (~/.cache/kagglehub). If the dataset is
    # already there, this returns immediately without downloading again.
    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print(f"Dataset path: {path}")
    return Path(path)


def find_metadata(root: Path) -> Path:
    """Recursively locate HAM10000_metadata.csv."""
    matches = sorted(root.rglob(config.METADATA_FILENAME))
    if not matches:
        raise FileNotFoundError(f"{config.METADATA_FILENAME} not found under {root}")
    return matches[0]


def build_image_index(root: Path) -> dict:
    """Map image_id -> file path for every .jpg under root.

    Images are split across HAM10000_images_part_1/part_2, and some dataset
    versions contain lowercase duplicate folders. We keep the first path seen
    for each image_id, which deduplicates them. hmnist_*.csv files are ignored
    because we only look at .jpg files.
    """
    index = {}
    n_files = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() != ".jpg" or not p.is_file():
            continue
        n_files += 1
        index.setdefault(p.stem, p)
    print(f"Found {n_files} .jpg files -> {len(index)} unique image_ids")
    return index


def group_split(df: pd.DataFrame) -> pd.Series:
    """Assign each row to train/val/test with a stratified GROUP split.

    All images of one lesion_id land in the same split, so the same lesion is
    never seen in both training and testing. We cut the data into 20 folds of
    ~5% each (stratified by dx, grouped by lesion_id), then give 3 folds to
    test (15%), 3 to val (15%), and the remaining 14 to train (70%).
    """
    n_folds = 20
    n_test = round(config.TEST_FRAC * n_folds)
    n_val = round(config.VAL_FRAC * n_folds)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    fold_of_row = np.empty(len(df), dtype=int)
    for fold, (_, idx) in enumerate(sgkf.split(df, df["dx"], groups=df["lesion_id"])):
        fold_of_row[idx] = fold

    split = np.where(fold_of_row < n_test, "test",
                     np.where(fold_of_row < n_test + n_val, "val", "train"))
    return pd.Series(split, index=df.index)


def main():
    root = download_dataset()

    meta_path = find_metadata(root)
    print(f"Metadata: {meta_path}")
    df = pd.read_csv(meta_path)
    df = df.drop_duplicates(subset="image_id").reset_index(drop=True)

    # Attach image paths; warn about metadata rows without an image on disk.
    index = build_image_index(root)
    df["image_path"] = df["image_id"].map(lambda i: str(index[i]) if i in index else None)
    missing = df["image_path"].isna()
    if missing.any():
        print(f"WARNING: {missing.sum()} metadata rows have no image file, e.g. "
              f"{df.loc[missing, 'image_id'].head(5).tolist()} -- dropping them.")
        df = df[~missing].reset_index(drop=True)

    # Map dx codes to display names; drop any unexpected code.
    unknown = ~df["dx"].isin(config.DX_TO_NAME)
    if unknown.any():
        print(f"WARNING: dropping {unknown.sum()} rows with unknown dx codes: "
              f"{sorted(df.loc[unknown, 'dx'].unique())}")
        df = df[~unknown].reset_index(drop=True)
    df["label"] = df["dx"].map(config.DX_TO_NAME)

    print("\nClass counts:")
    counts = df["label"].value_counts()
    for name, n in counts.items():
        print(f"  {name:<22} {n:>5}  ({n / len(df):6.1%})")
    print(f"  {'TOTAL':<22} {len(df):>5}")

    df["split"] = group_split(df)

    # Sanity check: no lesion may appear in more than one split.
    leaks = df.groupby("lesion_id")["split"].nunique().gt(1).sum()
    assert leaks == 0, f"{leaks} lesions appear in more than one split!"

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSplits (images / lesions):")
    for name, out in [("train", config.TRAIN_CSV), ("val", config.VAL_CSV), ("test", config.TEST_CSV)]:
        part = df[df["split"] == name]
        part[["image_path", "label"]].to_csv(out, index=False)
        print(f"  {name:<5} {len(part):>5} images ({len(part) / len(df):5.1%}), "
              f"{part['lesion_id'].nunique():>5} lesions -> {out}")
    print("No lesion overlaps between splits. Done.")


if __name__ == "__main__":
    main()
