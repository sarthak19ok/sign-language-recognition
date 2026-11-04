"""Merge WLASL landmark indexes and produce first-pass quality EDA."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "WLASL100"
SOURCES = {"repo": DATA / "landmarks_repo", "kaggle": DATA / "landmarks_kaggle"}
OUT = DATA / "landmarks_eda"
OUT.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv(DATA / "all_metadata.csv", dtype={"video_id": str})
metadata["video_id"] = metadata.video_id.str.zfill(5)
split = pd.read_csv(DATA / "split_v2.csv", dtype={"video_id": str})
split["video_id"] = split.video_id.str.zfill(5)

indexes = []
for source, folder in SOURCES.items():
    idx = pd.read_csv(folder / "index.csv", dtype={"video_id": str})
    idx["video_id"] = idx.video_id.str.zfill(5)
    idx["landmark_source"] = source
    idx["npy_path"] = idx.video_id.map(lambda vid: str((folder / f"{vid}.npy").resolve()))
    indexes.append(idx)
index = pd.concat(indexes, ignore_index=True)
if index.video_id.duplicated().any():
    duplicates = index.loc[index.video_id.duplicated(False), "video_id"].tolist()
    raise ValueError(f"Duplicate landmark IDs across sources: {duplicates[:10]}")
index = index.merge(metadata.drop(columns=["split", "experimental_split"], errors="ignore"), on="video_id", how="left", validate="one_to_one")
index = index.merge(split.rename(columns={"split": "experimental_split"}), on="video_id", how="left", validate="one_to_one", suffixes=("", "_split"))
if index["experimental_split"].isna().any():
    raise ValueError("Some landmark IDs are missing split_v2 assignments")
index.to_csv(OUT / "combined_landmark_index.csv", index=False)

records = []
for row in index.itertuples(index=False):
    path = Path(row.npy_path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 225:
        raise ValueError(f"{row.video_id}: expected (T,225), got {arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    pose, left, right = arr[:, :99], arr[:, 99:162], arr[:, 162:225]
    finite = np.isfinite(arr)
    records.append({
        "video_id": row.video_id, "gloss": row.gloss, "signer_id": row.signer_id,
        "split": row.experimental_split, "landmark_source": row.landmark_source,
        "frames": len(arr), "duration_sec": row.seconds,
        "pose_missing_pct": 100 * (1 - np.isfinite(pose).mean()),
        "left_hand_missing_pct": 100 * (1 - np.isfinite(left).mean()),
        "right_hand_missing_pct": 100 * (1 - np.isfinite(right).mean()),
        "all_missing_pct": 100 * (1 - finite.mean()),
    })
quality = pd.DataFrame(records)
quality.to_csv(OUT / "per_video_quality.csv", index=False)
quality.groupby("landmark_source").agg(
    videos=("video_id", "count"), median_frames=("frames", "median"),
    median_pose_missing_pct=("pose_missing_pct", "median"),
    median_left_hand_missing_pct=("left_hand_missing_pct", "median"),
    median_right_hand_missing_pct=("right_hand_missing_pct", "median"),
).to_csv(OUT / "source_summary.csv")
quality.groupby("gloss").agg(videos=("video_id", "count"), signers=("signer_id", "nunique"),
    median_frames=("frames", "median"), median_missing_pct=("all_missing_pct", "median")).to_csv(OUT / "class_summary.csv")
quality.groupby("signer_id").agg(videos=("video_id", "count"), classes=("gloss", "nunique"),
    median_frames=("frames", "median"), median_missing_pct=("all_missing_pct", "median")).to_csv(OUT / "signer_summary.csv")

summary = {
    "videos": int(len(quality)), "classes": int(quality.gloss.nunique()),
    "signers": int(quality.signer_id.nunique()),
    "source_counts": {str(k): int(v) for k, v in quality.landmark_source.value_counts().items()},
    "split_counts": {str(k): int(v) for k, v in quality.split.value_counts().items()},
    "array_shape": "(T,225), float32 expected",
    "median_frames": float(quality.frames.median()),
    "median_missing_pct": float(quality.all_missing_pct.median()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

