"""Merge WLASL landmark indexes and produce first-pass quality EDA."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
pca_vectors = []
for row in index.itertuples(index=False):
    path = Path(row.npy_path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 225:
        raise ValueError(f"{row.video_id}: expected (T,225), got {arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    pose, left, right = arr[:, :99], arr[:, 99:162], arr[:, 162:225]
    finite = np.isfinite(arr)
    # Descriptive motion: mean finite per-coordinate frame displacement.
    delta = np.diff(arr, axis=0)
    motion = float(np.nanmean(np.linalg.norm(delta.reshape(len(delta), 75, 3), axis=2))) if len(delta) else 0.0
    # Per-frame shoulder-centred and shoulder-width-scaled pose+hand vector for exploratory PCA.
    xyz = arr.reshape(-1, 75, 3).copy()
    pose_xyz = arr[:, :99].reshape(-1, 33, 3)
    shoulders = pose_xyz[:, [11, 12], :]
    center = np.nanmean(shoulders, axis=1)
    width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    width = np.where(np.isfinite(width) & (width > 1e-5), width, np.nan)
    normalized = (xyz - center[:, None, :]) / width[:, None, None]
    pca_vectors.append(np.nanmean(normalized, axis=0).reshape(-1))
    records.append({
        "video_id": row.video_id, "gloss": row.gloss, "signer_id": row.signer_id,
        "split": row.experimental_split, "landmark_source": row.landmark_source,
        "frames": len(arr), "duration_sec": row.seconds,
        "pose_missing_pct": 100 * (1 - np.isfinite(pose).mean()),
        "left_hand_missing_pct": 100 * (1 - np.isfinite(left).mean()),
        "right_hand_missing_pct": 100 * (1 - np.isfinite(right).mean()),
        "all_missing_pct": 100 * (1 - finite.mean()), "mean_frame_motion": motion,
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

# Confirm that the intended signer-independent split is actually disjoint.
signer_sets = {name: set(quality.loc[quality.split == name, "signer_id"].astype(str)) for name in ("train", "val", "test")}
overlap = {f"{a}_{b}": sorted(signer_sets[a] & signer_sets[b]) for a,b in (("train","val"),("train","test"),("val","test"))}
if any(overlap.values()):
    raise ValueError(f"Signer leakage across split_v2: {overlap}")

quality.groupby("gloss").agg(mean_motion=("mean_frame_motion", "mean"), median_motion=("mean_frame_motion", "median")).to_csv(OUT / "motion_by_class.csv")
quality.groupby("signer_id").agg(mean_motion=("mean_frame_motion", "mean"), median_motion=("mean_frame_motion", "median")).to_csv(OUT / "motion_by_signer.csv")

# Exploratory only: descriptive PCA uses all available videos for visualization; never reuse
# this fitted transform in a predictive evaluation. Model PCA/scaling must be fold-local.
X = SimpleImputer(strategy="median").fit_transform(np.asarray(pca_vectors, dtype=np.float32))
X = StandardScaler().fit_transform(X)
Z = PCA(n_components=2, random_state=42).fit_transform(X)
for column, filename, title in (("gloss", "pca_by_gloss.png", "Exploratory PCA colored by gloss"), ("signer_id", "pca_by_signer.png", "Exploratory PCA colored by signer")):
    labels = quality[column].astype(str)
    codes, categories = pd.factorize(labels, sort=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=codes, cmap="turbo", s=14, alpha=.75, linewidths=0)
    ax.set(title=title + " (all available clips; descriptive only)", xlabel="PC1", ylabel="PC2")
    if len(categories) <= 25:
        cb=fig.colorbar(scatter, ax=ax, ticks=np.arange(len(categories))); cb.ax.set_yticklabels(categories)
    else:
        ax.text(.99,.01,f"{len(categories)} categories; color IDs are not individually labeled",transform=ax.transAxes,ha="right",va="bottom",fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / filename,dpi=180); plt.close(fig)

summary = {
    "videos": int(len(quality)), "classes": int(quality.gloss.nunique()),
    "signers": int(quality.signer_id.nunique()),
    "source_counts": {str(k): int(v) for k, v in quality.landmark_source.value_counts().items()},
    "split_counts": {str(k): int(v) for k, v in quality.split.value_counts().items()},
    "array_shape": "(T,225), float32 expected",
    "median_frames": float(quality.frames.median()),
    "median_missing_pct": float(quality.all_missing_pct.median()),
    "median_mean_frame_motion": float(quality.mean_frame_motion.median()),
    "signer_overlap": overlap,
    "pca_note": "Exploratory descriptive PCA over all clips; not for model evaluation.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

