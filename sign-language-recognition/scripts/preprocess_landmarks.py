"""Prepare fixed-length, shoulder-normalized WLASL landmark sequences."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "WLASL100"
OUT = DATA / "preprocessed"
OUT.mkdir(parents=True, exist_ok=True)
N, MAX_GAP = 32, 3

split = pd.read_csv(DATA / "split_v2.csv", dtype={"video_id": str})
split.video_id = split.video_id.str.zfill(5)
meta = pd.read_csv(DATA / "all_metadata.csv", dtype={"video_id": str})
meta.video_id = meta.video_id.str.zfill(5)
meta = meta.drop(columns=["split", "experimental_split"], errors="ignore").merge(split[["video_id", "split"]], on="video_id", validate="one_to_one")
paths = {}
for source in ("landmarks_repo", "landmarks_kaggle"):
    for p in (DATA / source).glob("*.npy"):
        if p.stem in paths: raise ValueError(f"Duplicate ID {p.stem}")
        paths[p.stem] = (p, source)

def short_gap_fill(a):
    a=a.copy(); t=np.arange(len(a))
    for j in range(a.shape[1]):
        good=np.isfinite(a[:,j]); bad=~good
        starts=np.where(bad & np.r_[True,~bad[:-1]])[0]; ends=np.where(bad & np.r_[~bad[1:],True])[0]
        for lo,hi in zip(starts,ends):
            if lo>0 and hi<len(a)-1 and hi-lo+1<=MAX_GAP:
                a[lo:hi+1,j]=np.interp(t[lo:hi+1],[lo-1,hi+1],[a[lo-1,j],a[hi+1,j]])
    return a

def resample(a):
    # Interpolate only across bounded short gaps; retain long missing intervals as NaN.
    positions=np.linspace(0,len(a)-1,N); out=np.full((N,a.shape[1]),np.nan,np.float32)
    for j in range(a.shape[1]):
        good=np.flatnonzero(np.isfinite(a[:,j]))
        if not len(good): continue
        for k,pos in enumerate(positions):
            right=np.searchsorted(good,pos,side="left")
            if right<len(good) and good[right]==pos:
                out[k,j]=a[good[right],j]
            elif right>0 and right<len(good):
                lo,hi=good[right-1],good[right]
                if hi-lo<=MAX_GAP+1:
                    w=(pos-lo)/(hi-lo); out[k,j]=(1-w)*a[lo,j]+w*a[hi,j]
    return out

def normalize(a):
    xyz=a.reshape(-1,75,3).copy(); pose=xyz[:,:33]
    center=np.nanmean(pose[:,[11,12]],axis=1)
    width=np.linalg.norm(pose[:,11]-pose[:,12],axis=1)
    width=np.where(np.isfinite(width)&(width>1e-5),width,np.nan)
    return ((xyz-center[:,None,:])/width[:,None,None]).reshape(-1,225).astype(np.float32)

pose_angles=[(11,13,15),(12,14,16),(23,11,13),(24,12,14),(11,23,25),(12,24,26),(23,25,27),(24,26,28),(25,27,31),(26,28,32),(11,23,24),(12,24,23)]
hand_angles=[(0,1,2),(1,2,3),(0,5,6),(5,6,7),(0,9,10),(9,10,11),(0,13,14),(13,14,15),(0,17,18),(17,18,19)]
triplets=pose_angles+[(33+a,33+b,33+c) for a,b,c in hand_angles]+[(54+a,54+b,54+c) for a,b,c in hand_angles]
def angles(a):
    xyz=a.reshape(-1,75,3); out=[]
    for i,j,k in triplets:
        u=xyz[:,i]-xyz[:,j]; v=xyz[:,k]-xyz[:,j]; d=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        c=np.divide((u*v).sum(axis=1),d,out=np.full(len(a),np.nan),where=d>1e-8)
        out.append(np.arccos(np.clip(c,-1,1)))
    return np.stack(out,axis=1).astype(np.float32)

raw=[]; velocity=[]; joint_angles=[]; rows=[]
for r in meta.itertuples(index=False):
    vid=str(r.video_id).zfill(5)
    if vid not in paths: continue
    p,source=paths[vid]; a=np.load(p).astype(np.float32)
    if a.ndim!=2 or a.shape[1]!=225 or not len(a): raise ValueError(f"Bad array {vid}: {a.shape}")
    before=float(np.isnan(a).mean()); a=resample(normalize(short_gap_fill(a)))
    raw.append(a); velocity.append(np.diff(a,axis=0,prepend=a[:1])*(N-1)); joint_angles.append(angles(a))
    rows.append({"video_id":vid,"gloss":r.gloss,"signer_id":str(r.signer_id),"split":r.split,"source":source,"frames":N,"missing_before_pct":100*before,"missing_after_pct":100*np.isnan(a).mean()})
df=pd.DataFrame(rows); df.to_csv(OUT/"manifest.csv",index=False)
bundle={"video_id":df.video_id.to_numpy(dtype="U5"),"gloss":df.gloss.to_numpy(dtype="U32"),"signer_id":df.signer_id.to_numpy(dtype="U16"),"split":df.split.to_numpy(dtype="U12"),"raw":np.stack(raw),"velocity":np.stack(velocity),"angles":np.stack(joint_angles)}
np.savez_compressed(OUT/"sequences_v1.npz",**bundle)
summary={"n_videos":len(df),"split_counts":df.split.value_counts().to_dict(),"frames":N,"raw_shape":list(bundle["raw"].shape),"velocity_shape":list(bundle["velocity"].shape),"angles_shape":list(bundle["angles"].shape),"angle_features":len(triplets),"short_gap_limit_frames":MAX_GAP,"median_missing_before_pct":float(df.missing_before_pct.median()),"median_missing_after_pct":float(df.missing_after_pct.median()),"normalization":"per-sequence shoulder centre and shoulder width","scaler_fit":"No dataset-level imputation/scaling; fit within model training folds"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

