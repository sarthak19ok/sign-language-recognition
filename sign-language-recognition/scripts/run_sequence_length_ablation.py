"""Compare 16/32/64-frame representations from original landmark sequences."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=DATA/'ablation_study'; OUT.mkdir(parents=True,exist_ok=True)
meta=pd.read_csv(DATA/'all_metadata.csv',dtype={'video_id':str}); meta.video_id=meta.video_id.str.zfill(5)
split=pd.read_csv(DATA/'split_v2.csv',dtype={'video_id':str}); split.video_id=split.video_id.str.zfill(5)
meta=meta.drop(columns=['split','experimental_split'],errors='ignore').merge(split[['video_id','split']],on='video_id',validate='one_to_one')
paths={}
for source in ('landmarks_repo','landmarks_kaggle'):
 for p in (DATA/source).glob('*.npy'): paths[p.stem]=p
pose_angles=[(11,13,15),(12,14,16),(23,11,13),(24,12,14),(11,23,25),(12,24,26),(23,25,27),(24,26,28),(25,27,31),(26,28,32),(11,23,24),(12,24,23)]
hand_angles=[(0,1,2),(1,2,3),(0,5,6),(5,6,7),(0,9,10),(9,10,11),(0,13,14),(13,14,15),(0,17,18),(17,18,19)]
triplets=pose_angles+[(33+a,33+b,33+c) for a,b,c in hand_angles]+[(54+a,54+b,54+c) for a,b,c in hand_angles]
def fill(a):
 a=a.copy(); t=np.arange(len(a))
 for j in range(a.shape[1]):
  ok=np.isfinite(a[:,j]); bad=~ok; starts=np.where(bad&np.r_[True,~bad[:-1]])[0]; ends=np.where(bad&np.r_[~bad[1:],True])[0]
  for lo,hi in zip(starts,ends):
   if lo>0 and hi<len(a)-1 and hi-lo+1<=3:a[lo:hi+1,j]=np.interp(t[lo:hi+1],[lo-1,hi+1],[a[lo-1,j],a[hi+1,j]])
 return a
def resample(a,n):
 pos=np.linspace(0,len(a)-1,n); out=np.full((n,a.shape[1]),np.nan,np.float32)
 for j in range(a.shape[1]):
  good=np.flatnonzero(np.isfinite(a[:,j]))
  for k,q in enumerate(pos):
   right=np.searchsorted(good,q)
   if right<len(good) and good[right]==q:out[k,j]=a[good[right],j]
   elif right>0 and right<len(good):
    lo,hi=good[right-1],good[right]
    if hi-lo<=4:out[k,j]=np.interp(q,[lo,hi],[a[lo,j],a[hi,j]])
 return out
def normalize(a):
 xyz=a.reshape(-1,75,3).copy(); pose=xyz[:,:33]; center=np.nanmean(pose[:,[11,12]],axis=1); width=np.linalg.norm(pose[:,11]-pose[:,12],axis=1); width=np.where(np.isfinite(width)&(width>1e-5),width,np.nan)
 return ((xyz-center[:,None,:])/width[:,None,None]).reshape(-1,225)
def angle_seq(a):
 xyz=a.reshape(-1,75,3); cols=[]
 for i,j,k in triplets:
  u=xyz[:,i]-xyz[:,j]; v=xyz[:,k]-xyz[:,j]; d=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1); c=np.divide((u*v).sum(axis=1),d,out=np.full(len(a),np.nan),where=d>1e-8); cols.append(np.arccos(np.clip(c,-1,1)))
 return np.stack(cols,axis=1)
def pool(x):return np.concatenate([np.nanmean(x[:len(x)//2],axis=0),np.nanmean(x[len(x)//2:],axis=0)])

train=meta.split.eq('train').to_numpy(); y=meta.gloss.to_numpy(); groups=meta.signer_id.astype(str).to_numpy(); records=[]
for n in (16,32,64):
 print('sequence length',n,flush=True); X=[]
 for vid in meta.video_id:
  if vid not in paths: X.append(np.full(292,np.nan,np.float32)); continue
  a=normalize(fill(np.load(paths[vid]).astype(np.float32))); a=resample(a,n); ang=angle_seq(a)
  hands=a[:,99:225]; hand_ang=ang[:,12:]
  X.append(np.concatenate([pool(hands),pool(hand_ang)]).astype(np.float32))
 X=np.stack(X)
 if n==64: np.savez_compressed(OUT/'hand_angle_features_64.npz',X=X,video_id=meta.video_id.astype(str).to_numpy(dtype='U5'),gloss=meta.gloss.astype(str).to_numpy(dtype='U32'),signer_id=meta.signer_id.astype(str).to_numpy(dtype='U16'),split=meta.split.astype(str).to_numpy(dtype='U12'))
 for fold,(a,b) in enumerate(GroupKFold(3).split(X[train],y[train],groups[train]),1):
  model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('trees',ExtraTreesClassifier(n_estimators=450,class_weight='balanced',max_features=.25,min_samples_leaf=1,n_jobs=-1,random_state=42))])
  model.fit(X[train][a],y[train][a]); pred=model.predict(X[train][b]); present=np.unique(y[train][b])
  records.append({'frames':n,'fold':fold,'accuracy':accuracy_score(y[train][b],pred),'macro_f1':f1_score(y[train][b],pred,labels=present,average='macro',zero_division=0)})
df=pd.DataFrame(records); df.to_csv(OUT/'sequence_length_ablation.csv',index=False)
summary=df.groupby('frames').agg(mean_accuracy=('accuracy','mean'),std_accuracy=('accuracy','std'),mean_macro_f1=('macro_f1','mean')); summary.to_csv(OUT/'sequence_length_summary.csv'); print(summary.to_string())

