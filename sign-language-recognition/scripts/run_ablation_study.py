"""Run planned grouped-CV ablations: hands, angles, velocity, PCA, split strategy."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import GroupKFold,StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=DATA/'ablation_study'; OUT.mkdir(parents=True,exist_ok=True)
z=np.load(DATA/'preprocessed/sequences_v1.npz'); raw=z['raw'].astype(np.float32); vel=z['velocity'].astype(np.float32); angles=z['angles'].astype(np.float32)
y=z['gloss'].astype(str); signer=z['signer_id'].astype(str); split=z['split'].astype(str); tr=split=='train'
def two_bins(a): return np.concatenate([np.nanmean(a[:,:16],axis=1),np.nanmean(a[:,16:],axis=1)],axis=1)
left=raw[:,:,99:162]; right=raw[:,:,162:225]; left_ang=angles[:,:,12:22]; right_ang=angles[:,:,22:]
features={
 'left_hand_coords_angles':np.concatenate([two_bins(left),two_bins(left_ang)],axis=1),
 'right_hand_coords_angles':np.concatenate([two_bins(right),two_bins(right_ang)],axis=1),
 'both_hands_coords':two_bins(raw[:,:,99:225]),
 'both_hands_coords_angles':np.concatenate([two_bins(raw[:,:,99:225]),two_bins(angles[:,:,12:])],axis=1),
 'both_hands_coords_angles_velocity':np.concatenate([two_bins(raw[:,:,99:225]),two_bins(angles[:,:,12:]),two_bins(vel[:,:,99:225])],axis=1),
}
def et(): return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('model',ExtraTreesClassifier(n_estimators=450,class_weight='balanced',max_features=.25,min_samples_leaf=1,n_jobs=-1,random_state=42))])
rows=[]
for name,X in features.items():
 for fold,(a,b) in enumerate(GroupKFold(3).split(X[tr],y[tr],signer[tr]),1):
  model=et(); model.fit(X[tr][a],y[tr][a]); pred=model.predict(X[tr][b]); present=np.unique(y[tr][b])
  rows.append({'ablation':'feature_channels','setting':name,'fold':fold,'accuracy':accuracy_score(y[tr][b],pred),'macro_f1':f1_score(y[tr][b],pred,labels=present,average='macro',zero_division=0)})
channel=pd.DataFrame(rows); channel.to_csv(OUT/'channel_ablation.csv',index=False)

# PCA component sweep on flattened, shoulder-normalized hand+angle sequences.
flat=np.concatenate([raw[:,:,99:225],angles[:,:,12:]],axis=2).reshape(len(raw),-1); pca_rows=[]
for ncomp in (32,64,100,160):
 for fold,(a,b) in enumerate(GroupKFold(3).split(flat[tr],y[tr],signer[tr]),1):
  model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('pca',PCA(n_components=ncomp,svd_solver='randomized',random_state=42)),('logreg',LogisticRegression(C=1,class_weight='balanced',max_iter=3000))])
  model.fit(flat[tr][a],y[tr][a]); pred=model.predict(flat[tr][b]); present=np.unique(y[tr][b])
  pca_rows.append({'components':ncomp,'fold':fold,'accuracy':accuracy_score(y[tr][b],pred),'macro_f1':f1_score(y[tr][b],pred,labels=present,average='macro',zero_division=0)})
pca=pd.DataFrame(pca_rows); pca.to_csv(OUT/'pca_sweep.csv',index=False)

# Diagnostic only: random clip split can put the same signer in both partitions.
X=features['both_hands_coords_angles']; random_rows=[]
for seed,(a,b) in enumerate(StratifiedShuffleSplit(n_splits=3,test_size=.2,random_state=42).split(X[tr],y[tr]),42):
 model=et(); model.fit(X[tr][a],y[tr][a]); pred=model.predict(X[tr][b]); present=np.unique(y[tr][b]); overlap=len(set(signer[tr][a])&set(signer[tr][b]))
 random_rows.append({'seed':seed,'accuracy':accuracy_score(y[tr][b],pred),'macro_f1':f1_score(y[tr][b],pred,labels=present,average='macro',zero_division=0),'signer_overlap_count':overlap,'is_final_evaluation':False})
pd.DataFrame(random_rows).to_csv(OUT/'random_split_diagnostic.csv',index=False)

def summary(df,key): return df.groupby(key).agg(mean_accuracy=('accuracy','mean'),std_accuracy=('accuracy','std'),mean_macro_f1=('macro_f1','mean')).sort_values('mean_macro_f1',ascending=False)
channel_summary=summary(channel,'setting'); pca_summary=summary(pca,'components'); random_summary=summary(pd.DataFrame(random_rows),'seed')
channel_summary.to_csv(OUT/'channel_summary.csv'); pca_summary.to_csv(OUT/'pca_summary.csv')
result={'feature_channel_best':str(channel_summary.index[0]),'feature_channel_best_cv_accuracy':float(channel_summary.iloc[0].mean_accuracy),'pca_best_components':int(pca_summary.index[0]),'pca_best_cv_accuracy':float(pca_summary.iloc[0].mean_accuracy),'grouped_cv_accuracy':float(channel_summary.loc['both_hands_coords_angles'].mean_accuracy),'random_clip_split_mean_accuracy':float(np.mean([r['accuracy'] for r in random_rows])),'random_split_signer_overlap_per_seed':[int(r['signer_overlap_count']) for r in random_rows],'random_split_warning':'diagnostic only; not valid for model selection or final claims'}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(channel_summary.to_string()); print('\nPCA sweep\n',pca_summary.to_string()); print('\nSplit diagnostic\n',pd.DataFrame(random_rows).to_string(index=False)); print(json.dumps(result,indent=2))
