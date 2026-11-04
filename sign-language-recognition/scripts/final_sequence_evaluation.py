"""Evaluate the CV-selected LSTM and GRU variants once on the frozen test split."""
from pathlib import Path
import json,random
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=ROOT/'out'
z=np.load(DATA/'preprocessed/sequences_v1.npz'); raw=z['raw'].astype(np.float32); angle=z['angles'].astype(np.float32)
base=np.concatenate([raw[:,:,99:225],angle[:,:,12:]],axis=2); mask=np.isfinite(base).astype(np.float32)
X=base; y=z['gloss'].astype(str); split=z['split'].astype(str); ids=z['video_id'].astype(str); tr=split=='train'; va=split=='val'; te=split=='test'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv=pd.read_csv(DATA/'sequence_model_search'/'grouped_cv_folds.csv'); epochs_file=cv[cv.variant.eq('hand_angles_mask')]
cv_summary=pd.read_csv(DATA/'sequence_model_search'/'summary.csv').set_index(['variant','model'])
def stats_fit(a):
 flat=a.reshape(-1,a.shape[-1]); med=np.nanmedian(flat,axis=0); med=np.where(np.isfinite(med),med,0); filled=np.where(np.isfinite(flat),flat,med); mean=filled.mean(0); std=filled.std(0); std=np.where(std>1e-6,std,1); return tuple(v.astype(np.float32) for v in (med,mean,std))
def apply(a,m,s):
 med,mean,std=s; x=(np.where(np.isfinite(a),a,med[None,None,:])-mean[None,None,:])/std[None,None,:]; return np.concatenate([x,m],axis=2).astype(np.float32)
class Net(nn.Module):
 def __init__(self,kind,k):
  super().__init__(); cls=nn.LSTM if kind=='lstm' else nn.GRU; self.rnn=cls(292,96,batch_first=True,bidirectional=True); self.drop=nn.Dropout(.35); self.head=nn.Linear(192,k)
 def forward(self,x):
  _,h=self.rnn(x); h=h[0] if isinstance(h,tuple) else h; return self.head(self.drop(torch.cat([h[-2],h[-1]],1)))
def train(kind,a,labels,epochs,classes):
 model=Net(kind,len(classes)).to(device); enc={c:i for i,c in enumerate(classes)}; target=np.array([enc[c] for c in labels],np.int64); count=np.bincount(target,minlength=len(classes)); w=np.sqrt(len(target)/np.maximum(count,1)); w/=w.mean()
 loss=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3); loader=DataLoader(TensorDataset(torch.from_numpy(a),torch.from_numpy(target)),batch_size=64,shuffle=True,pin_memory=device.type=='cuda')
 for _ in range(epochs):
  model.train()
  for xb,yb in loader:
   xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); v=loss(model(xb),yb); v.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
 return model
def predict(model,a,classes):
 model.eval()
 with torch.no_grad(): logits=torch.cat([model(torch.from_numpy(a[i:i+128]).to(device)).cpu() for i in range(0,len(a),128)]).numpy()
 return classes[logits.argmax(1)]
def metrics(true,pred):
 return {'accuracy':float(accuracy_score(true,pred)),'balanced_accuracy':float(balanced_accuracy_score(true,pred)),'macro_f1':float(f1_score(true,pred,labels=np.unique(true),average='macro',zero_division=0))}

for kind in ('lstm','gru'):
 folds=epochs_file[epochs_file.model.eq(kind)]; epochs=max(1,int(np.median(folds.best_epoch)))
 cvrow=cv_summary.loc[('hand_angles_mask',kind)]
 # Validation score: fit only on train; then locked final refit on train+validation and evaluate test.
 classes=np.unique(y[tr]); s=stats_fit(X[tr]); xt=apply(X[tr],mask[tr],s); xv=apply(X[va],mask[va],s)
 m=train(kind,xt,y[tr],epochs,classes); pv=predict(m,xv,classes); vm=metrics(y[va],pv)
 fitmask=tr|va; classes=np.unique(y[fitmask]); s=stats_fit(X[fitmask]); xf=apply(X[fitmask],mask[fitmask],s); xe=apply(X[te],mask[te],s)
 final=train(kind,xf,y[fitmask],epochs,classes); pe=predict(final,xe,classes); tm=metrics(y[te],pe)
 row={'model':kind,'input':'both hands + hand angles + missingness mask','frames':32,'hidden_size':96,'dropout':.35,'weight_decay':.001,'epochs_median_cv':epochs,
  'cv_accuracy':float(cvrow.cv_accuracy),'cv_macro_f1':float(cvrow.cv_macro_f1),'validation_accuracy':vm['accuracy'],'validation_macro_f1':vm['macro_f1'],
  'test_accuracy':tm['accuracy'],'test_balanced_accuracy':tm['balanced_accuracy'],'test_macro_f1':tm['macro_f1'],'test_used_for_selection':False}
 d=OUT/kind; d.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(d/'results.csv',index=False)
 torch.save({'state_dict':final.cpu().state_dict(),'classes':classes.tolist(),'normalizer':s,'kind':kind},d/f'{kind}.pt')
 print(row,flush=True)
