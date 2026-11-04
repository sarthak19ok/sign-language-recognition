"""Compare LSTM/GRU inputs: coordinates, missing masks, and velocity."""
from pathlib import Path
import json,random
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=DATA/'sequence_model_search'; OUT.mkdir(parents=True,exist_ok=True)
z=np.load(DATA/'preprocessed/sequences_v1.npz'); raw=z['raw'].astype(np.float32); angle=z['angles'].astype(np.float32); velocity=z['velocity'].astype(np.float32)
base=np.concatenate([raw[:,:,99:225],angle[:,:,12:]],axis=2); mask=np.isfinite(base).astype(np.float32)
angle_velocity=np.diff(angle[:,:,12:],axis=1,prepend=angle[:,:1,12:])*(base.shape[1]-1)
dynamic=np.concatenate([base,velocity[:,:,99:225],angle_velocity],axis=2)
variants={'hand_angles':(base,None),'hand_angles_mask':(base,mask),'hand_angles_velocity':(dynamic,None),'hand_angles_velocity_mask':(dynamic,np.concatenate([mask,np.isfinite(dynamic[:,:,base.shape[2]:]).astype(np.float32)],axis=2))}
y=z['gloss'].astype(str); signer=z['signer_id'].astype(str); split=z['split'].astype(str); ids=z['video_id'].astype(str); train=split=='train'; val=split=='val'; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',device,torch.cuda.get_device_name(0) if device.type=='cuda' else '')

class Net(nn.Module):
 def __init__(self,kind,nclasses,nfeatures,hidden=96):
  super().__init__(); cls=nn.LSTM if kind=='lstm' else nn.GRU
  self.rnn=cls(input_size=nfeatures,hidden_size=hidden,batch_first=True,bidirectional=True)
  self.drop=nn.Dropout(.35); self.head=nn.Linear(hidden*2,nclasses)
 def forward(self,x):
  _,h=self.rnn(x); h=h[0] if isinstance(h,tuple) else h
  return self.head(self.drop(torch.cat([h[-2],h[-1]],dim=1)))
def norm_fit(x):
 f=x.reshape(-1,x.shape[-1]); med=np.nanmedian(f,axis=0); med=np.where(np.isfinite(med),med,0)
 fill=np.where(np.isfinite(f),f,med); mean=fill.mean(0); std=fill.std(0); std=np.where(std>1e-6,std,1)
 return tuple(a.astype(np.float32) for a in (med,mean,std))
def norm_apply(x,s):
 med,mean,std=s; return ((np.where(np.isfinite(x),x,med[None,None,:])-mean[None,None,:])/std[None,None,:]).astype(np.float32)
def pack(x,mask,s):
 x=norm_apply(x,s)
 return np.concatenate([x,mask],axis=2) if mask is not None else x
def fit(kind,nfeat,xtr,ytr,xstop,ystop,ncls):
 model=Net(kind,ncls,nfeat).to(device); counts=np.bincount(ytr,minlength=ncls); w=np.sqrt(len(ytr)/np.maximum(counts,1)); w/=w.mean()
 loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
 loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=64,shuffle=True,pin_memory=device.type=='cuda')
 sx=torch.from_numpy(xstop).to(device); sy=torch.from_numpy(ystop).to(device); best=float('inf'); state=None; best_epoch=0; stale=0
 for epoch in range(1,81):
  model.train()
  for xb,yb in loader:
   xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
  model.eval()
  with torch.no_grad(): score=float(loss_fn(model(sx),sy))
  if score<best-1e-4: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; stale=0
  else: stale+=1
  if stale>=10: break
 model.load_state_dict(state); return model,best_epoch
def predict(model,x):
 model.eval()
 with torch.no_grad(): return torch.cat([model(torch.from_numpy(x[i:i+128]).to(device)).cpu() for i in range(0,len(x),128)]).numpy().argmax(1)

folds=list(GroupKFold(3).split(np.zeros(train.sum()),y[train],signer[train])); records=[]; epochs={}
for vname,(cont,mask) in variants.items():
 epochs[vname]={}
 for kind in ('lstm','gru'):
  epochs[vname][kind]=[]
  print('variant/model',vname,kind,flush=True)
  for fold,(outer_a,outer_b) in enumerate(folds,1):
   ids_train=np.flatnonzero(train); outer_tr=ids_train[outer_a]; outer_va=ids_train[outer_b]
   sub,stop=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED+fold).split(cont[outer_tr],y[outer_tr],signer[outer_tr]))
   fit_ids=outer_tr[sub]; stop_ids=outer_tr[stop]; classes=np.unique(y[fit_ids]); enc={c:i for i,c in enumerate(classes)}
   known=np.array([c in enc for c in y[stop_ids]]); stop_ids=stop_ids[known]
   ya=np.array([enc[c] for c in y[fit_ids]],np.int64); ys=np.array([enc[c] for c in y[stop_ids]],np.int64)
   stats=norm_fit(cont[fit_ids]); xa=pack(cont[fit_ids],None if mask is None else mask[fit_ids],stats); xs=pack(cont[stop_ids],None if mask is None else mask[stop_ids],stats); xv=pack(cont[outer_va],None if mask is None else mask[outer_va],stats)
   model,epoch=fit(kind,xa.shape[2],xa,ya,xs,ys,len(classes)); pred=classes[predict(model,xv)]; truth=y[outer_va]; present=np.unique(truth)
   records.append({'variant':vname,'model':kind,'fold':fold,'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,labels=present,average='macro',zero_division=0),'best_epoch':epoch})
   epochs[vname][kind].append(epoch); print(' fold',fold,'acc',round(records[-1]['accuracy'],3),'macro',round(records[-1]['macro_f1'],3),'epoch',epoch,flush=True)
df=pd.DataFrame(records); df.to_csv(OUT/'grouped_cv_folds.csv',index=False)
summary=df.groupby(['variant','model']).agg(cv_accuracy=('accuracy','mean'),cv_accuracy_std=('accuracy','std'),cv_macro_f1=('macro_f1','mean'),median_best_epoch=('best_epoch','median')).sort_values('cv_macro_f1',ascending=False); summary.to_csv(OUT/'summary.csv'); print(summary.to_string())
vname,kind=summary.index[0]; cont,mask=variants[vname]; tr_ids=np.flatnonzero(train); va_ids=np.flatnonzero(val); classes=np.unique(y[tr_ids]); enc={c:i for i,c in enumerate(classes)}; ya=np.array([enc[c] for c in y[tr_ids]],np.int64)
stats=norm_fit(cont[tr_ids]); xa=pack(cont[tr_ids],None if mask is None else mask[tr_ids],stats); xv=pack(cont[va_ids],None if mask is None else mask[va_ids],stats)
final=Net(kind,len(classes),xa.shape[2]).to(device); cw=np.sqrt(len(ya)/np.maximum(np.bincount(ya,minlength=len(classes)),1)); cw/=cw.mean(); loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(final.parameters(),lr=8e-4,weight_decay=1e-3)
loader=DataLoader(TensorDataset(torch.from_numpy(xa),torch.from_numpy(ya)),batch_size=64,shuffle=True,pin_memory=device.type=='cuda'); n_epochs=max(1,int(np.median(epochs[vname][kind])))
for _ in range(n_epochs):
 final.train()
 for xb,yb in loader:
  xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=loss_fn(final(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(final.parameters(),1); opt.step()
pred=classes[predict(final,xv)]; truth=y[va_ids]
metrics={'selected_variant':vname,'selected_model':kind,'cv_accuracy':float(summary.loc[(vname,kind),'cv_accuracy']),'cv_macro_f1':float(summary.loc[(vname,kind),'cv_macro_f1']),'training_epochs':n_epochs,'validation_accuracy':float(accuracy_score(truth,pred)),'validation_balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'validation_macro_f1':float(f1_score(truth,pred,labels=np.unique(truth),average='macro',zero_division=0)),'validation_n':len(va_ids),'test_split_used':False,'device':str(device)}
torch.save({'state_dict':final.cpu().state_dict(),'classes':classes.tolist(),'normalizer':stats,'kind':kind,'variant':vname},OUT/f'{kind}_{vname}.pt')
(OUT/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)); pd.DataFrame({'video_id':ids[va_ids],'true_gloss':truth,'prediction':pred}).to_csv(OUT/'validation_predictions.csv',index=False); print(json.dumps(metrics,indent=2))
