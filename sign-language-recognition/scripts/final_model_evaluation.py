"""Compare six classifiers using signer-grouped CV and save one final result CSV per model."""
from pathlib import Path
import itertools,json,time
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=ROOT/'out'
z=np.load(DATA/'ablation_study'/'hand_angle_features_64.npz'); X=z['X'].astype(np.float32)
y=z['gloss'].astype(str); groups=z['signer_id'].astype(str); split=z['split'].astype(str); ids=z['video_id'].astype(str)
train=split=='train'; val=split=='val'; test=split=='test'; classes=np.unique(y[train])
if set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]): raise ValueError('Signer overlap in frozen splits')

grids={
 'logistic_regression':[{'C':c} for c in (.1,1,10)],
 'knn':[{'n_neighbors':k,'weights':w} for k,w in itertools.product((3,5,9),('uniform','distance'))],
 'rbf_svm':[{'C':c,'gamma':g} for c,g in itertools.product((10,100,300),('scale',.003,.01))],
 'random_forest':[{'max_features':f,'min_samples_leaf':l} for f,l in itertools.product(('sqrt',.5),(1,2))],
 'extra_trees':[{'max_features':f,'min_samples_leaf':l,'class_weight':cw} for f,l,cw in itertools.product((.25,.5,.75),(1,2),('balanced',None))],
 'xgboost':[{'max_depth':d,'n_estimators':n} for d,n in itertools.product((2,3),(80,140))],
}
def sklearn_model(family,p):
 imp=('imputer',SimpleImputer(strategy='median',keep_empty_features=True))
 if family=='logistic_regression':return Pipeline([imp,('scale',StandardScaler()),('model',LogisticRegression(C=p['C'],class_weight='balanced',max_iter=3000))])
 if family=='knn':return Pipeline([imp,('scale',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=p['n_neighbors'],weights=p['weights'],n_jobs=-1))])
 if family=='rbf_svm':return Pipeline([imp,('scale',StandardScaler()),('model',SVC(C=p['C'],gamma=p['gamma'],kernel='rbf',class_weight='balanced'))])
 cls=ExtraTreesClassifier if family=='extra_trees' else RandomForestClassifier
 opts={'n_estimators':450,'max_features':p['max_features'],'min_samples_leaf':p['min_samples_leaf'],'n_jobs':-1,'random_state':42}
 if family=='extra_trees':opts['class_weight']=p['class_weight']
 else:opts['class_weight']='balanced_subsample'
 return Pipeline([imp,('model',cls(**opts))])
def xgb_model(p,n_classes):
 return XGBClassifier(objective='multi:softprob',num_class=n_classes,eval_metric='mlogloss',learning_rate=.1,subsample=.85,colsample_bytree=.8,min_child_weight=2,reg_lambda=5,max_bin=64,tree_method='hist',n_jobs=4,random_state=42,**p)
def score(ytrue,pred):return {'accuracy':accuracy_score(ytrue,pred),'balanced_accuracy':balanced_accuracy_score(ytrue,pred),'macro_f1':f1_score(ytrue,pred,labels=np.unique(ytrue),average='macro',zero_division=0)}

OUT.mkdir(exist_ok=True); cv_splits=list(GroupKFold(3).split(X[train],y[train],groups[train])); summaries=[]
for family,configs in grids.items():
 print('Tuning',family,flush=True); rows=[]
 for ci,params in enumerate(configs,1):
  fold_rows=[]
  for fold,(a,b) in enumerate(cv_splits,1):
   tic=time.time(); ytrue=y[train][b]
   if family=='xgboost':
    imp=SimpleImputer(strategy='median',keep_empty_features=True); xa=imp.fit_transform(X[train][a]); xb=imp.transform(X[train][b]); cls=np.unique(y[train][a]); enc={v:i for i,v in enumerate(cls)}; ya=np.array([enc[v] for v in y[train][a]])
    cnt=np.bincount(ya,minlength=len(cls)); sw=1/np.maximum(cnt[ya],1); sw*=len(sw)/sw.sum(); model=xgb_model(params,len(cls)); model.fit(xa,ya,sample_weight=sw); pred=cls[model.predict(xb).astype(int)]
   else:
    model=sklearn_model(family,params); model.fit(X[train][a],y[train][a]); pred=model.predict(X[train][b])
   m=score(ytrue,pred); fold_rows.append(m); rows.append({'config_id':ci,'params':json.dumps(params,sort_keys=True),'fold':fold,**m,'seconds':time.time()-tic})
  # parameter selection uses only signer-grouped training folds
  configs[ci-1]['_cv_macro_f1']=float(np.mean([r['macro_f1'] for r in fold_rows])); configs[ci-1]['_cv_accuracy']=float(np.mean([r['accuracy'] for r in fold_rows]))
 best=max(range(len(configs)),key=lambda i:(configs[i]['_cv_macro_f1'],configs[i]['_cv_accuracy']))
 selected={k:v for k,v in configs[best].items() if not k.startswith('_')}
 # Validation measures the locked CV-selected configuration. Test is used only after this point.
 if family=='xgboost':
  imp=SimpleImputer(strategy='median',keep_empty_features=True); xtr=imp.fit_transform(X[train]); xv=imp.transform(X[val]); cls=np.unique(y[train]); enc={v:i for i,v in enumerate(cls)}; ya=np.array([enc[v] for v in y[train]])
  cnt=np.bincount(ya,minlength=len(cls)); sw=1/np.maximum(cnt[ya],1); sw*=len(sw)/sw.sum(); vm=xgb_model(selected,len(cls)); vm.fit(xtr,ya,sample_weight=sw); val_pred=cls[vm.predict(xv).astype(int)]
  # Refit from scratch on train+validation before final held-out test evaluation.
  fitmask=train|val; xt=imp.fit_transform(X[fitmask]); xe=imp.transform(X[test]); cls=np.unique(y[fitmask]); enc={v:i for i,v in enumerate(cls)}; yt=np.array([enc[v] for v in y[fitmask]])
  cnt=np.bincount(yt,minlength=len(cls)); sw=1/np.maximum(cnt[yt],1); sw*=len(sw)/sw.sum(); final=xgb_model(selected,len(cls)); final.fit(xt,yt,sample_weight=sw); test_pred=cls[final.predict(xe).astype(int)]
 else:
  vm=sklearn_model(family,selected); vm.fit(X[train],y[train]); val_pred=vm.predict(X[val]); fitmask=train|val; final=sklearn_model(family,selected); final.fit(X[fitmask],y[fitmask]); test_pred=final.predict(X[test])
 valm=score(y[val],val_pred); testm=score(y[test],test_pred)
 cvmean=[r['macro_f1'] for r in rows if r['config_id']==best+1]; cvacc=[r['accuracy'] for r in rows if r['config_id']==best+1]
 result={'model':family,'feature_set':'both hands + hand angles; 64 frames; two temporal bins','params':json.dumps(selected,sort_keys=True),'cv_macro_f1_mean':float(np.mean(cvmean)),'cv_accuracy_mean':float(np.mean(cvacc)),
  'validation_accuracy':float(valm['accuracy']),'validation_macro_f1':float(valm['macro_f1']),'test_accuracy':float(testm['accuracy']),'test_balanced_accuracy':float(testm['balanced_accuracy']),'test_macro_f1':float(testm['macro_f1']),
  'n_train':int(train.sum()),'n_validation':int(val.sum()),'n_test':int(test.sum()),'cv':'GroupKFold(3), signer grouped','test_used_for_selection':False}
 outdir=OUT/family; outdir.mkdir(parents=True,exist_ok=True); pd.DataFrame([result]).to_csv(outdir/'results.csv',index=False)
 summaries.append(result); print(family,'CV',round(result['cv_accuracy_mean'],3),'val',round(result['validation_accuracy'],3),'test',round(result['test_accuracy'],3),flush=True)

# An overall comparison is a Markdown/notebook table; keep only one CSV in each model folder.
best_family=max(summaries,key=lambda r:(r['cv_macro_f1_mean'],r['cv_accuracy_mean']))['model']
pd.DataFrame(summaries).sort_values(['cv_macro_f1_mean','cv_accuracy_mean'],ascending=False).to_csv(OUT/'comparison.csv',index=False)
(OUT/'final_selection.json').write_text(json.dumps({'selected_by_grouped_cv':best_family,'test_opened_after_model_selection':True,'note':'No model or parameter was selected using test metrics.'},indent=2))
print('Best by grouped-CV macro-F1:',best_family)
