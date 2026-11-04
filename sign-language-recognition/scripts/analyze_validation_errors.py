"""Compare validation errors for the current classical and recurrent candidates."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'; OUT=DATA/'error_analysis'; OUT.mkdir(parents=True,exist_ok=True)
sources={
 'extra_trees':DATA/'refined_classical_search'/'validation_predictions.csv',
 'lstm_mask':DATA/'sequence_model_search'/'validation_predictions.csv',
}
all_summary={}; confusions=[]
for name,path in sources.items():
 d=pd.read_csv(path,dtype={'video_id':str,'true_gloss':str,'prediction':str})
 if 'prediction' not in d and 'predicted_gloss' in d: d=d.rename(columns={'predicted_gloss':'prediction'})
 if d.video_id.duplicated().any(): raise ValueError(f'Duplicate validation IDs in {name}')
 labels=sorted(set(d.true_gloss)|set(d.prediction))
 report=classification_report(d.true_gloss,d.prediction,labels=labels,output_dict=True,zero_division=0)
 per=pd.DataFrame([{'gloss':g,'support':int(report[g]['support']),'precision':report[g]['precision'],'recall':report[g]['recall'],'f1':report[g]['f1-score']} for g in labels])
 per=per.sort_values(['recall','support']).reset_index(drop=True); per.to_csv(OUT/f'{name}_per_class.csv',index=False)
 cm=confusion_matrix(d.true_gloss,d.prediction,labels=labels); pairs=[]
 for i,t in enumerate(labels):
  for j,p in enumerate(labels):
   if i!=j and cm[i,j]: pairs.append({'true_gloss':t,'predicted_gloss':p,'count':int(cm[i,j])})
 pairs=pd.DataFrame(pairs).sort_values('count',ascending=False) if pairs else pd.DataFrame(columns=['true_gloss','predicted_gloss','count'])
 pairs.head(30).to_csv(OUT/f'{name}_top_confusions.csv',index=False)
 pairs.assign(pair=lambda x:x.true_gloss+' → '+x.predicted_gloss).head(15).iloc[::-1].plot.barh(x='pair',y='count',legend=False,figsize=(9,6),color='#4477AA')
 plt.title(f'{name}: most common validation confusions'); plt.xlabel('Misclassified clips'); plt.ylabel(''); plt.tight_layout(); plt.savefig(OUT/f'{name}_top_confusions.png',dpi=180); plt.close()
 per.nsmallest(20,'recall').sort_values('recall').plot.barh(x='gloss',y='recall',legend=False,figsize=(8,6),color='#CC6677')
 plt.title(f'{name}: 20 lowest validation class recalls'); plt.xlabel('Recall'); plt.ylabel(''); plt.tight_layout(); plt.savefig(OUT/f'{name}_lowest_class_recall.png',dpi=180); plt.close()
 all_summary[name]={'n':len(d),'accuracy':float(accuracy_score(d.true_gloss,d.prediction)),'macro_f1':float(f1_score(d.true_gloss,d.prediction,labels=labels,average='macro',zero_division=0)),'classes':len(labels),'zero_recall_classes':per.loc[per.recall.eq(0),'gloss'].tolist(),'top_confusions':pairs.head(10).to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(all_summary,indent=2)); print(json.dumps(all_summary,indent=2))
