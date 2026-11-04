"""Create a deterministic 20-clip visual check of stored pose/hand landmarks."""
from pathlib import Path
import cv2,numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/WLASL100'
OUT=DATA/'overlay_review'; OUT.mkdir(parents=True,exist_ok=True)
idx=pd.read_csv(DATA/'landmarks_kaggle'/'index.csv',dtype={'video_id':str})
meta=pd.read_csv(DATA/'all_metadata.csv',dtype={'video_id':str}); meta.video_id=meta.video_id.str.zfill(5)
idx.video_id=idx.video_id.str.zfill(5); idx=idx.merge(meta[['video_id','gloss','signer_id']],on='video_id',validate='one_to_one')
idx=idx[idx.status.eq('ok')].sort_values('video_id').reset_index(drop=True)
ids=np.random.default_rng(2026).choice(len(idx),size=min(20,len(idx)),replace=False)
pose_edges=[(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32),(27,29),(28,30)]
hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
tiles=[]; rows=[]
for n,i in enumerate(ids,1):
 r=idx.iloc[i]; vid=r.video_id; arr=np.load(DATA/'landmarks_kaggle'/f'{vid}.npy')
 video=Path(str(r.path)); cap=cv2.VideoCapture(str(video))
 if not cap.isOpened(): rows.append({'video_id':vid,'gloss':r.gloss,'overlay':'video_open_failed'}); continue
 frame_index=max(0,int(r.start)+len(arr)//2); cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index); ok,frame=cap.read(); cap.release()
 if not ok: rows.append({'video_id':vid,'gloss':r.gloss,'overlay':'frame_read_failed'}); continue
 h,w=frame.shape[:2]; lm=arr[len(arr)//2].reshape(75,3)
 pts=[]
 for x,y,z in lm:
  pts.append((int(np.clip(x*h,0,w-1)),int(np.clip(y*h,0,h-1))) if np.isfinite(x) and np.isfinite(y) else None)
 for a,b in pose_edges:
  if pts[a] and pts[b]: cv2.line(frame,pts[a],pts[b],(40,200,80),2,cv2.LINE_AA)
 for base,color in ((33,(255,120,20)),(54,(30,180,255))):
  for a,b in hand_edges:
   if pts[base+a] and pts[base+b]: cv2.line(frame,pts[base+a],pts[base+b],color,2,cv2.LINE_AA)
 for j,p in enumerate(pts):
  if p: cv2.circle(frame,p,2 if j<33 else 3,(0,40,255),-1,cv2.LINE_AA)
 frame=cv2.resize(frame,(320,180)); cv2.rectangle(frame,(0,0),(320,24),(0,0,0),-1)
 cv2.putText(frame,f'{vid} {r.gloss}',(6,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
 tiles.append(frame); rows.append({'video_id':vid,'gloss':r.gloss,'signer_id':r.signer_id,'overlay':'created'})
canvas=np.full((4*180,5*320,3),255,np.uint8)
for j,tile in enumerate(tiles): canvas[(j//5)*180:(j//5+1)*180,(j%5)*320:(j%5+1)*320]=tile
cv2.imwrite(str(OUT/'landmark_overlay_sample.png'),canvas)
pd.DataFrame(rows).to_csv(OUT/'review_sample.csv',index=False)
print(f'Created visual overlay contact sheet with {len(tiles)} clips: {OUT / "landmark_overlay_sample.png"}')
