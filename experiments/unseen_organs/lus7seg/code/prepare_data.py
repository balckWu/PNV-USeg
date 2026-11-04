import hashlib,io,json,os,time,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'data/LUS-7Seg_Data.zip'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def main():
 plan=json.loads((ROOT/'shared_plan.json').read_text())
 h=hashlib.sha256()
 with ARCHIVE.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==plan['source_sha256']
 seen={};duplicate_pairs=[];selected_pixels={}
 with zipfile.ZipFile(ARCHIVE) as z:
  jpgs=sorted(n for n in z.namelist() if n.endswith('.jpg'))
  assert len(jpgs)==10507
  for i,name in enumerate(jpgs):
   digest=sha(z.read(name));patient=name.split('/')[0]
   previous=seen.get(digest)
   if previous and previous[0]!=patient:
    duplicate_pairs.append(dict(first=previous[0],second=patient,cross_split=plan['patient_split'][previous[0]]!=plan['patient_split'][patient]))
   else:seen[digest]=(patient,name)
  write(ROOT/'records/raw_duplicate_audit.json',dict(images=len(jpgs),duplicate_patient_pairs=duplicate_pairs))
  if any(r['cross_split'] for r in duplicate_pairs):raise RuntimeError('Duplicate JPEG across patient splits; regroup before any model test')
  for organ in ('liver','gallbladder'):
   root=ROOT/organ;data=root/'data'
   manifest=json.loads((data/'manifest_plan.json').read_text())
   required={r['id'] for r in manifest['rows'] if r['split']=='test'}
   for sets in manifest['supports'].values():required.update(sets['16'])
   for folder in ('images','masks'):(data/folder).mkdir(exist_ok=False)
   exported=0
   for row in manifest['rows']:
    row['exported']=row['id'] in required
    if not row['exported']:continue
    image=Image.open(io.BytesIO(z.read(row['source_stem']+'.jpg'))).convert('L')
    mask=Image.open(io.BytesIO(z.read(row['source_stem']+'.png'))).convert('RGB')
    assert image.size==mask.size==(row['width'],row['height'])
    rgb=np.asarray(mask);colour=np.asarray(plan['png_mapping'][organ],dtype=np.uint8)
    target=np.all(rgb==colour,axis=-1).astype(np.uint8)
    valid=((rgb[:,:,0]==0)&(rgb[:,:,1]==0)&(rgb[:,:,2]==0))|np.all(rgb==[255,255,255],axis=-1)|np.all(rgb==[0,255,0],axis=-1)
    if not valid.all() or not target.any():raise RuntimeError('Invalid palette or absent annotated target: '+row['source_stem'])
    image=image.resize((512,512),Image.Resampling.BILINEAR)
    mask=Image.fromarray(target).resize((512,512),Image.Resampling.NEAREST)
    assert np.asarray(mask).sum()>0
    pixel_sha=sha(image.tobytes());previous=selected_pixels.get(pixel_sha)
    if previous and previous['patient_id']!=row['patient_id'] and previous['split']!=row['split']:
     raise RuntimeError('Identical resized grayscale image across patient splits')
    selected_pixels[pixel_sha]=row
    image.save(data/row['image'],compress_level=1);mask.save(data/row['mask'],compress_level=1)
    row['image_sha256']=sha((data/row['image']).read_bytes());row['mask_sha256']=sha((data/row['mask']).read_bytes())
    row['grayscale_pixel_sha256']=pixel_sha;exported+=1
    if exported%100==0:print('EXPORT',organ,exported,'of',len(required),flush=True)
   assert exported==len(required)
   write(data/'manifest.json',manifest)
   write(root/'records/data_ready.json',dict(organ=organ,source_sha256=plan['source_sha256'],exported_images=exported,
      test_patients=manifest['test_patients'],test_frames=manifest['test_frames'],patient_counts=manifest['patient_counts'],
      lazy_export='only predetermined test images and union of labelled supports; full original archive preserved',
      raw_duplicate_cross_split=False,selected_grayscale_pixel_duplicate_cross_split=False,
      manifest_sha256=sha((data/'manifest.json').read_bytes())))
   print('DATA_READY',organ,json.dumps(json.loads((root/'records/data_ready.json').read_text())),flush=True)
 write(ROOT/'records/all_data_ready.json',dict(passed=True,source_sha256=plan['source_sha256'],time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(ROOT/'records/data_prepare_failed.json',dict(type=type(error).__name__,message=str(error)));raise
