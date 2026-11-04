"""Zero target labels: frozen prediction, optionally a fixed source-only prior."""
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
SOURCE=Path('/tank/wk/TS/TSV3')
CAMPAIGN=PARENT.parent
SOURCE_MODELS=('F','iris2d','paper_US_adapter')
NEW_MODELS=('clipseg_official','lvit_native224','sammed2d_promptfree')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main(organ,method):
 spec=importlib.util.spec_from_file_location('direct_gpu_guard',ROOT/'heart/code/gpu_guard.py')
 guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 env=guard.gpu_environment(6)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==env['CUDA_VISIBLE_DEVICES']=='GPU-dc2317c7-139f-d155-c749-9216d85b13d4'
 sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(CAMPAIGN))
 import numpy as np
 import torch
 import torch.nn.functional as F
 from PIL import Image
 from tasksegment.data.datasets import get_image_transform,set_seed
 from tasksegment.training.metrics import binary_dice,binary_iou,binary_hd95
 torch.set_num_threads(2);set_seed(42,deterministic=True)
 assert torch.cuda.device_count()==1
 output=ROOT/organ/'results'/method/'zero_target'
 output.mkdir(parents=True,exist_ok=False)
 start=time.monotonic()
 device=torch.device('cuda:0');prior=None;target_text=None
 if method in NEW_MODELS:
  from extra_models import load_model
  model,meta=load_model(method,organ);checkpoint=Path(meta['checkpoint'])
 else:
  checkpoint=CAMPAIGN/'runs/F/seed_42/best.pt' if method=='F' else SOURCE/'paper_artifacts/runs'/('tasksegment_full2' if method=='paper_US_adapter' else method)/'seed_42/checkpoint.pth'
  ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if method=='F':
   from campaign.model import CampaignModel
   bank=torch.load(CAMPAIGN/'campaign/text_bank.pt',map_location='cpu',weights_only=True)
   model=CampaignModel(bank['features'],'F');model.load_state_dict(ck['model'],strict=True)
  elif method=='iris2d':
   from iris2d.model import Iris2D
   model=Iris2D(**ck['model_config']);model.load_state_dict(ck['model'],strict=True)
  else:
   from tasksegment.models import TaskSegmentModel
   model=TaskSegmentModel(**ck['model_config']);model.load_state_dict(ck['model_state_dict'],strict=True)
   model.task_encoder.use_domain_embedding=False
  source_bank=ck['prototypes'] if method=='iris2d' else ck['prototype_bank']
  domains=sorted(source_bank)
  assert domains==['BUS-BRA','BUSI_WHU','OTU','TN3K','prostate','thyroid']
  prior=torch.stack([source_bank[d].float() for d in domains]).mean(0)
  assert torch.isfinite(prior).all()
  meta=dict(method=method,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_source_loading=True,
            source_prior='fixed equal mean of six source-domain saved visual prototypes',source_prior_domains=domains,
            source_prior_sha256=hashlib.sha256(prior.contiguous().numpy().tobytes()).hexdigest(),
            target_support_images=0,target_support_masks=0,input_policy='source_zscore_clamp5_divide5')
  original=PARENT if organ=='heart' else PARENT/'lus7seg_20261027'/organ
  if method=='F':
   p=original/'records/target_text.pt';text=torch.load(p,map_location='cpu',weights_only=True)
   with torch.no_grad():model.sources.medical_features[0].copy_(F.normalize(text['features']['F'],dim=-1))
   meta['target_text_sha256']=sha(p)
  elif method=='paper_US_adapter':
   p=original/'records/paper_target_text.pt';text=torch.load(p,map_location='cpu',weights_only=True)
   target_text=text['features']['paper_US'].to(device);meta['target_text_sha256']=sha(p)
  del ck
 checkpoint_sha=sha(checkpoint)
 def parameter_sha():
  h=hashlib.sha256()
  for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 initial_sha=parameter_sha();model.to(device).eval()
 if prior is not None:prior=prior.to(device)
 if method=='paper_US_adapter':
  with torch.no_grad():prior=model.compose_task_tokens(prior,Ftext=target_text[None],domain_ids=None)
 protocol=json.loads((ROOT/organ/'protocol.json').read_text());data=Path(protocol['existing_data_directory'])
 manifest=json.loads((data/'manifest.json').read_text())
 rows=[r for r in manifest['rows'] if r['split']=='test']
 # Deliberately never access manifest['supports'] or any target train/validation image.
 normalize=(lambda x:x) if method in NEW_MODELS else get_image_transform()
 image_bank=torch.stack([normalize(torch.from_numpy(np.asarray(Image.open(data/r['image']).convert('L'),dtype=np.float32).copy()/255.)[None]) for r in rows]).to(device)
 def prediction(x):
  if method=='F':y=model.segment(x,prior,torch.zeros(len(x),device=device,dtype=torch.long))
  elif method=='iris2d':y=model.segment(x,prior)
  elif method=='paper_US_adapter':y=model.segment_with_task(xq=x,task_tokens=prior)
  else:y=model(x)
  if isinstance(y,dict):y=y.get('pred_masks',y.get('logits',next(iter(y.values()))))
  if isinstance(y,(list,tuple)):y=y[-1]
  assert y.shape==(len(x),2,512,512)
  return (y.float().softmax(1)[:,1]>=.5).cpu().numpy().astype(np.uint8)
 def metrics(pred,target):return binary_dice(pred,target),binary_iou(pred,target),binary_hd95(pred,target)
 cases=[];futures=[]
 with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool:
  for first in range(0,len(rows),2):
   predicted=prediction(image_bank[first:first+2])
   # Target masks are first read here, after frozen predictions, only for scoring.
   for j,r in enumerate(rows[first:first+2]):
    target=np.asarray(Image.open(data/r['mask']).convert('L'),dtype=np.uint8)
    futures.append((r,pool.submit(metrics,predicted[j],target)))
  for r,future in futures:
   values=future.result();assert all(np.isfinite(v) for v in values)
   cases.append(dict(id=r['id'],patient_id=r['patient_id'],**dict(zip(('dice','iou','hd95'),values))))
 patients=[dict(patient_id=p,**{k:statistics.mean(r[k] for r in cases if r['patient_id']==p) for k in ('dice','iou','hd95')}) for p in sorted({r['patient_id'] for r in cases})]
 assert len(cases)==manifest.get('test_frames',200) and len(patients)==manifest.get('test_patients',50)
 assert parameter_sha()==initial_sha and sha(checkpoint)==checkpoint_sha
 write(output/'per_case.json',cases);write(output/'per_patient.json',patients)
 result=dict(organ=organ,method=method,mode='zero_target',K=0,source_model_seed=42,
             target_labelled_samples=0,target_train_images_read=0,target_masks_used_in_model=False,
             no_weight_update=True,source_parameters_unchanged=True,source_parameter_sha256=initial_sha,
             checkpoint_sha256=checkpoint_sha,model_metadata=meta,manifest_sha256=sha(data/'manifest.json'),
             protocol_sha256=sha(ROOT/organ/'protocol.json'),direct_protocol_sha256=sha(ROOT/'direct_protocol.json'),
             worker_sha256=sha(__file__),test_frames=len(cases),test_patients=len(patients),seconds=time.monotonic()-start,
             **{k:statistics.mean(r[k] for r in patients) for k in ('dice','iou','hd95')})
 write(output/'result.json',result);write(output/'complete.json',dict(passed=True,settings=1))
 print('ZERO_TARGET_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--organ',choices=('heart','liver','gallbladder'),required=True);p.add_argument('--method',choices=SOURCE_MODELS+NEW_MODELS,required=True);a=p.parse_args()
 try:main(a.organ,a.method)
 except BaseException as error:
  write(ROOT/'records'/('zero_'+a.organ+'_'+a.method+'_failed.json'),dict(type=type(error).__name__,message=str(error),traceback=__import__('traceback').format_exc()));raise
