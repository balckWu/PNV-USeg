"""Predeclared patient-disjoint LUS organ transfer; physical GPU6 only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['USTS_ADAPT_JOB_ROOT'])
DATA=Path(os.environ['USTS_ADAPT_DATA'])
SOURCE=Path('/tank/wk/TS/TSV3')
CAMPAIGN=Path('/tank/wk/US-TextSeg/experiments/seed42_text_mechanisms_20261025')
PRIMARY=('F','iris2d','unet','transunet_pretrained','paper_US_adapter')
EXTRA=('unetpp','swinunet_official')
SUPPORT_MODELS=('F','iris2d','paper_US_adapter')

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def gpu_env():
 index=int(os.environ['USTS_GPU_INDEX']);visible=os.environ['CUDA_VISIBLE_DEVICES']
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
 mapping={int(x[0]):x[1] for row in rows for x in [[v.strip() for v in row.split(',')]]}
 assert index in (0,1,2,4,5,6) and mapping[3]=='GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468'
 assert mapping[index]==visible and visible!=mapping[3]
 return dict(os.environ)

def worker(method,mode):
 expected=gpu_env()['CUDA_VISIBLE_DEVICES']
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise RuntimeError('Worker GPU UUID mismatch')
 sys.path.insert(0,str(CAMPAIGN if method=='F' else SOURCE))
 import numpy as np
 import torch
 import torch.nn.functional as functional
 from PIL import Image
 from tasksegment.data.datasets import get_image_transform,set_seed
 from tasksegment.training.augmentations import augment_episode_medical
 from tasksegment.training.metrics import binary_dice,binary_iou,binary_hd95
 torch.set_num_threads(2);set_seed(42,deterministic=True)
 if torch.cuda.device_count()!=1:raise RuntimeError('Exactly physical GPU6 must be visible')
 device=torch.device('cuda:0')
 directory=ROOT/'results'/method/mode
 directory.mkdir(parents=True,exist_ok=False)
 text_path=Path(os.environ['USTS_TARGET_TEXT'])
 text=torch.load(text_path,map_location='cpu',weights_only=True)
 if method=='paper_US_adapter' and (ROOT/'records/paper_target_text.pt').exists():
  text_path=ROOT/'records/paper_target_text.pt'
  extra=torch.load(text_path,map_location='cpu',weights_only=True)
  text['features']['paper_US']=extra['features']['paper_US']
  text['metadata']['paper_US']=extra['metadata']
 if method=='paper_US_adapter' and 'paper_US' not in text['features']:
  write(directory/'skipped.json',dict(reason='Original frozen text encoder unavailable',metadata=text['metadata']))
  return
 checkpoint=Path(os.environ['USTS_SOURCE_CKPT'])
 checkpoint_sha=sha(checkpoint)
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if method=='F':
  from campaign.model import CampaignModel
  source_bank=torch.load(CAMPAIGN/'campaign/text_bank.pt',map_location='cpu',weights_only=True)
  model=CampaignModel(source_bank['features'],'F');model.load_state_dict(ck['model'],strict=True)
  with torch.no_grad():model.sources.medical_features[0].copy_(functional.normalize(text['features']['F'],dim=-1))
 elif method=='iris2d':
  from iris2d.model import Iris2D
  model=Iris2D(**ck['model_config']);model.load_state_dict(ck['model'],strict=True)
 elif method=='paper_US_adapter':
  from tasksegment.models import TaskSegmentModel
  model=TaskSegmentModel(**ck['model_config']);model.load_state_dict(ck['model_state_dict'],strict=True)
  model.task_encoder.use_domain_embedding=False
 else:
  spec=importlib.util.spec_from_file_location('existing_baseline_predict',SOURCE/'scripts/predict_baseline.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  key='transunet_official' if method=='transunet_pretrained' else method
  cls=module.import_model_class(key);cfg=module.sanitize_model_config(cls,ck['model_config'])
  model=cls(**cfg);model.load_state_dict(ck['model_state_dict'],strict=True)
 del ck
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.to(device)
 manifest=json.loads((DATA/'manifest.json').read_text())
 rows={r['id']:r for r in manifest['rows']}
 test_ids=[r['id'] for r in manifest['rows'] if r['split']=='test']
 supports=manifest['supports']
 supports['42']['8']=supports['42']['16'][:8]
 supports={'42':supports['42']}
 selected=set(test_ids)
 for sets in supports.values():selected.update(sets['16'])
 selected=sorted(selected)
 normalize=get_image_transform()
 images=[];masks=[]
 for identity in selected:
  r=rows[identity]
  a=np.asarray(Image.open(DATA/r['image']).convert('L'),dtype=np.float32)/255.
  b=np.asarray(Image.open(DATA/r['mask']).convert('L'),dtype=np.uint8)
  images.append(normalize(torch.from_numpy(a.copy())[None]))
  masks.append(torch.from_numpy(b.copy())[None].float())
 image_bank=torch.stack(images).to(device);mask_bank=torch.stack(masks).to(device)
 lookup={identity:i for i,identity in enumerate(selected)}
 targets={identity:masks[lookup[identity]][0].numpy().astype(np.uint8) for identity in test_ids}
 target_text=text['features'].get('paper_US',torch.empty(0)).to(device)
 def batch(ids):
  ii=torch.tensor([lookup[i] for i in ids],device=device)
  return image_bank.index_select(0,ii),mask_bank.index_select(0,ii)
 def parameters_sha():
  h=hashlib.sha256()
  for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 initial_parameters_sha=parameters_sha()
 def prototype(x,y):
  if method=='F':return model.support_tokens(x,y)
  if method=='iris2d':return model.encode_support(x,y)
  if method=='paper_US_adapter':
   v=model.task_encoder.forward_visual(model.encoder.forward_features(x)['bottleneck'],y).mean(0,keepdim=True)
   return model.compose_task_tokens(v,Ftext=target_text[None],domain_ids=None)
  return None
 def predict(x,proto):
  if method=='F':output=model.segment(x,proto,torch.zeros(len(x),device=device,dtype=torch.long))
  elif method=='iris2d':output=model.segment(x,proto)
  elif method=='paper_US_adapter':output=model.segment_with_task(xq=x,task_tokens=proto)
  else:output=model(x)
  if isinstance(output,dict):
   output=output.get('pred_masks',output.get('logits',next(iter(output.values()))))
  if isinstance(output,(list,tuple)):output=output[-1]
  if output.shape[1]!=2 or output.shape[-2:]!=(512,512):raise RuntimeError('Unexpected binary 512 logits')
  return output
 def loss_fn(logits,target):
  ce=functional.cross_entropy(logits.float(),target.long())
  p=logits.float().softmax(1)[:,1];y=target.float()
  dice=(2*(p*y).sum((-2,-1))+1)/(p.sum((-2,-1))+y.sum((-2,-1))+1)
  return ce+(1-dice).mean()
 def restore():
  model.load_state_dict(initial,strict=True)
  model.zero_grad(set_to_none=True)
 # Pre-test shape/gradient check uses only labelled training-support images.
 check=supports['42']['4'];sx,sy=batch(check[1:]);qx,qy=batch(check[:1])
 model.train() if method in PRIMARY else model.eval()
 optimizer=None
 if method in PRIMARY:
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
  scaler=torch.amp.GradScaler('cuda',init_scale=128.)
  with torch.autocast('cuda',dtype=torch.float16):
   logits=predict(qx,prototype(sx,sy));loss=loss_fn(logits,qy[:,0])
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
  scaler.scale(loss).backward();scaler.unscale_(optimizer)
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
  if not torch.isfinite(norm):raise RuntimeError('Nonfinite preflight gradient')
  scaler.step(optimizer);scaler.update()
 else:
  with torch.inference_mode():logits=predict(qx,None)
 restore();model.eval()
 if parameters_sha()!=initial_parameters_sha:raise RuntimeError('Preflight did not restore source weights exactly')
 del optimizer
 write(directory/'preflight.json',dict(passed=True,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
       strict_loading=True,source_weights_restored=True,training_images_only=True,physical_gpu=int(os.environ['USTS_GPU_INDEX']),time=now()))
 if os.environ['USTS_ADAPT_PREFLIGHT']=='1':return
 def metrics(pred,target):return binary_dice(pred,target),binary_iou(pred,target),binary_hd95(pred,target)
 @torch.inference_mode()
 def evaluate(support_ids,run):
  model.eval();proto=None
  if method in SUPPORT_MODELS:
   tokens=[]
   for first in range(0,len(support_ids),4):
    x,y=batch(support_ids[first:first+4]);tokens.append(prototype(x,y))
   proto=torch.stack(tokens).mean(0)
  cases=[];futures=[]
  with ThreadPoolExecutor(max_workers=2) as pool:
   for first in range(0,len(test_ids),2):
    ids=test_ids[first:first+2];x,y=batch(ids)
    logits=predict(x,proto)
    prediction=(logits.softmax(1)[:,1]>=.5).cpu().numpy().astype(np.uint8)
    for j,identity in enumerate(ids):futures.append((identity,pool.submit(metrics,prediction[j],targets[identity])))
   for identity,future in futures:
    numbers=future.result()
    if not all(np.isfinite(n) for n in numbers):raise RuntimeError('Nonfinite test metric')
    r=rows[identity]
    cases.append(dict(id=identity,patient_id=r['patient_id'],view=r['view'],phase=r['phase'],
                      dice=numbers[0],iou=numbers[1],hd95=numbers[2]))
  patient_ids=sorted({r['patient_id'] for r in cases})
  patients=[dict(patient_id=p,**{k:statistics.mean(r[k] for r in cases if r['patient_id']==p) for k in ('dice','iou','hd95')}) for p in patient_ids]
  assert len(patients)==manifest['test_patients'] and len(cases)==manifest['test_frames']
  write(run/'per_case.json',cases);write(run/'per_patient.json',patients)
  return {k:statistics.mean(r[k] for r in patients) for k in ('dice','iou','hd95')}
 def augmented(ids,seed):
  x,y=batch(ids)
  py=random.getstate();np_state=np.random.get_state()
  with torch.random.fork_rng(devices=[0]):
   random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed)
   x,y=augment_episode_medical(x,y,'generic')
  random.setstate(py);np.random.set_state(np_state)
  return x,y
 settings=[(int(k),42) for k in os.environ['USTS_ADAPT_KS'].split(',')]
 results=[]
 for k,seed in settings:
  run=directory/('K'+str(k)+'_seed'+str(seed));run.mkdir()
  support_ids=supports[str(seed)][str(k)] if k else []
  assert all(rows[i]['split']=='train' for i in support_ids)
  assert not {rows[i]['patient_id'] for i in support_ids}&{rows[i]['patient_id'] for i in test_ids}
  restore();set_seed(seed,deterministic=True)
  start=time.monotonic();training={}
  if mode=='adapt':
   optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
   scaler=torch.amp.GradScaler('cuda')
   completed=0;attempts=0;losses=[];model.train()
   while completed<100:
    if attempts>=200:raise RuntimeError('Too many discarded AMP overflow attempts')
    optimizer.zero_grad(set_to_none=True);attempt_loss=0.
    for micro in range(2):
     step=attempts*2+micro
     rng=np.random.default_rng(seed+k*1009+step*1000003)
     query=str(rng.choice(support_ids));pool=[i for i in support_ids if i!=query]
     chosen=rng.choice(pool,size=min(4,len(pool)),replace=False).tolist()
     qx,qy=augmented([query],seed+step*1009)
     with torch.autocast('cuda',dtype=torch.float16):
      proto=None
      if method in SUPPORT_MODELS:
       sx,sy=augmented(chosen,seed+step*1009+500000003);proto=prototype(sx,sy)
      logits=predict(qx,proto);loss=loss_fn(logits,qy[:,0])
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite adaptation loss')
     scaler.scale(loss/2).backward();attempt_loss+=float(loss.detach())/2
    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();attempts+=1
    if scaler.get_scale()>=old_scale:completed+=1;losses.append(attempt_loss)
    if completed and completed%20==0:
     write(directory/'status.json',dict(time=now(),method=method,mode=mode,K=k,seed=seed,optimizer_updates=completed,attempts=attempts))
   training=dict(successful_optimizer_updates=completed,attempts=attempts,discarded_overflow_attempts=attempts-completed,
                 last_loss=losses[-1],labelled_target_images=k,query_images_per_successful_update=2)
   torch.save(dict(model=model.state_dict(),method=method,K=k,seed=seed,source_checkpoint_sha256=checkpoint_sha,
                   protocol_sha256=sha(ROOT/'protocol.json'),training=training),run/'adapted_state.pt')
   del optimizer
  values=evaluate(support_ids,run)
  if mode=='noft' and parameters_sha()!=initial_parameters_sha:raise RuntimeError('Frozen evaluation changed weights')
  result=dict(method=method,mode=mode,K=k,support_draw_seed=seed,source_model_seed=int(os.environ['USTS_SOURCE_SEED']),time=now(),
              checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,protocol_sha256=sha(ROOT/'protocol.json'),
              text_sha256=sha(text_path),test_patients=manifest['test_patients'],test_frames=manifest['test_frames'],support_ids=support_ids,
              training=training,seconds=time.monotonic()-start,**values)
  write(run/'result.json',result);results.append(result)
  write(directory/'status.json',dict(time=now(),method=method,mode=mode,K=k,seed=seed,stage='evaluated',completed_settings=len(results)))
  print('RESULT',json.dumps(result),flush=True)
 if sha(checkpoint)!=checkpoint_sha:raise RuntimeError('Source checkpoint changed during experiment')
 write(directory/'complete.json',dict(time=now(),passed=True,settings=len(results),results=results))

if __name__=='__main__':
 try:worker(os.environ['USTS_ADAPT_METHOD'],'adapt')
 except BaseException as e:
  write(ROOT/'failure.json',dict(time=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  raise
