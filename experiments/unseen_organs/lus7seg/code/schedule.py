"""CPU-only scheduler, with one reserved UUID for all isolated workers."""
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=('F','iris2d','unet','transunet_pretrained','paper_US_adapter')
EXTRA=('unetpp','swinunet_official')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert (ROOT/'records/all_data_ready.json').exists() and (ROOT/'records/all_text_ready.json').exists()
 modules={};inputs={}
 for organ in ('liver','gallbladder'):
  root=ROOT/organ
  spec=importlib.util.spec_from_file_location('lus_'+organ,root/'code/experiment.py')
  e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);modules[organ]=e
  assert not list((root/'results').iterdir()),'Refusing duplicate existing results'
  manifest=json.loads((root/'data/manifest.json').read_text())
  for r in manifest['rows']:
   if r['exported']:
    assert sha(root/'data'/r['image'])==r['image_sha256'] and sha(root/'data'/r['mask'])==r['mask_sha256']
  inputs[organ]={n:sha(root/n) for n in ('protocol.json','target_prompts.json','data/manifest.json','code/experiment.py','code/gpu_guard.py','records/target_text.pt','records/paper_target_text.pt')}
 env=modules['liver'].gpu_env()
 lock=ROOT/'records/schedule.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 jobs=[(o,m,'noft') for o in ('liver','gallbladder') for m in PRIMARY+EXTRA]+[(o,m,'adapt') for o in ('liver','gallbladder') for m in PRIMARY]
 registration=dict(time=modules['liver'].now(),inputs=inputs,shared_plan_sha256=sha(ROOT/'shared_plan.json'),
    scheduler_sha256=sha(Path(__file__)),physical_gpu=6,physical_uuid=env['CUDA_VISIBLE_DEVICES'],forbidden_gpu=3,
    source_checkpoint_seed=42,jobs=jobs,expected_settings=104)
 write(ROOT/'records/experiment_registration.json',registration)
 for organ,e in modules.items():write(ROOT/organ/'records/experiment_registration.json',dict(**registration,organ=organ))
 done=[];failed=[]
 for organ,method,mode in jobs:
  e=modules[organ];root=ROOT/organ;env=e.gpu_env()
  assert sha(root/'code/experiment.py')==inputs[organ]['code/experiment.py']
  assert sha(root/'protocol.json')==inputs[organ]['protocol.json']
  command=[sys.executable,'-u',str(root/'code/experiment.py'),'--method',method,'--mode',mode]
  with (root/'logs'/(method+'_'+mode+'.log')).open('x') as log:
   child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=dict(organ=organ,method=method,mode=mode,pid=child.pid),completed=done,failed=failed))
   code=child.wait()
  result=root/'results'/method/mode
  if code or not (result/'complete.json').exists():failed.append(dict(organ=organ,method=method,mode=mode,exit_code=code))
  else:done.append(dict(organ=organ,method=method,mode=mode,skipped=False))
  print('JOB_FINISHED',organ,method,mode,code,flush=True)
 for organ,e in modules.items():
  write(ROOT/organ/'records/all_jobs_finished.json',dict(time=e.now(),completed=[r for r in done if r['organ']==organ],failed=[r for r in failed if r['organ']==organ]))
 write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=None,completed=done,failed=failed))
 write(ROOT/'records/all_jobs_finished.json',dict(time=e.now(),completed=done,failed=failed))
if __name__=='__main__':main()
