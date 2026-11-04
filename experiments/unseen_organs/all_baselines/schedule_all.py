"""CPU-only serial scheduler; exclusively physical GPU6 by live UUID checks."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=('unetpp','swinunet_official','clipseg_official','lvit_native224','sammed2d_promptfree')
DIRECT=('F','paper_US_adapter','iris2d','clipseg_official','lvit_native224','sammed2d_promptfree')
ORGANS=('heart','liver','gallbladder')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert json.loads((ROOT/'records/preflights_finished.json').read_text())['passed']
 assert not (ROOT/'records/schedule.lock').exists()
 assert shutil.disk_usage(ROOT).free>60*1024**3,'Insufficient space for retained adaptation states'
 spec=importlib.util.spec_from_file_location('schedule_extra_worker',ROOT/'heart/code/experiment_v2.py')
 e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
 # Five heart adaptation jobs first; then all K0; then liver/GB adaptation.
 jobs=[dict(organ='heart',method=m,mode='adapt') for m in METHODS]
 jobs += [dict(organ=o,method=m,mode='zero_target') for o in ORGANS for m in DIRECT]
 jobs += [dict(organ=o,method=m,mode='adapt') for o in ORGANS[1:] for m in METHODS]
 inputs={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')}
 inputs.update({str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('protocol.json')})
 for name in ('direct_protocol.json','prompts.json','records/extra_text.pt','records/cpu_assets_verified.json'):inputs[name]=sha(ROOT/name)
 for o in ORGANS:
  data=Path(json.loads((ROOT/o/'protocol.json').read_text())['existing_data_directory'])
  manifest=json.loads((data/'manifest.json').read_text())
  selected={r['id'] for r in manifest['rows'] if r['split']=='test'}
  for s in manifest['supports'].values():selected.update(s['16'])
  for r in manifest['rows']:
   if r['id'] in selected:
    for typ in ('image','mask'):
     h=hashlib.sha256((data/r[typ]).read_bytes()).hexdigest()
     if typ+'_sha256' in r:assert h==r[typ+'_sha256']
  inputs[o+'/existing_manifest']=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
 env=e.gpu_env()
 fd=os.open(ROOT/'records/schedule.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
 write(ROOT/'records/registration.json',dict(time=e.now(),jobs=jobs,inputs=inputs,physical_gpu=6,gpu_uuid=env['CUDA_VISIBLE_DEVICES'],forbidden_gpu=3,new_adapt_settings=90,new_zero_settings=18,final_ten_model_settings=210))
 done=[];failed=[]
 for job in jobs:
  o,m,mode=job['organ'],job['method'],job['mode']
  for name,h in inputs.items():
   if not name.endswith('/existing_manifest'):assert sha(ROOT/name)==h,'Executable/protocol changed: '+name
  destination=ROOT/o/'results'/m/mode;assert not destination.exists(),'Refusing to overwrite '+str(destination)
  command=[sys.executable,'-u',str(ROOT/o/'code/experiment_v2.py'),'--method',m,'--mode','adapt'] if mode=='adapt' else [sys.executable,'-u',str(ROOT/'direct_prediction.py'),'--organ',o,'--method',m]
  env=e.gpu_env();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  logpath=ROOT/o/'logs'/(m+'_'+mode+'.log')
  with logpath.open('x') as log:
   p=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=dict(job,pid=p.pid,log=str(logpath)),completed=done,failed=failed))
   code=p.wait()
  ok=code==0 and (destination/'complete.json').exists()
  (done if ok else failed).append(dict(job,exit_code=code))
  print('JOB_FINISHED',o,m,mode,code,flush=True)
  subprocess.run([sys.executable,str(ROOT/'analyze_all.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=None,completed=done,failed=failed))
 write(ROOT/'records/all_jobs_finished.json',dict(time=e.now(),completed=done,failed=failed,passed=not failed))
 bundle=ROOT/'results_and_records.tar.gz'
 with tarfile.open(bundle,'w:gz') as tf:
  for p in ROOT.rglob('*'):
   if p.is_file() and p.suffix in ('.json','.md','.log','.py') and '__pycache__' not in p.parts:
    tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 print('SCHEDULE_FINISHED',len(done),len(failed),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(ROOT/'records/scheduler_failed.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise
