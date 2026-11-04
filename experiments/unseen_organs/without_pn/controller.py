"""Run independently recorded paired target adaptations without disrupting EMCAD."""
import os,sys,time,json,subprocess,traceback,signal,datetime,shutil,hashlib,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
B=R.parent
E=B/'emcad_20261031'
U=B/'unseen_organs_20261027'
SRC=B/'f_parallel_20261029/jobs/F_remove_dual/source'
GPU=1
UUID='GPU-398c6dc4-dc8d-977a-3f85-3bd406a18008'
OWNER=3873447
ORGS={'heart':U,'liver':U/'lus7seg_20261027/liver','gallbladder':U/'lus7seg_20261027/gallbladder'}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def read(p):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return None
def alive(pid):
 p=Path('/proc')/str(pid)
 try:return p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except OSError:return False
def inventory():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 rows={int(x[0]):x[1:] for line in lines for x in [[s.strip() for s in line.split(',')]]}
 assert rows[GPU][0]==UUID and rows[3][0]!=UUID
 return rows[GPU]
def contexts():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return [int(x[1]) for line in s.splitlines() if len(x:=[v.strip() for v in line.split(',')])==2 and x[0]==UUID]
def mode():
 row=inventory();used=contexts();ctrl=read(E/'records/status_v2.json') or {}; job=(ctrl.get('active') or {}).get('EMCAD_full_seed42') or {}
 if job.get('stage')=='train' and job.get('pid')==OWNER and alive(OWNER) and used==[OWNER] and int(row[1])>=19000:
  # Do not start lengthy work near a possible early stop.
  hp=E/'runs/EMCAD_full_seed42/history.jsonl'
  try:last=json.loads(hp.read_text().splitlines()[-1]);bad=last['bad_epochs']
  except (OSError,IndexError,KeyError,ValueError):return None
  if bad<=12:return 'shared'
 complete=read(E/'records/v2_jobs/EMCAD_full_seed42/complete.json')
 if complete and complete.get('passed') and not used and int(row[2])<=256 and int(row[3])<=5:return 'exclusive'
 return None
def progress(label,**kw):write(R/'records/status.json',dict(time=now(),controller_pid=os.getpid(),stage=label,**kw))
def execute(organ,k,seed,preflight=False):
 tag=('preflight' if preflight else f'K{k}_seed{seed}')
 attempts=R/'attempts'/organ/tag;attempts.mkdir(parents=True,exist_ok=True)
 attempt=len(list(attempts.iterdir()))+1;run=attempts/f'attempt_{attempt:03d}';run.mkdir()
 shutil.copyfile(ORGS[organ]/'protocol.json',run/'protocol.json')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',DUAL_RUN_ROOT=str(run),DUAL_PARENT_ROOT=str(ORGS[organ]),DUAL_K=str(k),DUAL_SEED=str(seed),DUAL_PREFLIGHT='1' if preflight else '0',DUAL_GPU_INDEX=str(GPU),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8',PAPERLAB_PROGRESS='off',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 lease=mode()
 if lease is None:return 'wait',None
 with (run/'worker.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(R/'code'/f'{organ}.py'),'--method','F','--mode','adapt'],cwd=run,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  progress('preflight' if preflight else 'adapt',organ=organ,K=k,support_seed=seed,pid=p.pid,physical_gpu=GPU,uuid=UUID,sharing=lease,attempt=str(run))
  began=time.monotonic();interrupted=None
  while p.poll() is None:
   time.sleep(3)
   try:
    inventory()
    if lease=='shared' and (not alive(OWNER) or not (E/'records/jobs/EMCAD_full_seed42/process.json').exists() or (read(E/'records/jobs/EMCAD_full_seed42/process.json') or {}).get('stage')!='train'):
     interrupted='original EMCAD training ended; release GPU for its evaluation';p.terminate();break
    if preflight and time.monotonic()-began>600:
     interrupted='preflight timeout';p.terminate();break
   except Exception as exc:
    interrupted='GPU/owner monitoring failed: '+str(exc);p.terminate();break
  if interrupted:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   write(run/'interrupted.json',dict(time=now(),reason=interrupted,exit_code=p.returncode));return 'interrupted',str(run)
  code=p.wait()
 if code!=0:
  write(run/'controller_failure.json',dict(time=now(),exit_code=code,log=str(run/'worker.log')));return 'failed',str(run)
 directory=run/'results/F/adapt';p=directory/'preflight.json' if preflight else directory/'complete.json';d=read(p)
 if not d or not d.get('passed'):return 'failed',str(run)
 if not preflight:
  res=read(directory/f'K{k}_seed{seed}'/'result.json');parent=read(ORGS[organ]/f'results/F/adapt/K{k}_seed{seed}/result.json')
  assert res and parent and res['support_ids']==parent['support_ids'] and res['training']['successful_optimizer_updates']==100
  assert res['checkpoint_sha256']=='4894238b8340a69f196fffc70a4db54a83326bdb0c10c16d8405f0468c1b8a5c'
  cases=read(directory/f'K{k}_seed{seed}'/'per_patient.json');assert len(cases)==res['test_patients']
  for metric in ('dice','iou','hd95'):assert abs(statistics.mean(x[metric] for x in cases)-res[metric])<1e-9
  return 'complete',{'organ':organ,'K':k,'support_seed':seed,'run':str(run),'metrics':{m:res[m] for m in ('dice','iou','hd95')},'parent_metrics':{m:parent[m] for m in ('dice','iou','hd95')}}
 return 'complete',str(run)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 fd=os.open(R/'records/controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
 done=[]
 for organ in ('heart','liver','gallbladder'):
  for k,seed,pre in [(16,42,True)]+[(k,seed,False) for k in (4,16) for seed in (42,3407,2026)]:
   while True:
    lease=mode()
    if lease is None:progress('waiting_for_safe_gpu',organ=organ,K=k,support_seed=seed,completed=len(done));time.sleep(15);continue
    state,result=execute(organ,k,seed,pre)
    if state in ('wait','interrupted'):time.sleep(15);continue
    if state=='failed':raise RuntimeError(f'{organ} {k} {seed} failed: {result}')
    if not pre:done.append(result);write(R/'records/paired_partial.json',done)
    break
 write(R/'records/complete.json',dict(time=now(),passed=True,settings=len(done),paired=done))
 progress('complete',completed=len(done))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(R/'records/failed.json',dict(time=now(),error=str(exc),traceback=traceback.format_exc()));raise
