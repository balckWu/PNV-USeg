import datetime,hashlib,importlib.util,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
V2=Path('/tank/wk/US-TextSeg/experiments/no_text_branch_v2_20261026')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 for name,h in read(ROOT/'manifest.json')['files'].items():assert sha(ROOT/name)==h,('Queue code changed',name)
def environment(seed):
 spec=importlib.util.spec_from_file_location('queue_gpu_guard',ROOT/'gpu_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 for attempt in range(31):
  try:env=guard.gpu_environment(4);break
  except RuntimeError as e:
   if str(e)!='Physical GPU 4 is occupied; refusing launch' or attempt==30:raise
   time.sleep(1)
 assert env['CUDA_VISIBLE_DEVICES']=='GPU-39c4f7c8-df6a-a19d-c83e-4154a7cf5b5a'
 env.update(USTS_TRAIN_SEED=str(seed),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',PAPERLAB_PROGRESS='off')
 return env
def job_named(name):return next(j for j in read(ROOT/'jobs.json') if j['id']==name)
