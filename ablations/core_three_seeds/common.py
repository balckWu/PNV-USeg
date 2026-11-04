import os,json,hashlib,subprocess,datetime,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FORBIDDEN_UUID='GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468'
ALLOWED=(0,1,2,4,5,6)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def job_named(name):return next(j for j in read(ROOT/'jobs.json') if j['id']==name)
def verify():
 for f,h in read(ROOT/'manifest.json')['files'].items():assert sha(ROOT/f)==h,('Source changed',f)
def environment(seed,job_id=None):
 j=job_named(job_id or os.environ['USTS_JOB_ID']);assert seed==j['seed'] and seed in [3407,2026]
 physical=j['physical_gpu'];assert physical in ALLOWED and physical!=3
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
 mapping={int(r.split(',')[0]):r.split(',')[1].strip() for r in rows}
 assert mapping[physical]==j['gpu_uuid'] and mapping[3]==FORBIDDEN_UUID
 assert j['gpu_uuid']!=FORBIDDEN_UUID
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=j['gpu_uuid'],USTS_TRAIN_SEED=str(seed),PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PAPERLAB_PROGRESS='off')
 return env
def wait_idle(j,seconds=120):
 deadline=time.time()+seconds
 while True:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
  row=next([x.strip() for x in r.split(',')] for r in gpu if int(r.split(',')[0])==j['physical_gpu'])
  assert row[1]==j['gpu_uuid'] and row[1]!=FORBIDDEN_UUID
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  busy=any(r.split(',')[0].strip()==j['gpu_uuid'] for r in apps.splitlines())
  if not busy and int(row[2])<=256 and int(row[3])<=5:return
  if time.time()>deadline:raise RuntimeError('Assigned physical GPU did not settle: '+str(row))
  time.sleep(1)
