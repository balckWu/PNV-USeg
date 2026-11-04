from pathlib import Path
import os,sys,time,json,signal,subprocess,hashlib,shutil,statistics,traceback,fcntl,threading
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];B=R.parent;U=B/'unseen_organs_20261027';F=B/'official_fullsource_adapted_20261101';S=Path('/tank/wk/TS/TSV3/paper_artifacts/runs')
ORG={'heart':U,'liver':U/'lus7seg_20261027/liver','gallbladder':U/'lus7seg_20261027/gallbladder'}
PRIMARY=('F','unet','iris2d','transunet_pretrained');EXTRA=('unetpp','swinunet_official','clipseg_official','lvit_native224','sammed2d_promptfree')
UUUID={0:'GPU-d18bd3a8-741c-30dc-2039-4f358724578f',1:'GPU-398c6dc4-dc8d-977a-3f85-3bd406a18008',2:'GPU-a8f0398a-f1da-ab77-a801-adc63f56ec73',4:'GPU-39c4f7c8-df6a-a19d-c83e-4154a7cf5b5a',5:'GPU-c44b0e55-0b1e-06ee-966f-51879c4bac81',6:'GPU-dc2317c7-139f-d155-c749-9216d85b13d4'}
DENY='GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468';active={};failures=[];done=[];pause_plan=[];watchdog=None

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.'+str(threading.get_ident())+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def read(p):return json.loads(p.read_text())
def proc(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return {'state':s[0],'startticks':int(s[19])}
 except (OSError,ValueError):return None
def same(j):
 p=proc(j['pid']);return p and p['startticks']==j['startticks'] and p['state']!='Z'
def inventory():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 d={int(x[0]):{'uuid':x[1],'free':int(x[2]),'util':int(x[3])} for l in s.splitlines() for x in [[v.strip() for v in l.split(',')]]}
 assert d[3]['uuid']==DENY and all(d[k]['uuid']==v for k,v in UUUID.items())
 c=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 ctx={}
 for line in c.splitlines():
  try:u,p=[x.strip() for x in line.split(',')];ctx.setdefault(u,set()).add(int(p))
  except (ValueError,TypeError):pass
 assert not ctx.get(DENY)
 return d,ctx
def mem_gib():return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/1048576
def snapshot(stage,**kw):
 workers={name:{'gpu':v['gpu'],'pid':v.get('pid'),'startticks':v.get('startticks'),'phase':v.get('phase')} for name,v in list(active.items())}
 write(R/'records/status.json',{'stage':stage,'time':time.time(),'workers':workers,'completed':done,'failed':failures,'total_groups':81,'new_settings':189,'reused_settings':54,'paused_originals':pause_plan,'gpu3_forbidden':True,**kw})
def checkpoint(method,seed):
 if method=='F':return B/'runs/F/seed_42/best.pt' if seed==42 else B/'f_extra_seeds_20261027/runs/F'/('seed_'+str(seed))/'best.pt'
 return S/method/('seed_'+str(seed))/'checkpoint.pth'
def dataset(organ):return U/'data/camus_lv512' if organ=='heart' else ORG[organ]/'data'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def run_task(job,gpu):
 name=job['id'];root=R/'jobs'/name;root.mkdir(parents=True,exist_ok=False);method=job['method'];organ=job['organ'];seed=job['seed'];ks=job['ks'];ck=checkpoint(method,seed);data=dataset(organ);assert ck.is_file() and (data/'manifest.json').is_file()
 protocol=read(R/'protocols'/('protocol_'+organ+'.json'));protocol.update(source_checkpoint_seed=seed,support_K=ks,support_draw_seeds=[42],policy='Old full-data source; K8 is first eight of original seed42 K16; patient-disjoint original test.')
 e=dict(os.environ,CUDA_VISIBLE_DEVICES=UUUID[gpu],CUDA_DEVICE_ORDER='PCI_BUS_ID',USTS_GPU_INDEX=str(gpu),USTS_SOURCE_SEED=str(seed),USTS_SOURCE_CKPT=str(ck),USTS_ADAPT_DATA=str(data),USTS_ADAPT_METHOD=method,USTS_ADAPT_ORGAN=organ,USTS_ADAPT_KS=','.join(map(str,ks)),USTS_TARGET_TEXT=str(ORG[organ]/'records/target_text.pt'),USTS_EXTRA_TEXT=str(U/'five_baseline_supplement_20261027/records/extra_text.pt'),USTS_ADAPT_CODE=str(R/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PAPERLAB_PROGRESS='off',PYTHONHASHSEED='42',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 code=R/'code'/((('primary' if method in PRIMARY else 'extra')+'_'+organ)+'.py');expected_sha=sha(ck)
 pre_sha=None
 for stage in ['preflight','train_test']:
  loc=root/stage;loc.mkdir();write(loc/'protocol.json',protocol);env=dict(e,USTS_ADAPT_JOB_ROOT=str(loc),USTS_ADAPT_PREFLIGHT='1' if stage=='preflight' else '0')
  with (R/'logs'/(name+'_'+stage+'.log')).open('x') as log:
   child=subprocess.Popen([sys.executable,'-u',str(R/'code/child_exec.py'),str(code)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   pr=proc(child.pid);active[name].update(pid=child.pid,startticks=pr['startticks'] if pr else None,phase=stage)
   snapshot('running')
   rc=child.wait()
  active[name].update(pid=None,startticks=None)
  if rc:raise RuntimeError(stage+' failed exit '+str(rc))
  sub=loc/('preflights' if stage=='preflight' and method in EXTRA else 'results')/method/('preflight' if stage=='preflight' and method in EXTRA else 'adapt')
  pre=read(sub/'preflight.json');assert pre['passed'] and pre['source_weights_restored'] and pre['checkpoint_sha256']==expected_sha
  if stage=='preflight':pre_sha=pre['checkpoint_sha256'];continue
  assert pre['checkpoint_sha256']==pre_sha
  complete=read(sub/'complete.json');assert complete['passed'] and complete['settings']==len(ks)
  man=read(data/'manifest.json');byid={str(x['id']):x for x in man['rows']};expectedpatients={str(x['patient_id']) for x in man['rows'] if x['split']=='test'};expectedcases={str(x['id']) for x in man['rows'] if x['split']=='test'}
  for k in ks:
   d=sub/('K'+str(k)+'_seed42');r=read(d/'result.json');patients=read(d/'per_patient.json');cases=read(d/'per_case.json')
   support=man['supports']['42']['16'][:k];assert r['source_model_seed']==seed and r['support_draw_seed']==42 and r['K']==k and r['training']['successful_optimizer_updates']==100 and r['support_ids']==support and r['checkpoint_sha256']==expected_sha
   assert {str(x['patient_id']) for x in patients}==expectedpatients and {str(x['id']) for x in cases}==expectedcases
   for m in ['dice','iou','hd95']:
    assert abs(statistics.mean(x[m] for x in patients)-r[m])<1e-9
   assert (d/'adapted_state.pt').is_file()
 assert sha(ck)==expected_sha
 write(root/'complete.json',{'passed':True,'time':time.time(),'source_seed':seed,'support_seed':42,'Ks':ks,'checkpoint':str(ck),'checkpoint_sha256':expected_sha,'gpu':gpu,'evaluation':'complete_test_patient_mean','successful_updates':100})
 return name

def main():
 global pause_plan,watchdog
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 reuse=read(R/'records/reused_audit.json');assert reuse['passed'] and reuse['complete_settings']==54
 ta=read(R/'records/text_seed_audit.json')
 for method in ('clipseg_official','lvit_native224'):
  assert all(ta[method][str(s)]['code']==0 for s in (42,3407,2026))
  assert len({ta[method][str(s)]['result']['hash'] for s in (42,3407,2026)})==1
 fd=os.open(R/'records/controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
 owner=proc(os.getpid());write(R/'records/owner.json',{'pid':os.getpid(),'startticks':owner['startticks']})
 inventory_data,contexts=inventory();full=read(F/'records/status.json');assert full['authority']=='idle_parallel_independent_seeds_20261102' and not full['failed']
 for name,j in full['active_jobs'].items():
  rec=read(F/'records/jobs'/name/'status.json');pid=rec['pid'];p=proc(pid);gpu=j['physical_gpu'];assert gpu in UUUID and p and p['state'] not in ('Z','T','t')
  assert pid in contexts.get(UUUID[gpu],set()),(name,'not currently on expected GPU; defer handoff')
  pause_plan.append({'name':name,'gpu':gpu,'pid':pid,'startticks':p['startticks'],'stage':rec['stage']})
 assert len({x['gpu'] for x in pause_plan})==len(pause_plan)
 write(R/'records/pause_plan.json',{'jobs':pause_plan,'reason':'User requested temporary pause during old-scheme fine-tuning','resume':'SIGCONT same PID/startticks; preserve GPU context, optimizer and RNG in memory'})
 with (R/'logs/watchdog.log').open('x') as log:watchdog=subprocess.Popen([sys.executable,'-u',str(R/'code/watchdog.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(50):
  if (R/'records/watchdog_ready.json').exists():break
  if watchdog.poll() is not None:raise RuntimeError('Watchdog failed before pause')
  time.sleep(.1)
 else:raise RuntimeError('Watchdog readiness timeout')
 for j in pause_plan:
  assert same(j);os.kill(j['pid'],signal.SIGSTOP)
  for _ in range(30):
   p=proc(j['pid'])
   if p and p['startticks']==j['startticks'] and p['state'] in ('T','t'):break
   time.sleep(.1)
  else:raise RuntimeError('Could not verify stopped state for '+j['name'])
 snapshot('original_training_paused')
 jobs=[{'id':o+'__'+m+'__'+str(s),'organ':o,'method':m,'seed':s,'ks':[8] if s==42 else [4,8,16]} for s in [42,3407,2026] for m in PRIMARY+EXTRA for o in ORG]
 assert len(jobs)==81 and sum(len(x['ks']) for x in jobs)==189
 write(R/'records/jobs_manifest.json',{'jobs':jobs,'new_settings':189,'reuse_settings':54,'support_draw_seed':42,'forbidden_gpu':3})
 time.sleep(4)
 first=jobs.pop(0);d,ctx=inventory();assert not ctx.get(UUUID[0]) and d[0]['free']>=14000
 active[first['id']]={'gpu':0,'phase':'preflight','pid':None};snapshot('real_preflight_and_first_job')
 with ThreadPoolExecutor(max_workers=3) as pool:
  future=pool.submit(run_task,first,0);future.result();done.append(first['id']);del active[first['id']];snapshot('first_job_complete')
  futures={};last_launch=0
  while jobs or futures:
   for name,f in list(futures.items()):
    if f.done():
     try:f.result();done.append(name)
     except BaseException as ex:failures.append({'job':name,'error':str(ex),'traceback':traceback.format_exc()})
     del futures[name];del active[name]
   d,ctx=inventory();full=read(F/'records/status.json');reserved={x['gpu']:x for x in pause_plan if same(x)};used={v['gpu'] for v in active.values()}
   free=[]
   for gpu,u in UUUID.items():
    if gpu in used or d[gpu]['free']<14000 or d[gpu]['util']>5:continue
    held=reserved.get(gpu);expected={held['pid']} if held and proc(held['pid'])['state'] in ('T','t') else set()
    if held and not expected:raise RuntimeError('A paused original process resumed unexpectedly')
    if ctx.get(u,set())!=expected:continue
    if gpu in full.get('active',{}).values() and not held:continue
    free.append(gpu)
   if jobs and free and len(futures)<3 and mem_gib()>16 and time.monotonic()-last_launch>12:
    job=jobs.pop(0);gpu=free[0];active[job['id']]={'gpu':gpu,'phase':'assigned','pid':None};futures[job['id']]=pool.submit(run_task,job,gpu);last_launch=time.monotonic()
   snapshot('running',pending_groups=len(jobs),memory_available_gib=mem_gib());time.sleep(3)
 snapshot('jobs_finished',pending_groups=0)
 write(R/'records/all_finished.json',{'passed':not failures,'finished':len(done),'failed':failures,'expected_groups':81,'new_settings':189,'reuse_settings':54,'support_seed':42,'time':time.time()})

try:main()
except BaseException as e:
 failures.append({'controller_error':str(e),'traceback':traceback.format_exc()});write(R/'records/controller_failed.json',failures[-1]);snapshot('failed')
finally:
 # Normal path waits until every new GPU process is gone before restoring original GPU contexts.
 for _ in range(90):
  if not any(v.get('pid') and proc(v['pid']) and proc(v['pid'])['state']!='Z' for v in active.values()):break
  time.sleep(1)
 resumed=[];missing=[]
 for j in pause_plan:
  if same(j):
   p=proc(j['pid'])
   if p['state'] in ('T','t'):os.kill(j['pid'],signal.SIGCONT);resumed.append(j['pid'])
  else:missing.append(j['name'])
 write(R/'records/original_training_resumed.json',{'time':time.time(),'resumed_pids':resumed,'missing':missing,'failures':failures})
 print(json.dumps({'completed':len(done),'failed':len(failures),'resumed':resumed,'missing':missing}),flush=True)
