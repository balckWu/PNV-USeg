import os,sys,time,json,hashlib,subprocess,statistics,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent;U=B/'unseen_organs_20261027';OLD=B/'old_three_source_transfer_20261102_v1'
SRC=B/'f_parallel_20261029/jobs/F_remove_dual/source/runs/F/seed_42/best.pt'
SHA='4894238b8340a69f196fffc70a4db54a83326bdb0c10c16d8405f0468c1b8a5c'
UUID='GPU-d18bd3a8-741c-30dc-2039-4f358724578f';DENY='GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468'
ORG={'heart':U,'liver':U/'lus7seg_20261027/liver','gallbladder':U/'lus7seg_20261027/gallbladder'}
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(4194304),b''):h.update(v)
 return h.hexdigest()
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();d={int(v[0]):v for x in rows for v in [[y.strip() for y in x.split(',')]]}
 assert d[0][1]==UUID and d[3][1]==DENY
 ctx=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return int(d[0][2])<=256 and int(d[0][3])<=5 and UUID not in ctx
def main():
 assert sha(SRC)==SHA
 write(R/'records/status.json',{'stage':'waiting_for_authorized_three_seed_queue','source_seed':42,'support_seed':42,'K':8})
 while not (OLD/'records/all_finished.json').exists() or not (OLD/'records/original_training_resumed.json').exists():
  if (OLD/'records/controller_failed.json').exists():raise RuntimeError('Prior queue controller failed; manual review required')
  time.sleep(15)
 prior=read(OLD/'records/all_finished.json');assert prior['passed']
 resume=read(OLD/'records/original_training_resumed.json');assert not resume['missing']
 for organ,parent in ORG.items():
  while not idle():time.sleep(15)
  data=U/'data/camus_lv512' if organ=='heart' else parent/'data';man=read(data/'manifest.json');support=man['supports']['42']['16'][:8]
  byid={str(x['id']):x for x in man['rows']};assert len(support)==8 and len({byid[str(x)]['patient_id'] for x in support})==8
  assert all(byid[str(x)]['split']=='train' for x in support)
  assert not {byid[str(x)]['patient_id'] for x in support}&{x['patient_id'] for x in man['rows'] if x['split']=='test'}
  ck=sha(SRC);assert ck==SHA
  for phase in ('preflight','train_test'):
   while not idle():time.sleep(15)
   root=R/'jobs'/organ/phase;root.mkdir(parents=True,exist_ok=False);p=read(OLD/'protocols'/('protocol_'+organ+'.json'));p.update(source_checkpoint_seed=42,support_K=[8],support_draw_seeds=[42]);write(root/'protocol.json',p)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',USTS_GPU_INDEX='0',USTS_SOURCE_SEED='42',USTS_SOURCE_CKPT=str(SRC),USTS_ADAPT_DATA=str(data),USTS_ADAPT_METHOD='F',USTS_ADAPT_ORGAN=organ,USTS_ADAPT_KS='8',USTS_TARGET_TEXT=str(parent/'records/target_text.pt'),USTS_ADAPT_CODE=str(R/'code'),USTS_ADAPT_JOB_ROOT=str(root),USTS_ADAPT_PREFLIGHT='1' if phase=='preflight' else '0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42',PYTHONUNBUFFERED='1')
   with (R/'logs'/(organ+'_'+phase+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,'-u',str(R/'code'/('dual8_'+organ+'.py'))],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(R/'records/status.json',{'stage':phase,'organ':organ,'pid':child.pid,'gpu':0,'source_seed':42,'support_seed':42,'K':8})
    rc=child.wait()
   if rc:raise RuntimeError(organ+' '+phase+' failed '+str(rc))
   d=root/'results/F/adapt';pre=read(d/'preflight.json');assert pre['passed'] and pre['source_weights_restored'] and pre['checkpoint_sha256']==SHA
   if phase=='preflight':continue
   c=read(d/'complete.json');assert c['passed'] and c['settings']==1
   run=d/'K8_seed42';result=read(run/'result.json');patients=read(run/'per_patient.json');cases=read(run/'per_case.json')
   assert result['K']==8 and result['source_model_seed']==42 and result['support_draw_seed']==42 and result['support_ids']==support and result['training']['successful_optimizer_updates']==100 and result['checkpoint_sha256']==SHA
   assert {str(x['patient_id']) for x in patients}=={str(x['patient_id']) for x in man['rows'] if x['split']=='test'}
   assert {str(x['id']) for x in cases}=={str(x['id']) for x in man['rows'] if x['split']=='test'}
   for m in ('dice','iou','hd95'):assert abs(statistics.mean(x[m] for x in patients)-result[m])<1e-9
   assert (run/'adapted_state.pt').is_file()
  write(R/'records'/('complete_'+organ+'.json'),{'passed':True,'organ':organ,'K':8,'source_seed':42,'support_seed':42,'checkpoint_sha256':SHA})
 write(R/'records/all_finished.json',{'passed':True,'organs':list(ORG),'source_seed':42,'support_seed':42,'K':8,'prior_K4_K16':'reuse verified original results'})
 write(R/'records/status.json',{'stage':'complete'})
try:main()
except BaseException as ex:
 write(R/'records/failed.json',{'error':str(ex),'traceback':traceback.format_exc()})
 write(R/'records/status.json',{'stage':'failed','error':str(ex)})
 raise
