import os,sys,time,subprocess,traceback,concurrent.futures,statistics,json
from pathlib import Path
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def run_job(j):
 rec=ROOT/'records/jobs'/j['id'];rec.mkdir(parents=True,exist_ok=True)
 source=Path(j['source']);run=Path(j['run'])
 def status(stage,**kw):write(rec/'status.json',dict(time=now(),stage=stage,job=j['id'],seed=j['seed'],physical_gpu=j['physical_gpu'],gpu_uuid=j['gpu_uuid'],**kw))
 try:
  assert not (rec/'launch.json').exists(),'A job launch already exists; no implicit restart'
  write(rec/'launch.json',dict(time=now(),worker_parent_pid=os.getpid(),job=j))
  for stage in ['preflight','train','test']:
   verify();wait_idle(j)
   env=environment(j['seed'],j['id']);env['USTS_JOB_ID']=j['id']
   if stage=='test':args=[str(ROOT/('evaluate_visual.py' if j['kind']=='visual' else 'evaluate_f.py')),'--job',j['id']];cwd=ROOT
   else:args=[str(source/'train_ablation.py'),'--seed',str(j['seed']),'--physical-gpu',str(j['physical_gpu'])]+(['--preflight'] if stage=='preflight' else []);cwd=source
   with (ROOT/'logs'/(j['id']+'_'+stage+'.log')).open('x') as log:
    p=subprocess.Popen([sys.executable,'-u']+args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]);status(stage,pid=p.pid,startticks=ticks)
    code=p.wait()
   assert code==0,(j['id'],stage,code)
   if stage=='preflight':
    pre=read(source/'records'/('preflight_seed'+str(j['seed'])+'.json'))
    assert pre['passed'] and pre['real_adamw_update'] and pre['all_six_domain_inputs_bitwise_equal']
    ref=read(ROOT/'reference_runs'/('F_seed'+str(j['seed'])+'.json'))
    for key in ['support_manifest','train_data_sha256','val_data_sha256']:assert pre[key]==ref[key],key
    assert pre['initial_state_sha256']==j['expected_initial_sha256']
    write(rec/'preflight_verified.json',dict(time=now(),passed=True,initial_hash=pre['initial_state_sha256'],all_six_domain_inputs_bitwise_equal=True,real_adamw_update=True,reference_support_and_data_match=True))
   elif stage=='train':
    complete=read(run/'complete.json');assert 1<=complete['last_epoch']<=150
   else:
    out=ROOT/'results'/j['id'];assert read(out/'complete.json')['cases']==1767
    assert read(out/'validation_reproduction.json')['passed']
    result=read(out/'result.json');cases=[json.loads(line) for line in (out/'per_case.jsonl').read_text().splitlines()]
    assert len(cases)==len({(x['domain'],str(x['sample_id'])) for x in cases})==1767
    refcases=[json.loads(line) for line in Path(j['full_test_dir'],'per_case.jsonl').read_text().splitlines()]
    assert [(x['domain'],str(x['sample_id'])) for x in cases]==[(x['domain'],str(x['sample_id'])) for x in refcases]
    maxerr=0.
    for d,means in result['per_domain'].items():
     for metric in ['dice','iou','hd95']:
      err=abs(statistics.mean(x[metric] for x in cases if x['domain']==d)-means[metric]);maxerr=max(maxerr,err);assert err<1e-9
    for metric in ['dice','iou','hd95']:assert abs(statistics.mean(v[metric] for v in result['per_domain'].values())-result['macro_'+metric])<1e-9
    result.update(evaluation_split='test',result_status='test_complete');write(out/'result.json',result)
    write(rec/'complete.json',dict(time=now(),passed=True,result_dir=str(out),cases=1767,case_id_sequence_matches_full=True,aggregation_max_error=maxerr,macro_dice=result['macro_dice'],macro_iou=result['macro_iou'],macro_hd95=result['macro_hd95']))
  status('complete');return j['id'],True
 except BaseException as e:
  write(rec/'failed.json',dict(time=now(),error=str(e),traceback=traceback.format_exc()));status('failed');return j['id'],False
def snapshot():
 rows={}
 for j in read(ROOT/'jobs.json'):
  p=ROOT/'records/jobs'/j['id']/'status.json';rows[j['id']]=read(p) if p.exists() else {'stage':'pending'}
 write(ROOT/'records/status.json',dict(time=now(),controller_pid=os.getpid(),authority='768_core_three_training_seeds_20261104',forbidden_physical_gpu=3,jobs=rows,reused=read(ROOT/'reused_results.json')))
if __name__=='__main__':
 verify();assert read(ROOT/'records/cpu_preparation_audit.json')['passed']
 write(ROOT/'records/controller_launch.json',dict(time=now(),pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split()[21]),python=sys.executable))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(run_job,j) for j in read(ROOT/'jobs.json')]
  while not all(f.done() for f in futures):snapshot();time.sleep(5)
  results=[f.result() for f in futures]
 snapshot();write(ROOT/'records/all_finished.json',dict(time=now(),passed=all(ok for _,ok in results),results=results,scope='Six new 768 core ablation runs plus six reused complete tests; no target adaptation or significance tests'))
