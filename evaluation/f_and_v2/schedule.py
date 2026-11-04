"""CPU-only queue: all pending tests first, then two seed42 F controls."""
import os,subprocess,sys,time,traceback,statistics
from pathlib import Path
from common import *
def result_at(job):return ROOT/'results'/job['id']
def summarize(phase):
 results={}
 for job in read(ROOT/'jobs.json'):
  p=result_at(job)
  if (p/'complete.json').exists():
   r=read(p/'result.json');cases=[json.loads(s) for s in (p/'per_case.jsonl').read_text().splitlines()]
   assert len(cases)==1767
   for d in r['per_domain']:
    for m in ('dice','iou','hd95'):assert abs(r['per_domain'][d][m]-statistics.mean(c[m] for c in cases if c['domain']==d))<1e-9
   results[job['id']]=r
 external=PARENT/'f_10pct_seed42_20261028/evaluation/F'
 if (external/'complete.json').exists():results['F_10pct_seed42']=read(external/'result.json')
 write(ROOT/'records/current_summary.json',dict(time=now(),phase=phase,results=results,existing_tests=read(ROOT/'records/reused_tests_catalog.json')))
def run_job(job,kind,command,cwd):
 source=Path(job['source']);manifest=source/('source_manifest.json' if job['family']=='v2' else 'bundle_manifest.json')
 assert sha(manifest)==job['source_manifest_sha256'],('Source manifest changed',job['id'])
 env=environment(job['seed'])
 log_path=ROOT/'logs'/(job['id']+'_'+kind+'.log')
 with log_path.open('x') as log:
  child=subprocess.Popen([sys.executable,'-u']+command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(ROOT/'records/schedule_status.json',dict(time=now(),phase='initial_tests' if job['phase']==1 else 'F_ablation_seed42',running=dict(job=job['id'],kind=kind,pid=child.pid),physical_gpu=4))
  code=child.wait()
 assert code==0,('Worker failed',job['id'],kind,code,str(log_path))
 print('JOB_FINISHED',job['id'],kind,flush=True)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';verify()
 fd=os.open(ROOT/'records/schedule.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 jobs=read(ROOT/'jobs.json');initial=[j for j in jobs if j['phase']==1]
 external=PARENT/'f_10pct_seed42_20261028'
 while True:
  pending=[j for j in initial if not (result_at(j)/'complete.json').exists()]
  for job in pending:
   run=Path(job['run'])
   if (run/'failed.json').exists() or (run/'stopped.json').exists():raise RuntimeError('Source run failed/stopped: '+job['id'])
   if not (run/'complete.json').exists():continue
   p=Path('/proc')/str(job['training_pid'])
   if p.exists():continue
   worker='evaluate_f.py' if job['family']=='F' else 'evaluate_v2.py'
   run_job(job,'test',[str(ROOT/worker),'--job',job['id']],ROOT)
   assert (result_at(job)/'complete.json').exists();summarize('initial_tests')
  external_done=(external/'records/all_jobs_finished.json').exists() and (external/'evaluation/F/complete.json').exists()
  if (external/'records/scheduler_failed_v2.json').exists() or (external/'records/evaluation_failed.json').exists():raise RuntimeError('Existing F10% test pipeline failed; preserve and repair independently')
  remaining=[j['id'] for j in initial if not (result_at(j)/'complete.json').exists()]
  if not remaining and external_done:break
  write(ROOT/'records/schedule_status.json',dict(time=now(),phase='waiting_for_initial_training_or_test',running=None,pending=remaining,external_F10pct_test_done=external_done))
  time.sleep(30)
 assert read(ROOT/'records/reused_tests_catalog.json')['complete'],'Existing test catalog incomplete'
 summarize('all_initial_tests_complete');write(ROOT/'records/tests_all_finished.json',dict(time=now(),passed=True,initial_jobs=5,external_F10pct=True))
 # No new ablation may start before the full initial-test barrier above.
 for job in [j for j in jobs if j['phase']==2]:
  source=Path(job['source'])
  for kind,extra in [('preflight',['--preflight']),('train',[])]:
   run_job(job,kind,[str(source/'train_ablation.py'),'--seed','42','--physical-gpu','4']+extra,source)
   success=source/('records/preflight_seed42.json' if kind=='preflight' else 'runs/F/seed_42/complete.json')
   assert success.exists()
  run_job(job,'test',[str(ROOT/'evaluate_f.py'),'--job',job['id']],ROOT)
  assert (result_at(job)/'complete.json').exists();summarize('F_ablation_seed42')
 write(ROOT/'records/all_jobs_finished.json',dict(time=now(),passed=True));write(ROOT/'records/schedule_status.json',dict(time=now(),phase='complete',running=None))
if __name__=='__main__':
 try:main()
 except BaseException as e:write(ROOT/'records/scheduler_failed.json',dict(time=now(),error=str(e),traceback=traceback.format_exc()));raise
