"""CPU-only continuation: reorder jobs to overlap original encoder upload."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed_experiment',ROOT/'code/experiment_v2.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def main():
 old=e.json.loads((ROOT/'records/schedule_status.json').read_text())
 if old.get('running',{}).get('stage')!='waiting_for_original_encoder':raise RuntimeError('Scheduler no longer waiting')
 pid=int((ROOT/'records/schedule.lock').read_text())
 command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 if 'experiment_v2.py --schedule' not in command:raise RuntimeError('Unexpected old scheduler')
 env=e.gpu_env() # Must be idle and mapped to the reserved physical GPU6.
 os.kill(pid,15)
 for _ in range(50):
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(.1)
 else:raise RuntimeError('Old CPU scheduler did not exit')
 e.write(ROOT/'records/scheduler_order_amendment.json',dict(time=e.now(),old_cpu_pid=pid,new_cpu_pid=os.getpid(),
  reason='Overlap original frozen encoder asset transfer with other unchanged jobs; no model/protocol changes',
  worker_code_sha256=e.sha(ROOT/'code/experiment_v2.py'),old_schedule=old))
 (ROOT/'records/schedule.lock').write_text(str(os.getpid()))
 done=old['completed'];failed=old['failed']
 jobs=[('unetpp','noft'),('swinunet_official','noft')]+[(m,'adapt') for m in ('F','iris2d','unet','transunet_pretrained')]+[('paper_US_adapter','noft'),('paper_US_adapter','adapt')]
 for method,mode in jobs:
  if (ROOT/'results'/method/mode).exists():raise RuntimeError('Refusing duplicate job '+method+' '+mode)
  if method=='paper_US_adapter':
   for _ in range(1200):
    if (ROOT/'records/paper_target_text.pt').exists() or (ROOT/'records/biobert_encoding_failed.json').exists():break
    e.write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=dict(method=method,mode=mode,stage='waiting_for_original_encoder'),completed=done,failed=failed));time.sleep(1)
  env=e.gpu_env()
  with (ROOT/'logs'/(method+'_'+mode+'.log')).open('x') as log:
   child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/experiment_v2.py'),'--method',method,'--mode',mode],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   e.write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=dict(method=method,mode=mode,pid=child.pid),completed=done,failed=failed));code=child.wait()
  directory=ROOT/'results'/method/mode
  if code or not ((directory/'complete.json').exists() or (directory/'skipped.json').exists()):failed.append(dict(method=method,mode=mode,exit_code=code))
  else:done.append(dict(method=method,mode=mode,skipped=(directory/'skipped.json').exists()))
  print('JOB_FINISHED',method,mode,code,flush=True)
 e.write(ROOT/'records/schedule_status.json',dict(time=e.now(),running=None,completed=done,failed=failed))
 e.write(ROOT/'records/all_jobs_finished.json',dict(time=e.now(),completed=done,failed=failed))
if __name__=='__main__':main()
