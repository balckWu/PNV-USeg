from pathlib import Path
import os,sys,time,json,signal,subprocess
R=Path(__file__).resolve().parents[1]

def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def ident(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return {'state':s[0],'startticks':int(s[19])}
 except (OSError,ValueError):return None
def same(pid,ticks):
 v=ident(pid);return v and v['startticks']==ticks and v['state']!='Z'
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
owner=read(R/'records/owner.json');plan=read(R/'records/pause_plan.json')
assert owner.get('pid') and plan.get('jobs') is not None
write(R/'records/watchdog_ready.json',{'pid':os.getpid(),'owner':owner})
while same(owner['pid'],owner['startticks']):time.sleep(3)
state=read(R/'records/status.json');killed=[]
for x in state.get('workers',{}).values():
 pid=x.get('pid');ticks=x.get('startticks')
 if pid and ticks and same(pid,ticks):
  try:os.killpg(pid,signal.SIGTERM);killed.append(pid)
  except ProcessLookupError:pass
for _ in range(40):
 if not any(same(x.get('pid',0),x.get('startticks',-1)) for x in state.get('workers',{}).values()):break
 time.sleep(.5)
for x in state.get('workers',{}).values():
 pid=x.get('pid');ticks=x.get('startticks')
 if pid and ticks and same(pid,ticks):
  try:os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError:pass
restored=[];deferred=[]
for j in plan['jobs']:
 if not same(j['pid'],j['startticks']):continue
 v=ident(j['pid'])
 if v['state'] not in ('T','t'):continue
 ready=False
 for _ in range(120):
  try:
   lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
   mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
   uuids={int(a.strip()):b.strip() for a,b in [line.split(',') for line in mapping]}
   pids={int(b.strip()) for line in lines for a,b in [line.split(',')] if a.strip()==uuids[j['gpu']]}
   if pids<={j['pid']}:ready=True;break
  except (OSError,ValueError,subprocess.SubprocessError):pass
  time.sleep(.5)
 if ready:os.kill(j['pid'],signal.SIGCONT);restored.append(j['pid'])
 else:deferred.append(j['pid'])
write(R/'records/watchdog_recovery.json',{'reason':'controller_exit','resumed':restored,'deferred_due_to_GPU_context':deferred,'terminated_workers':killed,'time':time.time()})
