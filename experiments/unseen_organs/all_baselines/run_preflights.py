import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=('unetpp','swinunet_official','clipseg_official','lvit_native224','sammed2d_promptfree')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
spec=importlib.util.spec_from_file_location('extra_worker',ROOT/'heart/code/experiment_v2.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
done=[]
for method in METHODS:
 assert not (ROOT/'heart/preflights'/method).exists(),'Refusing duplicate preflight'
 env=e.gpu_env()
 with (ROOT/'logs'/('preflight_'+method+'.log')).open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'heart/code/experiment_v2.py'),'--method',method,'--mode','preflight'],env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (ROOT/'records/preflight_status.json').write_text(json.dumps(dict(method=method,pid=p.pid,done=done)))
  code=p.wait()
 if code:raise RuntimeError('Preflight failed: '+method)
 result=json.loads((ROOT/'heart/preflights'/method/'preflight/preflight.json').read_text());assert result['passed'] and result['source_weights_restored']
 done.append(method);print('PREFLIGHT_PASSED',method,flush=True)
(ROOT/'records/preflights_finished.json').write_text(json.dumps(dict(done=done,passed=True,time=e.now()),indent=2))
