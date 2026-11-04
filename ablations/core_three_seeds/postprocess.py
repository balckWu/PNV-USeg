import os,time,json,csv,statistics,traceback,subprocess,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from common import read,write,now,sha
def aggregate():
 entries=read(R/'reused_results.json')+[dict(condition=j['condition'],seed=j['seed'],result_dir=str(R/'results'/j['id'])) for j in read(R/'jobs.json')]
 raw={};audit=[];domains=None;ids_by_seed={}
 for item in sorted(entries,key=lambda r:(r['condition']!='F',r['condition'],r['seed'])):
  folder=Path(item['result_dir']);assert read(folder/'complete.json')['cases']==1767 and read(folder/'validation_reproduction.json')['passed']
  result=read(folder/'result.json');cases=[json.loads(s) for s in (folder/'per_case.jsonl').read_text().splitlines()]
  assert len(cases)==1767 and len({(r['domain'],str(r['sample_id'])) for r in cases})==1767
  ids=[(r['domain'],str(r['sample_id'])) for r in cases]
  if item['condition']=='F':ids_by_seed[item['seed']]=ids
  else:assert ids==ids_by_seed[item['seed']]
  per=result.get('per_domain',result.get('per_dataset'))
  assert per
  domains=list(per)
  for d in domains:
   for metric in ['dice','iou','hd95']:assert abs(statistics.mean(r[metric] for r in cases if r['domain']==d)-per[d][metric])<1e-9
  for metric in ['dice','iou','hd95']:assert abs(statistics.mean(r[metric] for r in per.values())-result['macro_'+metric])<1e-9
  raw[(item['condition'],item['seed'])]=result
  audit.append(dict(**item,per_case_sha256=sha(folder/'per_case.jsonl'),result_sha256=sha(folder/'result.json'),cases=1767,case_ids_match_full=True,aggregation_passed=True))
 conditions=['F','F_remove_dual','F_remove_summary','F_remove_all_text'];seeds=[42,3407,2026];rows=[];table=[]
 for cond in conditions:
  row={'condition':cond,'test_mean':{},'test_sd':{},'change_mean':{},'change_sd':{}}
  for domain in ['macro']+domains:
   for metric in ['dice','iou','hd95']:
    values=[raw[(cond,s)]['macro_'+metric] if domain=='macro' else raw[(cond,s)]['per_domain'][domain][metric] for s in seeds]
    scale=100 if metric!='hd95' else 1
    mean=statistics.mean(values)*scale;sd=statistics.stdev(values)*scale
    table.append(dict(condition=cond,dataset=domain,metric=metric,mean=mean,sample_sd=sd,training_seeds='42;3407;2026',evaluation_split='test',result_status='test_complete'))
    if domain=='macro':
     row['test_mean'][metric]=mean;row['test_sd'][metric]=sd
     changes=[(raw[('F',s)]['macro_'+metric]-raw[(cond,s)]['macro_'+metric])*(1 if metric!='hd95' else -1)*scale for s in seeds]
     row['change_mean'][metric]=statistics.mean(changes);row['change_sd'][metric]=statistics.stdev(changes)
  rows.append(row)
 out=R/'comparisons';out.mkdir(exist_ok=True)
 data=dict(evaluation_split='test',result_status='test_complete',training_seeds=seeds,rows=rows)
 write(out/'plot_data_three_seeds.json',data);write(out/'independent_aggregation_audit.json',dict(time=now(),passed=True,configurations=len(entries),records=audit))
 with (out/'core_ablation_three_seeds_all_domains.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 plot=subprocess.run([sys.executable,str(R/'plot_ablation.py'),'--data',str(out/'plot_data_three_seeds.json'),'--out',str(out/'figures')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 write(R/'records/aggregation_complete.json',dict(time=now(),passed=True,configurations=len(entries),plot_passed=plot.returncode==0,plot_output=plot.stdout))
if __name__=='__main__':
 write(R/'records/postprocess_launch.json',dict(time=now(),pid=os.getpid()))
 try:
  while not (R/'records/all_finished.json').exists():time.sleep(15)
  assert read(R/'records/all_finished.json')['passed'],'One or more training/testing jobs failed; no three-seed result fabricated'
  aggregate()
 except BaseException as e:
  write(R/'records/aggregation_failed.json',dict(time=now(),error=str(e),traceback=traceback.format_exc()));raise
