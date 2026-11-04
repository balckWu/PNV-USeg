"""Independently validate and aggregate completed paired target test records."""
import json,os,time,statistics,traceback,datetime,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;U=B/'unseen_organs_20261027'
parents={'heart':U,'liver':U/'lus7seg_20261027/liver','gallbladder':U/'lus7seg_20261027/gallbladder'}
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(p,data):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def check_set(directory,expected_count):
 res=read(directory/'result.json');cases=read(directory/'per_case.json');patients=read(directory/'per_patient.json')
 assert len(patients)==expected_count and len(cases)==res['test_frames']
 groups={}
 for case in cases:groups.setdefault(case['patient_id'],[]).append(case)
 assert len(groups)==len(patients) and len({x['patient_id'] for x in patients})==len(patients)
 for row in patients:
  assert row['patient_id'] in groups
  for m in ('dice','iou','hd95'):assert abs(row[m]-statistics.mean(x[m] for x in groups[row['patient_id']]))<1e-9
 for m in ('dice','iou','hd95'):assert abs(res[m]-statistics.mean(x[m] for x in patients))<1e-9
 return res,{x['patient_id'] for x in patients}
def main():
 while not (R/'records/complete.json').exists():
  if (R/'records/failed.json').exists():raise RuntimeError('adaptation controller failed; see failed.json')
  time.sleep(10)
 c=read(R/'records/complete.json');assert c['passed'] and c['settings']==18
 raw=[]
 for row in c['paired']:
  o,k,seed=row['organ'],row['K'],row['support_seed'];par=parents[o]
  a=Path(row['run'])/f'results/F/adapt/K{k}_seed{seed}';b=par/f'results/F/adapt/K{k}_seed{seed}'
  data=U/'data/camus_lv512' if o=='heart' else par/'data';manifest=read(data/'manifest.json')
  new,new_ids=check_set(a,manifest['test_patients']);full,full_ids=check_set(b,manifest['test_patients'])
  assert new_ids==full_ids and new['support_ids']==full['support_ids']==manifest['supports'][str(seed)][str(k)]
  assert new['training']['successful_optimizer_updates']==full['training']['successful_optimizer_updates']==100
  assert new['checkpoint_sha256']=='4894238b8340a69f196fffc70a4db54a83326bdb0c10c16d8405f0468c1b8a5c'
  raw.append({'organ':o,'K':k,'support_seed':seed,'test_patients':len(new_ids),'full':{m:full[m] for m in ('dice','iou','hd95')},'without_dual':{m:new[m] for m in ('dice','iou','hd95')}})
 assert len(raw)==len({(x['organ'],x['K'],x['support_seed']) for x in raw})==18
 summary=[]
 for o in parents:
  for k in (4,16):
   rs=[x for x in raw if x['organ']==o and x['K']==k];assert len(rs)==3
   d={'organ':o,'K':k,'test_patients':rs[0]['test_patients']}
   for method in ('full','without_dual'):
    d[method]={m:{'mean':statistics.mean(x[method][m] for x in rs),'sample_sd':statistics.stdev(x[method][m] for x in rs)} for m in ('dice','iou','hd95')}
   d['paired_difference_full_minus_without_dual']={m:statistics.mean(x['full'][m]-x['without_dual'][m] for x in rs) for m in ('dice','iou','hd95')}
   summary.append(d)
 write(R/'records/independent_audit.json',{'time':now(),'passed':True,'settings':18,'paired_supports_identical':True,'patient_sets_identical':True,'per_case_to_patient_and_mean_recomputed':True,'source_training_seeds':1,'support_draws':3,'raw':raw})
 write(R/'records/comparison.json',{'time':now(),'evaluation_split':'test','result_status':'test_complete','support_variation':'three support draws; source seed42','summary':summary})
 lines=['# Paired target adaptation: full PNV-USeg versus remove-positive-negative','',f'Completed: {now()}','', 'Each cell: mean ± sample SD over three paired support draws; source training seed 42. Dice/IoU in percent; HD95 in pixels.','', '| Organ | K | Full Dice | Remove dual Dice | Full IoU | Remove dual IoU | Full HD95 | Remove dual HD95 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for d in summary:
  def fmt(method,m):
   x=d[method][m];factor=100 if m in ('dice','iou') else 1
   return f"{x['mean']*factor:.2f} ± {x['sample_sd']*factor:.2f}"
  lines.append(f"| {d['organ']} | {d['K']} | {fmt('full','dice')} | {fmt('without_dual','dice')} | {fmt('full','iou')} | {fmt('without_dual','iou')} | {fmt('full','hd95')} | {fmt('without_dual','hd95')} |")
 lines += ['', 'Both target supports and test patients are paired. The three repeats are support draws, not independent source-training seeds. All settings are reported; no test-based selection.']
 (R/'records/comparison.md').write_text('\n'.join(lines)+'\n')
 write(R/'records/collection_complete.json',{'time':now(),'passed':True,'settings':18,'summary':str(R/'records/comparison.md')})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(R/'records/collection_failed.json',{'time':now(),'error':str(e),'traceback':traceback.format_exc()});raise
