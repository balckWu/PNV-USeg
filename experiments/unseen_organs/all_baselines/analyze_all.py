"""CPU-only independent patient aggregation and complete 10-model tables."""
import hashlib,json,math,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
METHODS=('F','paper_US_adapter','iris2d','transunet_pretrained','unet','clipseg_official','lvit_native224','sammed2d_promptfree','unetpp','swinunet_official')
LABELS={'F':'F 正负文本＋摘要＋通道门控','paper_US_adapter':'原论文完整模型（适配版）','iris2d':'Iris-2D','transunet_pretrained':'TransUNet','unet':'U-Net','clipseg_official':'CLIPSeg','lvit_native224':'LViT（原生224）','sammed2d_promptfree':'SAM-Med2D（prompt-free）','unetpp':'U-Net++','swinunet_official':'Swin-Unet'}
OLD_ADAPT=METHODS[:5]
OLD_ZERO=('transunet_pretrained','unet','unetpp','swinunet_official')
ORGANS={'heart':'心脏左室腔','liver':'肝脏','gallbladder':'胆囊'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze():
 index={};checks=[];sources={}
 for organ in ORGANS:
  original=PARENT if organ=='heart' else PARENT/'lus7seg_20261027'/organ
  manifest=read(Path(read(ROOT/organ/'protocol.json')['existing_data_directory'])/'manifest.json')
  test={r['id']:r['patient_id'] for r in manifest['rows'] if r['split']=='test'}
  for method in METHODS:
   for mode in ('adapt','zero_target'):
    old=(method in OLD_ADAPT) if mode=='adapt' else (method in OLD_ZERO)
    directory=(original if old else ROOT/organ)/'results'/method/('noft' if mode=='zero_target' and old else mode)
    paths=sorted(directory.glob('*/result.json')) if mode=='adapt' or old else [directory/'result.json']
    for p in paths:
     if not p.exists():continue
     r=read(p);k=r['K'];draw=r.get('support_draw_seed',42)
     assert (mode=='adapt' and k in (4,16)) or (mode=='zero_target' and k==0)
     cases=read(p.parent/'per_case.json');patients=read(p.parent/'per_patient.json')
     assert len(cases)==len(test) and {x['id']:x['patient_id'] for x in cases}==test
     pids=sorted(set(test.values()));assert {x['patient_id'] for x in patients}==set(pids) and len(patients)==len(pids)
     if mode=='adapt':
      assert r['training']['successful_optimizer_updates']==100
      assert r['support_ids']==manifest['supports'][str(draw)][str(k)]
      assert len(r['support_ids'])==k
     elif not old:
      assert r['target_labelled_samples']==0 and r['target_train_images_read']==0 and not r['target_masks_used_in_model']
      assert r['source_parameters_unchanged']
     for metric in ('dice','iou','hd95'):
      assert all(math.isfinite(x[metric]) for x in cases)
      recomputed={pid:statistics.mean(x[metric] for x in cases if x['patient_id']==pid) for pid in pids}
      assert all(abs(x[metric]-recomputed[x['patient_id']])<1e-9 for x in patients)
      assert abs(r[metric]-statistics.mean(recomputed.values()))<1e-9
     rsha=r['checkpoint_sha256']
     if method in sources:assert sources[method]==rsha
     sources[method]=rsha
     key=(organ,method,mode,k);index.setdefault(key,[]).append(r)
     checks.append(dict(organ=organ,method=method,mode=mode,K=k,draw=draw,result=str(p),result_sha256=sha(p),passed=True))
 summary=[]
 for (o,m,mode,k),rr in index.items():
  row=dict(organ=o,method=m,mode=mode,K=k,n=len(rr),source_model_seed=42)
  for metric in ('dice','iou','hd95'):
   row[metric+'_mean']=statistics.mean(r[metric] for r in rr)
   row[metric+'_sd']=statistics.stdev(r[metric] for r in rr) if len(rr)>1 else 0.
  summary.append(row)
 complete=len(checks)==210 and len(summary)==90
 if complete:
  assert all(r['n']==(3 if r['mode']=='adapt' else 1) for r in summary)
 out=ROOT/'analysis';out.mkdir(exist_ok=True)
 payload=dict(complete=complete,completed_configurations=len(checks),expected_configurations=210,
              source_model_seed=42,standard_deviation='three target support draws, not source training seeds',
              checked_patient_aggregations=True,source_checkpoint_sha256=sources,summary=summary,checks=checks)
 (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 look={(r['organ'],r['method'],r['mode'],r['K']):r for r in summary}
 lines=['# 全部10模型：未见器官少样本微调及无目标样本预测','',
        f'已核验配置 {len(checks)}/210；完成状态：{complete}。源模型统一seed42。','',
        '微调K4/K16：相同患者支持抽样、100次成功AdamW更新、lr1e-5、CE+前景SoftDice、最后一步评估。±为三次支持抽样标准差。','',
        '直接预测：K=0，不提供任何目标训练图像或标注，不更新权重。F/Iris/原论文模型采用固定六原域原型等权均值，标注为源域原型迁移；CLIPSeg/LViT使用预先固定的器官文本；SAM不使用真值框/点/掩码提示。','',
        'LViT内部224，其余内部512；评估均512。保留原模型输入归一化与训练参数策略（CLIPSeg decoder；SAM adapters/no-mask embedding/mask decoder）。','',
        '心脏为CAMUS官方50测试患者200帧；肝脏481、胆囊392患者，LUS为新建患者划分。SAM-Med2D外部医学预训练的目标器官/患者暴露未认证排除，单独标注，不作严格未见于预训练的声明。','',
        '本补充是在原156设置结果已知后按用户要求新增，新增模型提示词/预算在其测试前固定；没有改原始实验或根据测试调参。','']
 def cell(o,m,mode,k,metric):
  r=look.get((o,m,mode,k))
  if r is None or r['n']!=(3 if mode=='adapt' else 1):return '待完成'
  scale=1 if metric=='hd95' else 100
  mean=f"{r[metric+'_mean']*scale:.2f}"
  return mean+(f"±{r[metric+'_sd']*scale:.2f}" if mode=='adapt' else '')
 def table(headers,rows):
  lines.append('| '+' | '.join(headers)+' |');lines.append('| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |')
  for row in rows:lines.append('| '+' | '.join(row)+' |')
  lines.append('')
 for metric,title in [('dice','Dice（%）↑'),('iou','IoU（%）↑'),('hd95','HD95（像素）↓')]:
  lines+=['## '+title,'','### 少样本微调','']
  table(['模型']+[ORGANS[o]+f' K{k}' for o in ORGANS for k in (4,16)],
        [[LABELS[m]]+[cell(o,m,'adapt',k,metric) for o in ORGANS for k in (4,16)] for m in METHODS])
  lines+=['### 无目标样本直接预测','']
  table(['模型']+list(ORGANS.values()),[[LABELS[m]]+[cell(o,m,'zero_target',0,metric) for o in ORGANS] for m in METHODS])
 lines+=['## 来源','',f'- 全量核验与路径：`{out}/summary.json`',
         '- 旧K4/K16免权重更新实验获得目标标注支持，另存于原报告，不混为K0直接预测。','']
 (out/'all_algorithms_comparison.md').write_text('\n'.join(lines))
 return payload

if __name__=='__main__':
 result=analyze();print('ALL_ALGORITHMS_ANALYSIS',result['completed_configurations'],result['complete'])
