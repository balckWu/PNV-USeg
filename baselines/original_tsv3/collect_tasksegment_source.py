#!/usr/bin/env python3
"""Zip TaskSegment source/config only; excludes data, checkpoints and third-party code."""
from pathlib import Path
from datetime import datetime
import zipfile
ROOT = Path('/tank/wk/TSV3').resolve()
OUT = ROOT / f'tasksegment_source_{datetime.now():%Y%m%d_%H%M%S}.zip'
EXCLUDE = {'data','paper_artifacts','third_party','pretrained','__pycache__','.git','.cache','figure_samples','result_summary_exports'}
ALLOW = {'.py','.json','.yaml','.yml','.toml','.txt','.md'}
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file(): continue
        r = p.relative_to(ROOT)
        if any(x in EXCLUDE for x in r.parts): continue
        if p.suffix.lower() not in ALLOW or p.stat().st_size > 2*1024*1024: continue
        z.write(p, arcname=r)
print('请上传这个文件：', OUT)
