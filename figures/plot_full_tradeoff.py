"""Full-label Dice–HD95 tradeoff from complete per-domain, per-seed tests."""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['PNV-USeg', 'TransUNet', 'SAM-Med2D', 'Iris-2D', 'U-Net++',
          'U-Net', 'CLIPSeg', 'LViT', 'Swin-Unet']
COLORS = ['#D95F5F', '#326C9A', '#6D8F5B', '#8E77A5', '#B08352',
          '#3C9492', '#7D8C9B', '#A0798D', '#8B8769']
DOMAINS = {'thyroid', 'TN3K', 'BUSI_WHU', 'BUS-BRA', 'OTU', 'prostate'}
SEEDS = [42, 3407, 2026]


def aggregate(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    indexed = {}
    for row in rows:
        if row['evaluation_split'] != 'test' or row['result_status'] != 'test_complete':
            raise ValueError('Only completed held-out test rows are accepted')
        key = row['model'], int(row['train_seed']), row['dataset']
        if key in indexed:
            raise ValueError('Duplicate model/seed/domain')
        if key[0] not in MODELS or key[1] not in SEEDS or key[2] not in DOMAINS:
            raise ValueError('Unexpected comparison configuration')
        values = {m: float(row[m]) for m in ['dice', 'iou', 'hd95']}
        import math
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('Metrics must be finite and nonnegative')
        if values['dice'] > 1 or values['iou'] > 1:
            raise ValueError('Input Dice and IoU must be fractions')
        indexed[key] = values
    if len(indexed) != len(MODELS)*len(SEEDS)*len(DOMAINS):
        raise ValueError('Complete 9 methods × 3 seeds × 6 domains are required')
    result = {}
    for model in MODELS:
        per_seed = {seed: {metric: statistics.mean(indexed[model, seed, domain][metric]
                     for domain in sorted(DOMAINS)) for metric in ['dice', 'iou', 'hd95']}
                    for seed in SEEDS}
        result[model] = dict(per_seed=per_seed,
            mean={m: statistics.mean(per_seed[s][m] for s in SEEDS) for m in ['dice','iou','hd95']},
            sd={m: statistics.stdev(per_seed[s][m] for s in SEEDS) for m in ['dice','iou','hd95']})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = aggregate(args.data)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=.11, right=.95, bottom=.21, top=.79)
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        record = data[model]
        for seed, marker in zip(SEEDS, ['o', 's', '^']):
            value = record['per_seed'][seed]
            ax.scatter(value['hd95'], value['dice']*100, marker=marker,
                       color=color, s=28, alpha=.55, edgecolor='white', linewidth=.4)
        mean, sd = record['mean'], record['sd']
        ax.errorbar(mean['hd95'], mean['dice']*100, xerr=sd['hd95'], yerr=sd['dice']*100,
                    fmt='*' if i == 0 else 'D', color=color, markersize=13 if i == 0 else 6,
                    capsize=2.5, elinewidth=1.1, markeredgecolor='white', markeredgewidth=.5)
        ax.annotate(model, (mean['hd95'], mean['dice']*100), xytext=(8, 8 if i%2 == 0 else -13),
                    textcoords='offset points', fontsize=9, color=color,
                    fontweight='bold' if i == 0 else 'normal')
    ax.set_xlabel('HD95 (pixels)  ↓', labelpad=10)
    ax.set_ylabel('Dice (%)  ↑', labelpad=10)
    ax.grid(color='#E7EEF0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.margins(x=.16, y=.17)
    fig.text(.11, .94, 'Full-label performance across three training seeds', fontsize=17, fontweight='bold')
    fig.text(.11, .865, 'Nine methods · Equal-weight six-domain test macro averages', fontsize=10, color='#64717D')
    fig.text(.11, .10, 'Small markers: individual seeds 42, 3407, 2026 · Large markers: mean ± sample SD', fontsize=9, color='#64717D')
    fig.text(.11, .045, 'Error bars show marginal sample SD across training seeds; they are not confidence intervals.', fontsize=8.5, color='#64717D')
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(args.out/('Full_Three_Seed_Tradeoff_EN.'+ext), dpi=300)
    plt.close(fig)
    import json
    (args.out/'macro_statistics.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
