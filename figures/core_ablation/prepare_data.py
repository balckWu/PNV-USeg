"""Convert complete per-seed aggregate test metrics to a plotting summary."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

CONDITIONS = ['F', 'F_remove_dual', 'F_remove_summary', 'F_remove_all_text']
METRICS = {'dice': 'dice_percent', 'iou': 'iou_percent', 'hd95': 'hd95_pixels'}
FIELDS = {'condition', 'seed', *METRICS.values(), 'evaluation_split', 'result_status'}


def prepare(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(set(reader.fieldnames or [])):
            raise ValueError('Required CSV columns: ' + ', '.join(sorted(FIELDS)))
        records = list(reader)
    if not records:
        raise ValueError('The test metrics CSV is empty')
    indexed = {}
    for row in records:
        if row['evaluation_split'] != 'test' or row['result_status'] != 'test_complete':
            raise ValueError('Only completed held-out test rows are allowed')
        condition, seed = row['condition'], int(row['seed'])
        if condition not in CONDITIONS:
            raise ValueError('Unknown configuration: ' + condition)
        key = condition, seed
        if key in indexed:
            raise ValueError('Duplicate configuration/seed: ' + repr(key))
        numbers = {m: float(row[column]) for m, column in METRICS.items()}
        for metric, value in numbers.items():
            if not math.isfinite(value) or value < 0 or (metric != 'hd95' and value > 100):
                raise ValueError('Invalid percentage/pixel metric: ' + repr(key))
        indexed[key] = numbers
    seeds = sorted({s for _, s in indexed})
    if seeds != [42] and set(seeds) != {42, 3407, 2026}:
        raise ValueError('Use seed42 only, or all three seeds 42, 3407, 2026')
    missing = [(condition, seed) for condition in CONDITIONS for seed in seeds
               if (condition, seed) not in indexed]
    if missing:
        raise ValueError('Missing completed test configurations: ' + repr(missing))
    rows = []
    for condition in CONDITIONS:
        output = dict(condition=condition, test_mean={}, test_sd={},
                      change_mean={}, change_sd={})
        for metric in METRICS:
            values = [indexed[(condition, seed)][metric] for seed in seeds]
            changes = [(indexed[('F', seed)][metric] - indexed[(condition, seed)][metric])
                       * (-1 if metric == 'hd95' else 1) for seed in seeds]
            changes = [0.0 if abs(value) < 1e-12 else value for value in changes]
            output['test_mean'][metric] = statistics.mean(values)
            output['change_mean'][metric] = statistics.mean(changes)
            if len(seeds) == 3:
                output['test_sd'][metric] = statistics.stdev(values)
                output['change_sd'][metric] = statistics.stdev(changes)
        rows.append(output)
    return dict(training_seeds=seeds, rows=rows, evaluation_split='test',
                result_status='test_complete',
                three_seed_status='Complete' if len(seeds) == 3 else 'Pending',
                source_input=path.name,
                source_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary = prepare(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
