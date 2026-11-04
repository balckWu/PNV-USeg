# Core ablation plotting code

The accepted layout combines a module configuration table with three horizontal bar panels showing changes relative to full PNV-USeg. The approved purple palette is in `palette.json`.

Install the plotting dependencies in `requirements.txt`, then provide your own completed test summary:

```bash
python plot_ablation.py --data /path/to/summary.json --out /path/to/figures
python prepare_data.py --input /path/to/per_seed_test.csv --output /path/to/summary.json
```

No example metric data or rendered images are included. CSV columns are `condition,seed,dice_percent,iou_percent,hd95_pixels,evaluation_split,result_status`. Conditions are `F`, `F_remove_dual`, `F_remove_summary`, and `F_remove_all_text`. All rows must have `evaluation_split=test` and `result_status=test_complete`.

The converter accepts either the four matched seed42 configurations or all four configurations at seeds 42, 3407, and 2026. It rejects duplicates, missing settings, incomplete tests, and validation rows. Dice/IoU use percentages; HD95 uses evaluation-grid pixels. Positive changes indicate degradation relative to the full model at the same seed. Three-seed error bars are sample SD of the paired differences, with denominator n−1; no significance test is performed. Outputs are PNG, editable-text SVG, and vector PDF.
