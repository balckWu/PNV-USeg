# PNV-USeg

**Few-Shot Ultrasound Image Segmentation via Fusion of Positive–Negative Medical Semantics and Visual Prototypes**

Source-only research code for the 768-dimensional PNV-USeg model (internal experiment ID `F`), its training and held-out evaluation, eight comparison methods, matched core ablations, cross-organ fine-tuning, and paper figures.

## Code map

| Component | Source |
|---|---|
| Main 768 model, positive–negative semantic head, dense guidance | [`main_f/current_multiseed/campaign/model.py`](main_f/current_multiseed/campaign/model.py) |
| Original seed42 training and input-cache engine | [`main_f/seed42/`](main_f/seed42/) |
| Additional full-model training seeds | [`main_f/current_multiseed/fresh_train.py`](main_f/current_multiseed/fresh_train.py) |
| BioLORD text encoding and fixed prompts | [`main_f/current_multiseed/campaign/`](main_f/current_multiseed/campaign/) |
| Eight baselines using the selected original configuration | [`baselines/original_tsv3/`](baselines/original_tsv3/) |
| Remove positive–negative branch / remove dense guidance, seed42 | [`ablations/f_remove_dual/`](ablations/f_remove_dual/), [`ablations/f_remove_summary/`](ablations/f_remove_summary/) |
| F-derived removal of both text branches, seed42 | [`ablations/f_remove_all_text/`](ablations/f_remove_all_text/) |
| All six added core-ablation runs, seeds 3407 and 2026 | [`ablations/core_three_seeds/`](ablations/core_three_seeds/) |
| Original target-organ preparation and adaptation | [`experiments/unseen_organs/`](experiments/unseen_organs/) |
| Selected original-recipe three-source-seed K4/K8/K16 adaptation | [`experiments/unseen_organs/old_recipe_three_source/`](experiments/unseen_organs/old_recipe_three_source/) |
| Target adaptation without the positive–negative branch | [`experiments/unseen_organs/without_pn/`](experiments/unseen_organs/without_pn/), [`without_pn_k8/`](experiments/unseen_organs/without_pn_k8/) |
| Validation reproduction and full test evaluators | [`evaluation/`](evaluation/), [`ablations/core_three_seeds/evaluate_f.py`](ablations/core_three_seeds/evaluate_f.py) |
| Paper plotting code | [`figures/`](figures/) |

The active paper protocol is recorded in [`configs/paper_protocol.json`](configs/paper_protocol.json). The eight baselines are U-Net, U-Net++, TransUNet, Swin-Unet, CLIPSeg, LViT, SAM-Med2D, and Iris-2D. The original-recipe source is retained as selected; later official-recipe replacements are not substituted into this paper configuration.

## Start here

```bash
git clone https://github.com/balckWu/PNV-USeg.git
cd PNV-USeg
python -m pip install -r requirements.txt
python tools/validate_release.py
```

Read [paper reproduction](docs/REPRODUCE_PAPER.md) for environment, data layout, external assets, training/evaluation entry points, and path migration. Each experiment snapshot has its own import root; do not combine multiple `tasksegment`/`campaign` copies in one Python process.

These are the actual experimental source snapshots, including model, data-loading, sampling, augmentation, loss, optimizer, checkpoint, test, and adaptation implementations. Original paths and fingerprint checks are preserved for traceability. The snapshot entry points require the documented data, text bank, and path configuration. The source release has been audited; the full private-data experiments have not been rerun from this GitHub checkout.

## Paper figures

```bash
python figures/core_ablation/plot_ablation.py --data /path/to/ablation_summary.json --out /path/to/figures/ablation
python figures/plot_radar.py --data /path/to/finetuning_profiles.json --out /path/to/figures/radar
python figures/plot_full_tradeoff.py --data /path/to/per_seed_test.csv --out /path/to/figures/full_tradeoff
```

This repository contains plotting code only. Supply aggregate result inputs separately; result tables, example metric values, rendered figures, and prediction files are not distributed. The ablation converter accepts the complete three-seed configuration set after all tests finish. Source-training seeds and support-draw seeds are separate quantities.

## Assets and provenance

Prepare datasets, authorized patient splits, source checkpoints, text embeddings, and upstream pretrained assets separately. They are not part of this source release. Required assets and model-specific dependencies are listed in the reproduction guide. Third-party notices and licenses are preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and vendor directories; no new blanket license is imposed on those sources.

[`provenance/source_manifest.json`](provenance/source_manifest.json) records the earlier source collection. [`provenance/latest_source_manifest.json`](provenance/latest_source_manifest.json) records the newly retrieved, hash-verified frozen sources. The manifests list the retained paper source files. Method and experiment choices are defined by `configs/paper_protocol.json`.
