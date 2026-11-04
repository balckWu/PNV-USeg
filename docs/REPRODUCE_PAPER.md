# Reproducing the paper experiments

## Environment and external inputs

Use Python 3.10–3.12 and a PyTorch build appropriate for your hardware. Install `requirements.txt`; use the separate `requirements-iris2d.txt`, `requirements-clipseg.txt`, `requirements-lvit.txt`, and `requirements-sammed2d.txt` for the corresponding baselines. Install each upstream model's dependencies and pretrained assets as described in its included source. Plot-only dependencies are pinned in `figures/core_ablation/requirements.txt`.

The source-domain data root must contain `thyroid`, `TN3K`, `BUSI_WHU`, `BUS-BRA`, `OTU`, and `prostate`. Each domain has `train/images`, `train/masks`, `val/images`, `val/masks`, `test/images`, and `test/masks`. Images and masks share their stem; the loader also accepts matching PNG masks. Preserve the registered splits. Expected totals are 5,892 training images, 1,361 validation images, and 1,767 held-out test images. A file name alone is not a cross-dataset patient identifier.

The frozen source uses `DATA` in `campaign/train.py` and snapshot-specific constants in target-organ scripts. Configure those in a separate working copy. Original source fingerprints include text-cache hashes; a relocated or modified run needs its own recorded source manifest rather than silently bypassing integrity checks. Original data fingerprints include paths, sizes, and modification times, so moving a dataset changes that fingerprint even if pixels are identical.

Prepare the frozen BioLORD-2023 encoder locally and use `campaign/encode_text.py --model /path/to/BioLORD` from the chosen source root with that root on `PYTHONPATH`. Encoding runs on CPU and uses the fixed prompts, attention-mask mean pooling, and L2 normalization. The source bank has shape `[6, 9, 768]`: four positive, four negative, and one summary vector per domain. Encoder weights and generated `campaign/text_bank.pt` are external assets. Do not substitute the legacy BioBERT bank.

## Main training and test

Read `main_f/seed42/campaign/train.py` for the original seed42 fresh-start training and `main_f/seed42/cache_v2/` for the actual cached continuation. For additional full-model seeds, `main_f/current_multiseed/fresh_train.py` is the frozen from-scratch entry with `--seed`, `--physical-gpu`, and `--preflight`. Set `USTS_TRAIN_SEED` consistently. The scripts explicitly bind a live GPU UUID and check source and text fingerprints; set up `campaign/gpu_map.json` for the intended machine in a recorded working copy.

All source-domain runs use 512 × 512 inputs, four support slots and four query slots per episode, 1,500 episodes per epoch, gradient accumulation 4, up to 150 epochs, and validation early stopping with patience 20. Main-model AdamW uses learning rate 5e-5 and weight decay 1e-4. Preserve the scheduler, losses, AMP behavior during training, and validation checkpoint selection in the source. Baseline-specific settings are retained in the selected baseline scripts; they are not all identical to the main model's optimizer or native resolution.

Held-out testing uses the validation-selected best checkpoint, FP32, a 512 evaluation grid, threshold 0.5, and no postprocessing. See `evaluation/main_six_variants/test_runner_v2.py`, `evaluation/f_and_v2/evaluate_f.py`, and the latest evaluators under `ablations/core_three_seeds/`. These verify validation reproduction, saved versus rebuilt support prototypes, and complete test coverage. The evaluator's queue metadata and source/checkpoint paths must refer to the intended working copy; historical controllers are not a generic one-command launcher.

Compute image metrics within each domain, then average the six domain means equally. For three training seeds, compute a macro result for each seed before calculating the mean and sample SD. Never average domain SD values to obtain a macro SD. Only completed held-out test results enter formal paper tables.

## Core 2 × 2 ablation

The four conditions are full `F`, `F_remove_dual`, `F_remove_summary`, and F-derived `F_remove_all_text`. All retain visual prototypes, visual interaction, and visual channel gating. The old `no_text_v2` architecture is not the fourth row of this ablation and is not included in this paper source release.

Seed42 deletions are in `ablations/f_remove_dual`, `ablations/f_remove_summary`, and `ablations/f_remove_all_text`. The six frozen additional runs are in `ablations/core_three_seeds/jobs/<condition>_seed<seed>/source`, with seeds 3407 and 2026. Each includes the actual model, training loop, same-seed initialization logic, and its expected initialization metadata. `structural_init.py` maps retained tensors from the same-seed random full-model initialization; no trained full-model weights initialize the source-training ablations. The visual-only run avoids production text-asset loads.

The latest queue controller, CPU audit, test evaluators, and postprocessing code are included in `ablations/core_three_seeds`. Those scripts retain deployment-specific paths; inspect and configure them before use. The code release does not start or reschedule server jobs. Physical GPU 3 and the recorded prohibited GPU UUID remain excluded.

## Eight baselines

Use the independent source root `baselines/original_tsv3`. Training scripts in its `scripts/` directory cover `train_unet.py`, `train_unetpp.py`, `train_transunet_official.py`, `train_swinunet_official.py`, `train_clipseg_official.py`, `train_lvit_official.py`, `train_sammed2d_promptfree.py`, and `train_iris2d.py`. Corresponding `predict_*.py` scripts and training metric implementations are included. Consult `--help` for each baseline's own arguments after installing its optional dependencies. Keep LViT's native input handling, Iris-2D's adaptation, and the selected CLIPSeg/SAM freezing and pretrained policies.

For example, after explicitly binding an allowed device, U-Net accepts `python scripts/train_unet.py --data-root /path/to/data --save-path /path/to/run/model.pth --seed 42 --deterministic`; its matching evaluator is `scripts/predict_baseline.py --method unet --checkpoint /path/to/run/model.pth --data-root /path/to/data --output-dir /path/to/test --no-amp`. Preserve the archived paper configuration rather than assuming every command-line default matches every run.

## Cross-organ fine-tuning

Target preparation and text encoding for CAMUS left ventricle and LUS liver/gallbladder are in `experiments/unseen_organs`. The chosen original-recipe three-source-seed implementation is in `old_recipe_three_source/code/`, with registered protocols in `old_recipe_three_source/protocols/`. It references the original source-model families rather than later official-recipe replacements.

Fine-tuning uses 100 successful optimizer updates. K denotes the registered target annotation budget, and support/test patients must be disjoint. The displayed K4/K8/K16 profiles use source training seeds 42, 3407, and 2026 with support-draw seed 42; K8 is the first eight registered supports of K16. Consult the per-organ protocol for image/frame and patient handling. Source training seeds and support sampling seeds are separate axes of variation.

`without_pn/code/{heart,liver,gallbladder}.py` is the completed historical target-adaptation ablation without positive–negative semantics, retaining dense summary guidance. It uses source seed42, K4/K16, and the three recorded support draws. `without_pn_k8/code/` contains the K8 extension. It is a distinct experiment from the complete-model versus eight-baseline comparison and must not be described as removing all text.

## Validation scope

`python tools/validate_release.py` checks source hashes, Python/JSON syntax, required active entry points, asset/credential exclusions without CUDA. `python tools/smoke_model_cpu.py` performs a small synthetic CPU interface check; it neither trains a model nor reproduces any reported test score. Full private-data training, target adaptation, and held-out evaluation require the external assets above and are not claimed as rerun by this packaging operation.
