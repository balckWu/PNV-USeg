# TaskSegment paper experiment repository

See [README_CN.md](README_CN.md) for installation, the six-GPU scheduler, compact logging, the three-seed protocol, and the canonical SQLite results database.


## TransUNet pretraining

The main-table TransUNet run is `transunet_pretrained` and requires the
ImageNet-21k `R50+ViT-B_16.npz` checkpoint at
`pretrained/transunet/R50+ViT-B_16.npz`. Run
`scripts/prepare_transunet_pretrained.sh` or
`scripts/verify_transunet_pretrained.py` before preflight. The disabled
`transunet_scratch` entry is retained only for an optional initialization
ablation.


See `docs/CHANGELOG_V5.md` for the new formal ablation and unified monitor.
