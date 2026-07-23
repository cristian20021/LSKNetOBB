# LSKNet backbone ablation for AOPG (Anchor-free Oriented Proposal Generator)

This repository contains a working port of the [LSKNet](https://github.com/zcablii/LSKNet)
backbone (Li et al., ICCV 2023 / IJCV 2024) into
[AOPG](https://github.com/jbwang1997/AOPG) (Wang et al., TGRS 2022), swapping out
AOPG's original ResNet-50 backbone for LSKNet-S, and evaluated on the DIOR-R
oriented object detection benchmark.

## Result

| Backbone     | mAP (DIOR-R test set) |
|--------------|------------------------|
| ResNet-50 (AOPG paper) | 64.41% |
| **LSKNet-S (this repo)** | **64.95%** |

Trained for 12 epochs, identical AOPG head/schedule/hyperparameters to the
original paper, only the backbone and matching FPN input channels were changed.

## What's in this repo

- `mmdet/models/backbones/lsknet.py` — LSKNet backbone
- `configs/obb/aopg/aopg_lsknet_s_fpn_1x_dior.py` — training configurations
- `tools/visualize_lsknet.py` — visualizes per-stage backbone feature maps
- `tools/visualize_predictions_vs_gt.py` — visualizes predicted vs.
  ground-truth oriented boxes side by side
- `demo/huge_image_demo.py` — patched to save output to disk 

## Setup

See [the original AOPG install instructions](https://github.com/jbwang1997/AOPG/blob/master/docs/install.md)
for base environment setup (Python 3.7, PyTorch 1.10.1, mmcv 0.6.2, CUDA 11.3).
Additionally requires `timm==0.6.13` for the LSKNet backbone.

Download LSKNet-S ImageNet-pretrained weights from the
[official LSKNet repo](https://github.com/zcablii/LSKNet) and place at
`pretrained/lsknet_s_backbone.pth`.

Dataset: [DIOR-R](https://gcheng-nwpu.github.io/), placed under `data/dior/`
in standard `JPEGImages/` + `Annotations/obb/` + `ImageSets/Main/` layout.

## Training

```bash
python tools/train.py configs/obb/aopg/aopg_lsknet_s_fpn_1x_dior.py \\
  --work-dir work_dirs/aopg_lsknet_s_dior --no-validate --gpus 1
```

## Evaluation

```bash
python tools/test.py configs/obb/aopg/aopg_lsknet_s_fpn_1x_dior.py \\
  work_dirs/aopg_lsknet_s_dior/epoch_12.pth --eval mAP
```

## Credits

Built on [AOPG](https://github.com/jbwang1997/AOPG) and
[OBBDetection](https://github.com/jbwang1997/OBBDetection) (Wang et al.).
Backbone architecture from [LSKNet](https://github.com/zcablii/LSKNet)
(Li et al.).
