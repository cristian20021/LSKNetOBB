"""
Visualize LSKNet backbone feature maps on a real image.

Usage:
    python tools/visualize_lsknet.py \
        --img demo/dota_demo.png \
        --config configs/obb/aopg/aopg_lsknet_s_fpn_1x_dior.py \
        --checkpoint work_dirs/aopg_lsknet_s_dior/epoch_1.pth \
        --out-dir work_dirs/lsknet_viz

If --checkpoint is omitted, uses the ImageNet-pretrained backbone weights
straight from the config's `pretrained` field instead of a trained detector.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmcv import Config
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--config', required=True, help='Model config file')
    parser.add_argument('--checkpoint', default=None,
                         help='Trained detector checkpoint (optional)')
    parser.add_argument('--out-dir', default='work_dirs/lsknet_viz',
                         help='Where to save visualizations')
    parser.add_argument('--img-size', type=int, default=800,
                         help='Resize input to this size (square)')
    return parser.parse_args()


def load_model(cfg_path, checkpoint_path):
    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if checkpoint_path is not None:
        from mmcv.runner import load_checkpoint
        print(f'Loading trained checkpoint: {checkpoint_path}')
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    else:
        print('No checkpoint given — using pretrained backbone weights only')
        model.init_weights()

    model.eval()
    return model


def preprocess_image(img_path, size):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img.copy()

    img = cv2.resize(img, (size, size))
    img_norm = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)

    return img_tensor, img, orig


def feature_map_to_heatmap(feat, target_size):
    # feat: (C, H, W) tensor for one stage
    heatmap = feat.mean(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # relu-like, keep positive activations
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.5):
    return (img.astype(np.float32) * (1 - alpha) +
            heatmap.astype(np.float32) * alpha).astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.config, args.checkpoint)
    img_tensor, img_resized, orig = preprocess_image(args.img, args.img_size)

    with torch.no_grad():
        feats = model.backbone(img_tensor)  # tuple of 4 stage outputs

    print(f'Got {len(feats)} stage feature maps:')
    for i, f in enumerate(feats):
        print(f'  Stage {i+1}: shape {tuple(f.shape)}')

    h, w = img_resized.shape[:2]
    fig, axes = plt.subplots(2, len(feats) + 1, figsize=(4 * (len(feats) + 1), 8))

    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title('Input image')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img_resized)
    axes[1, 0].set_title('Input image')
    axes[1, 0].axis('off')

    for i, feat in enumerate(feats):
        feat_single = feat[0]  # remove batch dim -> (C, H, W)
        heatmap = feature_map_to_heatmap(feat_single, (w, h))
        overlay = overlay_heatmap(img_resized, heatmap)

        axes[0, i + 1].imshow(heatmap)
        axes[0, i + 1].set_title(f'Stage {i+1} activation\n{tuple(feat_single.shape)}')
        axes[0, i + 1].axis('off')

        axes[1, i + 1].imshow(overlay)
        axes[1, i + 1].set_title(f'Stage {i+1} overlay')
        axes[1, i + 1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, 'lsknet_feature_maps.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved visualization to: {out_path}')


if __name__ == '__main__':
    main()
