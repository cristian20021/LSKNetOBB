import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import BboxToolkit as bt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-id', required=True,
                         help='Image ID from DIOR test set, e.g. 23430 (no extension)')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--out-dir', default='work_dirs/pred_vs_gt')
    return parser.parse_args()


def draw_obboxes(img, obboxes, labels, class_names, color, thickness=2, show_label=True):
    img = img.copy()
    if len(obboxes) == 0:
        return img
    polys = bt.obb2poly(obboxes)  # (N, 8) -> x1,y1,x2,y2,x3,y3,x4,y4
    for poly, label in zip(polys, labels):
        pts = poly.reshape(4, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        if show_label:
            cls_name = class_names[label] if label < len(class_names) else str(label)
            cv2.putText(img, cls_name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    # Build the test dataset purely to fetch this image's ground truth
    test_cfg_data = cfg.data.test.copy()
    dataset = build_dataset(test_cfg_data)

    # Find the index for the requested image id
    target_idx = None
    for i, info in enumerate(dataset.data_infos):
        if info.get('id') == args.img_id or info.get('filename', '').startswith(args.img_id):
            target_idx = i
            break
    if target_idx is None:
        raise ValueError(f'Image id {args.img_id} not found in test set data_infos. '
                          f'Check the exact id format with a quick debug print.')

    ann = dataset.get_ann_info(target_idx)
    gt_obboxes = ann['bboxes']  # already in obb (cx,cy,w,h,theta) or poly depending on pipeline
    gt_labels = ann['labels']
    class_names = dataset.CLASSES

    img_path = os.path.join(cfg.data.test.img_prefix, dataset.data_infos[target_idx]['filename'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Build model and load trained checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cfg = cfg
    model.CLASSES = dataset.CLASSES
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

	
    from mmdet.apis import inference_detector
    result = inference_detector(model, img_path)

    # result: list of arrays per class, each (N, 6) -> cx,cy,w,h,theta,score (obb) or similar
    pred_boxes = []
    pred_labels = []
    for cls_idx, dets in enumerate(result):
        if len(dets) == 0:
            continue
        keep = dets[:, -1] >= args.score_thr
        dets = dets[keep]
        for d in dets:
            pred_boxes.append(d[:5])
            pred_labels.append(cls_idx)
    pred_boxes = np.array(pred_boxes) if pred_boxes else np.zeros((0, 5))
    pred_labels = np.array(pred_labels, dtype=np.int64)

    gt_img = draw_obboxes(img, gt_obboxes, gt_labels, class_names, color=(0, 255, 0))
    pred_img = draw_obboxes(img, pred_boxes, pred_labels, class_names, color=(255, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(gt_img)
    axes[0].set_title(f'Ground Truth ({len(gt_obboxes)} boxes)')
    axes[0].axis('off')
    axes[1].imshow(pred_img)
    axes[1].set_title(f'Predictions, score>{args.score_thr} ({len(pred_boxes)} boxes)')
    axes[1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f'{args.img_id}_pred_vs_gt.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to: {out_path}')


if __name__ == '__main__':
    main()
