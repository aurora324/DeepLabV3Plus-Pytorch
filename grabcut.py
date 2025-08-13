# tools/test_grabcut_dataloader.py
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

# 你的扁平目录版数据集（上一条我给你的）
from datasets.GrabCut import GrabCutSegmentationFlatDir

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class JointResizeToTensorNormalize:
    """
    同时对 image/mask 做变换：
      - 可选 resize
      - image: ToTensor + Normalize
      - mask: 转为 {0,1} 的 long tensor
    """
    def __init__(self, size=None):
        self.size = size

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.size is not None:
            # 图像双线性，mask 最近邻
            img  = img.resize(self.size, resample=Image.BILINEAR)
            mask = mask.resize(self.size, resample=Image.NEAREST)

        # to tensor
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = (mask_np > 0).astype(np.uint8)
        mask_t = torch.from_numpy(mask_np).long()  # [H, W] in {0,1}

        return img_t, mask_t

def denormalize(img_t):
    # img_t: [C,H,W] (normalized)
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_t.dtype, device=img_t.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  dtype=img_t.dtype, device=img_t.device)[:, None, None]
    return img_t * std + mean

def overlay_mask(img_t, mask_t, alpha=0.45):
    """
    简单叠加：把前景涂成红色叠在原图上
    img_t: [3,H,W], 0-1 范围（已反归一化）
    mask_t: [H,W] {0,1}
    """
    img_np = (img_t.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    h, w = mask_t.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    color[..., 0] = 255  # 红
    m = mask_t.cpu().numpy().astype(bool)[..., None]
    blended = img_np.copy()
    blended[m] = (alpha * color[m] + (1 - alpha) * img_np[m]).astype(np.uint8)
    return Image.fromarray(blended)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/code/DeepLabV3Plus-Pytorch', type=str)
    parser.add_argument('--base_dir', default='GrabCut', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resize', default=512, type=int, help='短边或固定边，传 0 表示不缩放')
    parser.add_argument('--save_vis', action='store_true', help='是否保存可视化到 out_vis/')
    args = parser.parse_args()

    size = None if args.resize == 0 else (args.resize, args.resize)
    transform = JointResizeToTensorNormalize(size=size)

    dataset = GrabCutSegmentationFlatDir(
        root=args.root,
        base_dir=args.base_dir,
        transform=transform,
        split_txt=None  # 若有划分文件就填路径
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f'Found {len(dataset)} samples.')
    print('First few pairs:')
    for i in range(min(5, len(dataset))):
        print(f'  {os.path.basename(dataset.images[i])}  <->  {os.path.basename(dataset.masks[i])}')

    if args.save_vis:
        os.makedirs('out_vis', exist_ok=True)

    # 简单遍历一次，检查 shape/类型，统计前景占比，并可选保存可视化
    foreground_ratio_all = []
    with torch.no_grad():
        for step, (imgs, masks, label_cls) in enumerate(loader):
            # imgs: [B,3,H,W] float; masks: [B,H,W] long {0,1}; label_cls: [B,2]
            B, C, H, W = imgs.shape
            print(f'[Batch {step}] imgs={imgs.shape}, masks={masks.shape}, label_cls={label_cls.shape}')

            # 统计前景比例
            fg_count = masks.sum(dim=(1,2)).float()  # [B]
            total = torch.tensor(H*W, dtype=torch.float32)  # 单值
            ratio = (fg_count / total).cpu().numpy().tolist()
            foreground_ratio_all.extend(ratio)
            print(f'  foreground ratio (per image): {["{:.3f}".format(r) for r in ratio]}')

            if args.save_vis:
                # 保存每个 batch 的前 2 张可视化
                save_n = min(2, B)
                imgs_denorm = denormalize(imgs[:save_n].cpu())
                for k in range(save_n):
                    vis = overlay_mask(imgs_denorm[k], masks[k].cpu())
                    stem = os.path.splitext(os.path.basename(dataset.images[step*args.batch_size + k]))[0]
                    vis.save(os.path.join('out_vis', f'{stem}_overlay.jpg'))

    if len(foreground_ratio_all) > 0:
        arr = np.array(foreground_ratio_all)
        print(f'==== Summary ====')
        print(f'Avg foreground ratio: {arr.mean():.4f}  |  Min: {arr.min():.4f}  Max: {arr.max():.4f}')
        if args.save_vis:
            print('Saved overlays to: out_vis/')

if __name__ == '__main__':
    main()
