import os
import numpy as np
from PIL import Image
import torch.utils.data as data

def gc_cmap(N=256, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    cmap[0] = np.array([0, 0, 0], dtype=dtype)       # background
    cmap[1] = np.array([255, 255, 255], dtype=dtype) # foreground
    if normalized:
        cmap = cmap / 255.0
    return cmap

class GrabCutSegmentationFlatDir(data.Dataset):
    """
    扁平目录版 GrabCut：
      data_dir/
        xxx.jpg / .JPG / .jpeg
        xxx.png   (mask, 前景>0, 背景=0)

    可选 split_txt：里面每行一个文件主名（不含后缀），用于划分 train/val。
    """
    cmap = gc_cmap()

    def __init__(self,
                 root,                 # 例如 '/root/code/DeepLabV3Plus-Pytorch'
                 base_dir='GrabCut',   # 例如 'GrabCut'
                 transform=None,
                 split_txt=None):
        self.root = os.path.expanduser(root)
        self.data_dir = os.path.join(self.root, base_dir)
        self.transform = transform

        if not os.path.isdir(self.data_dir):
            raise RuntimeError(f'Dataset directory not found: {self.data_dir}')

        # 如果给了 split_txt，就按里面的主名筛选
        allow_names = None
        if split_txt is not None:
            if not os.path.exists(split_txt):
                raise ValueError(f'split file not found: {split_txt}')
            with open(split_txt, 'r') as f:
                allow_names = set([ln.strip() for ln in f if ln.strip()])

        exts_img = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        img_candidates = {}
        for fn in os.listdir(self.data_dir):
            if fn.startswith('.'):  # 忽略 .DS_Store 等隐藏文件
                continue
            stem, ext = os.path.splitext(fn)
            if ext in exts_img and ext.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                # 这里把 png 也当作可能的图像（以防有人提供 png 图像）
                img_candidates.setdefault(stem, []).append(fn)

        images, masks = [], []
        for stem, files in img_candidates.items():
            # mask 固定找 png（兼容 .png/.PNG）
            mask_path_lower = os.path.join(self.data_dir, stem + '.png')
            mask_path_upper = os.path.join(self.data_dir, stem + '.PNG')
            mask_path = mask_path_lower if os.path.exists(mask_path_lower) else \
                        (mask_path_upper if os.path.exists(mask_path_upper) else None)
            if mask_path is None:
                continue

            # 选一个作为图像文件（优先 jpg/jpeg/JPG/JPEG，再退回 png/bmp）
            chosen_img = None
            prefer_order = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
            have = {os.path.splitext(f)[1]: f for f in files}
            for ext in prefer_order:
                if ext in have:
                    chosen_img = have[ext]
                    break
            if chosen_img is None:
                continue

            if allow_names is not None and stem not in allow_names:
                continue

            images.append(os.path.join(self.data_dir, chosen_img))
            masks.append(mask_path)

        if len(images) == 0:
            raise RuntimeError(f'No image-mask pairs found in {self.data_dir}')

        # 排序以保证确定性
        pairs = sorted(zip(images, masks), key=lambda x: os.path.basename(x[0]).lower())
        self.images, self.masks = zip(*pairs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        # 读 mask，统一为 {0,1}
        m = Image.open(self.masks[index])
        m_np = np.array(m)
        # 有些 mask 可能是三通道或 0/255，这里一律非零为前景
        if m_np.ndim == 3:
            m_np = m_np[..., 0]
        m_np = (m_np > 0).astype(np.uint8)
        target = Image.fromarray(m_np, mode='L')

        if self.transform is not None:
            img, target = self.transform(img, target)

        # 图像级标签：[1(背景恒为1), fg_present]
        fg_present = 1 if (m_np.sum() > 0) else 0
        label_cls = np.array([1, fg_present], dtype=np.int64)

        return img, target, label_cls

    @classmethod
    def decode_target(cls, mask):
        mask = np.asarray(mask).astype(np.int64)
        mask = np.clip(mask, 0, 1)
        return cls.cmap[mask]

if __name__ == '__main__':
    ds = GrabCutSegmentationFlatDir(
        root='/root/code/DeepLabV3Plus-Pytorch',
        base_dir='GrabCut',
        transform=None,      # 放你的 transform(img, mask)
        split_txt=None       # 如果需要划分，可给个 txt
    )

    print(len(ds), ds.images[0], ds.masks[0])
