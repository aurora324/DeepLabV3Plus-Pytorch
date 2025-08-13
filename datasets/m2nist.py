import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.tv_tensors import Image, Mask

# ----------------------------
# 1) Dataset
# ----------------------------
class M2NISTSeg(Dataset):
    def __init__(self, combined_path, segmented_path, transform=None):
        self.images = np.load(combined_path)      # (N,64,84)
        self.masks  = np.load(segmented_path)     # (N,64,84,11)
        assert self.images.shape[0] == self.masks.shape[0]
        self.bg_idx = self.masks.shape[-1] - 1
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        # -> float32 tensor in [0,1], shape (1,64,84)
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        msk = self.masks[idx]                         # (64,84,11)

        msk = torch.from_numpy(msk).permute(2,0,1).to(torch.uint8)  # (11,64,84)
        # msk: torch.uint8 或 bool，形状 (C,H,W)
        label_cls = msk.any(dim=(1, 2))           # (C,)
        # 如果需要 0/1 的 uint8：
        label_cls = label_cls.to(torch.uint8)     # (C,)
        msk = Mask(msk)
        if self.transform is not None:
            img, msk = self.transform(img, msk)
        msk = msk.argmax(dim=0).long()

        return img, msk, label_cls
    PALETTE = np.array([
        [  0,   0,   0],   # 0 background
        [128,   0,   0],   # 1
        [  0, 128,   0],   # 2
        [128, 128,   0],   # 3
        [  0,   0, 128],   # 4
        [128,   0, 128],   # 5
        [  0, 128, 128],   # 6
        [128, 128, 128],   # 7
        [ 64,   0,   0],   # 8
        [192,   0,   0],   # 9
        [ 64, 128,   0],   # 10
    ], dtype=np.uint8)

    NUM_CLASSES = 11
    IGNORE_INDEX = 255

    @classmethod
    def encode_target(cls, target):
        if isinstance(target, torch.Tensor):
            t = target
            if t.ndim == 3:
                # one-hot -> 索引图
                if t.shape[0] == cls.NUM_CLASSES:   # (C,H,W)
                    t = t.argmax(dim=0)
                elif t.shape[-1] == cls.NUM_CLASSES: # (H,W,C)
                    t = t.argmax(dim=-1)
            return t.to(torch.long)

        # numpy
        m = np.asarray(target)
        if m.ndim == 3:
            if m.shape[0] == cls.NUM_CLASSES:      # (C,H,W)
                m = m.argmax(axis=0)
            elif m.shape[-1] == cls.NUM_CLASSES:   # (H,W,C)
                m = m.argmax(axis=-1)
        return m.astype(np.int64, copy=False)

    @classmethod
    def decode_target(cls, target, to_tensor: bool = False):
        # 先得到 (H,W) 索引图
        t = cls.encode_target(target)

        if isinstance(t, torch.Tensor):
            m = t.detach().cpu().numpy()
        else:
            m = t

        if cls.IGNORE_INDEX is not None:
            if cls.PALETTE.shape[0] == cls.NUM_CLASSES:
                palette = np.vstack([cls.PALETTE, np.array([[0, 0, 0]], dtype=np.uint8)])
                ignore_row = cls.NUM_CLASSES
            else:
                palette = cls.PALETTE
                ignore_row = cls.PALETTE.shape[0] - 1
            m_vis = m.copy()
            m_vis[m_vis == cls.IGNORE_INDEX] = ignore_row
        else:
            palette = cls.PALETTE
            m_vis = m

        color = palette[m_vis]

        if to_tensor:
            return torch.from_numpy(color).permute(2, 0, 1).float() / 255.0
        return color

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgb
    from pathlib import Path
    import torch

    CLASS_COLORS = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                    'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    def show_one(img_t: torch.Tensor, mask_t: torch.Tensor, normalize=True,
                alpha=0.35, save_overlay=None, save_masks=None):
        Path("vis").mkdir(exist_ok=True)

        base = img_t.detach().cpu()
        if normalize:
            base = base * 0.5 + 0.5           # 反归一化到[0,1]
        base = base.squeeze(0).numpy()        # (64,84)

        # ---- 叠掩码图 ----
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.imshow(base, cmap="gray", interpolation="nearest")
        ax.axis("off")

        present = []
        C = mask_t.shape[0]
        for d in range(C - 1):
            m = mask_t[d].bool().detach().cpu().numpy()
            if m.any():
                present.append(d)
                color = to_rgb(CLASS_COLORS[d])
                cmap = ListedColormap([(0,0,0,0), (color[0], color[1], color[2], alpha)])
                ax.imshow(m.astype(float), cmap=cmap, interpolation="nearest")
                # 外接框（可选）
                ys, xs = np.where(m)
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                ax.add_patch(plt.Rectangle((x0, y0), x1-x0+1, y1-y0+1, fill=False, linewidth=1.2))
        ax.set_title(f"digits: {present}", fontsize=9)
        plt.tight_layout()
        if save_overlay:
            plt.savefig(save_overlay, dpi=200, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for d in range(10):
            ax = axes[d // 5, d % 5]
            m = mask_t[d].detach().cpu().numpy()
            ax.imshow(m, cmap="gray", interpolation="nearest")
            ax.set_title(str(d))
            ax.axis("off")
        plt.tight_layout()
        if save_masks:
            plt.savefig(save_masks, dpi=200, bbox_inches="tight")
        plt.show()

    dataset = M2NISTSeg("data/combined.npy", "data/segmented.npy", transform=None)
    N = len(dataset)
    n_train = int(0.9 * N)
    n_val = N - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    imgs, msks = next(iter(train_loader))
    print("Batch shapes:", imgs.shape, msks.shape)  # (B,1,64,84), (B,11,64,84)

    rand_index = np.random.randint(0, len(train_set))
    c, s, label_cls = train_set[rand_index]  # c:(1,64,84), s:(11,64,84)

    # 1) 仅灰度
    plt.figure(figsize=(4.2,4))
    plt.imshow((c*0.5+0.5).squeeze(0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("vis/sample_gray.png", dpi=200, bbox_inches="tight")
    plt.show()

    # 2) 叠掩码 + 每类子图
    show_one(c, s, normalize=True,
            save_overlay="vis/sample_overlay.png",
            save_masks="vis/sample_masks.png")
