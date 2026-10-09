import os, csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class TokenOnlyDataset(Dataset):
    """
    无标签：仅提供观测 RGB token。
    目录结构：
      root/
        images/xxx.png (或 jpg/jpeg/bmp/tif)
    说明：
      - 将图像 resize 到 (H,W)，作为色卡网格；
      - 观测 tokenRGB = [H*W, 3]，来自该网格的像素（归一化到[0,1]）。
    """
    def __init__(self, root_dir: str, token_grid=(4,6)):
        super().__init__()
        self.root = root_dir
        self.H, self.W = token_grid
        img_dir = os.path.join(root_dir, "images")
        assert os.path.isdir(img_dir), f"images/ not found in {root_dir}"
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(exts)])
        assert len(self.files) > 0, f"No images found in {img_dir}"

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.W, self.H), Image.BILINEAR)  # 与 token_grid 对齐
        arr = np.asarray(img).astype(np.float32) / 255.0    # [H,W,3]
        tokens = arr.reshape(self.H*self.W, 3)              # [T,3]
        arr = np.transpose(arr, (2,0,1))                    # [3,H,W] 给模型用
        return torch.from_numpy(arr), torch.from_numpy(tokens)  # (img[3,H,W], tokens[T,3])

def get_token_loaders(train_root: str, val_root: str,
                      batch_size: int = 64, num_workers: int = 0,
                      token_grid=(4,6)):
    ds_tr = TokenOnlyDataset(train_root, token_grid=token_grid)
    ds_va = TokenOnlyDataset(val_root,   token_grid=token_grid)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=False)
    return dl_tr, dl_va
