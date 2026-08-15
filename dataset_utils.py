# dataset_utils.py
# Keeping Dataset class in a separate .py file (not inside notebook cells)
# fixes the Windows multiprocessing hang issue when num_workers > 0

import os
from PIL import Image
from torch.utils.data import Dataset

class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label