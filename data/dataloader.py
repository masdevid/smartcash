# File: src/data/dataloader.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline data loading dan augmentasi khusus untuk deteksi mata uang Rupiah

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from pathlib import Path
import cv2
import numpy as np
from utils.logging import ColoredLogger
from typing import Tuple, List
import multiprocessing as mp

class RupiahDataset(Dataset):
    def __init__(self, 
                 img_dir: Path,
                 img_size: int = 640,
                 augment: bool = True,
                 cache: bool = True):
        self.logger = ColoredLogger('Dataset')
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.cache = {} if cache else None
        
        # Scan files
        self.img_files = sorted(self.img_dir.glob('*.jpg'))
        self.label_dir = self.img_dir.parent.parent / 'labels'
        self._verify_files()
        
        # Setup currency-specific augmentations
        self.transform = self._get_currency_transforms() if augment else None
        
        # Preload labels
        self.labels = self._preload_labels()
        self.logger.info(f"✨ Dataset initialized with {len(self.img_files)} images")

    def _verify_files(self):
        valid_files = []
        for img_file in self.img_files:
            label_file = self.label_dir / f'{img_file.stem}.txt'
            if label_file.exists():
                valid_files.append(img_file)
        self.img_files = valid_files

    def _get_currency_transforms(self) -> A.Compose:
        return A.Compose([
            # Lighting variations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.CLAHE(clip_limit=4.0, p=0.8),
                A.RandomShadow(p=0.8),
            ], p=0.5),
            
            # Orientation variations
            A.OneOf([
                A.Rotate(limit=30, p=0.8),
                A.SafeRotate(limit=30, p=0.8),
                A.RandomRotate90(p=0.8),
            ], p=0.5),
            
            # Banknote conditions
            A.OneOf([
                A.GaussianBlur(p=0.8),  # Worn banknotes
                A.ImageCompression(quality_lower=60, p=0.8),  # Camera quality
                A.GaussNoise(p=0.8),  # Low light conditions
            ], p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def _preload_labels(self) -> List[np.ndarray]:
        labels = []
        for img_file in self.img_files:
            label_file = self.label_dir / f'{img_file.stem}.txt'
            if label_file.exists():
                with open(label_file) as f:
                    label = np.array([
                        list(map(float, line.strip().split()))
                        for line in f.readlines()
                    ])
            else:
                label = np.zeros((0, 5))
            labels.append(label)
        return labels

    def _load_image(self, idx: int) -> Tuple[np.ndarray, float]:
        if self.cache and idx in self.cache:
            img = self.cache[idx]
        else:
            img = cv2.imread(str(self.img_files[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.cache is not None:
                self.cache[idx] = img

        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)

        # Make divisible by stride 32
        new_h, new_w = [self.img_size] * 2
        img = cv2.copyMakeBorder(img, 0, new_h - img.shape[0], 0, 
                                new_w - img.shape[1], cv2.BORDER_CONSTANT, 
                                value=(114, 114, 114))
        return img, r

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, r = self._load_image(idx)
        labels = self.labels[idx].copy()
        
        if len(labels):
            labels[:, :4] = labels[:, :4] * r
        
        # Apply currency-specific augmentations
        if self.transform and len(labels):
            transformed = self.transform(
                image=img,
                bboxes=labels[:, :4],
                class_labels=labels[:, 4]
            )
            img = transformed['image']
            if len(transformed['bboxes']):
                labels = np.column_stack([
                    transformed['bboxes'],
                    transformed['class_labels']
                ])

        # Convert to tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        labels = torch.from_numpy(labels).float()
        
        return img, labels

    def __len__(self) -> int:
        return len(self.img_files)

def create_dataloader(
    path: Path,
    img_size: int = 640,
    batch_size: int = 16,
    augment: bool = True,
    cache: bool = True,
    workers: int = None
) -> DataLoader:
    """Create optimized dataloader for Rupiah detection"""
    if workers is None:
        workers = min(8, max(1, mp.cpu_count() - 1))
        
    dataset = RupiahDataset(
        img_dir=path,
        img_size=img_size,
        augment=augment,
        cache=cache
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for variable size labels"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    
    # Pad labels to same length
    max_labels = max(label.shape[0] for label in labels)
    padded_labels = torch.zeros((len(labels), max_labels, 5))
    
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            padded_labels[i, :label.shape[0]] = label
            
    return imgs, padded_labels