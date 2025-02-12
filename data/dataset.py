# File: src/data/dataset.py
# Author: Alfrida Sabar
# Deskripsi: Dataset loader dengan konversi label untuk SmartCash Detector

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, List, Dict
import multiprocessing as mp
from utils.logging import ColoredLogger
from config.labels import LabelConfig

class RupiahDataset(Dataset):
    def __init__(self, 
                 img_dir: Path,
                 img_size: int = 640,
                 augment: bool = True,
                 cache: bool = True):
        """
        Inisialisasi dataset Rupiah dengan konversi label otomatis
        
        Args:
            img_dir: Path direktori gambar
            img_size: Ukuran gambar target
            augment: Flag untuk augmentasi data
            cache: Flag untuk caching gambar
        """
        self.logger = ColoredLogger('Dataset')
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.cache = {} if cache else None
        self.label_config = LabelConfig()
        
        # Scan dan verifikasi file
        self.img_files = sorted(self.img_dir.glob('*.jpg'))
        self.label_dir = self.img_dir.parent.parent / 'labels'
        self._verify_files()
        
        # Setup augmentasi
        self.transform = self._get_transforms() if augment else None
        
        # Preload dan konversi label
        self.logger.info("📝 Memuat dan mengkonversi label...")
        self.labels = self._preload_labels()
        self.logger.info(f"✨ Dataset siap dengan {len(self.img_files)} gambar")

    def _verify_files(self):
        """Verifikasi keberadaan file gambar dan label"""
        valid_files = []
        for img_file in self.img_files:
            label_file = self.label_dir / f'{img_file.stem}.txt'
            if label_file.exists():
                valid_files.append(img_file)
            else:
                self.logger.warning(f"⚠️ Label tidak ditemukan: {label_file}")
        self.img_files = valid_files

    def _preload_labels(self) -> List[np.ndarray]:
        """
        Muat dan konversi label ke format baru
        Returns:
            List label yang telah dikonversi
        """
        labels = []
        for img_file in self.img_files:
            label_file = self.label_dir / f'{img_file.stem}.txt'
            if label_file.exists():
                # Baca label dari file
                with open(label_file) as f:
                    rows = []
                    for line in f.readlines():
                        values = line.strip().split()
                        if len(values) == 5:  # Format YOLO: class, x, y, w, h
                            # Konversi indeks kelas
                            old_cls = int(values[0])
                            old_label = ['100k', '10k', '1k', '20k', '2k', '50k', '5k'][old_cls]
                            new_label = self.label_config.convert_label(old_label)
                            new_cls = self.label_config.get_label_idx(new_label)
                            
                            # Gabungkan dengan koordinat bounding box
                            bbox = list(map(float, values[1:]))
                            rows.append([new_cls] + bbox)
                            
                label = np.array(rows)
            else:
                label = np.zeros((0, 5))
            labels.append(label)
        return labels

    def _get_transforms(self) -> A.Compose:
        """Setup transformasi augmentasi data"""
        return A.Compose([
            # Variasi pencahayaan
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
                A.CLAHE(p=0.8),
            ], p=0.5),
            
            # Noise dan blur
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.ISONoise(p=0.8),
                A.MotionBlur(p=0.8),
            ], p=0.3),
            
            # Transformasi geometrik
            A.OneOf([
                A.SafeRotate(limit=20, p=0.8),
                A.RandomScale(scale_limit=0.2, p=0.8),
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.8, 1.0),
                    p=0.8
                ),
            ], p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def _load_image(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Muat dan praproses gambar
        
        Args:
            idx: Indeks gambar
            
        Returns:
            Tuple (gambar, rasio scaling)
        """
        if self.cache and idx in self.cache:
            img = self.cache[idx]
        else:
            img = cv2.imread(str(self.img_files[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.cache is not None:
                self.cache[idx] = img

        # Resize dengan aspect ratio yang sama
        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)

        # Padding ke ukuran target
        new_h, new_w = [self.img_size] * 2
        img = cv2.copyMakeBorder(
            img, 0, new_h - img.shape[0], 0, new_w - img.shape[1],
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        return img, r

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ambil item dataset dengan indeks tertentu
        
        Args:
            idx: Indeks item
            
        Returns:
            Tuple (gambar, label) dalam format tensor
        """
        img, r = self._load_image(idx)
        labels = self.labels[idx].copy()
        
        if len(labels):
            labels[:, 1:] = labels[:, 1:] * r  # Skala koordinat bbox
        
        # Terapkan augmentasi
        if self.transform and len(labels):
            transformed = self.transform(
                image=img,
                bboxes=labels[:, 1:],
                class_labels=labels[:, 0]
            )
            img = transformed['image']
            if len(transformed['bboxes']):
                labels = np.column_stack([
                    transformed['class_labels'],
                    transformed['bboxes']
                ])

        # Konversi ke tensor
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
    """
    Buat DataLoader untuk dataset Rupiah
    
    Args:
        path: Path dataset
        img_size: Ukuran gambar target
        batch_size: Ukuran batch
        augment: Flag untuk augmentasi
        cache: Flag untuk caching
        workers: Jumlah worker untuk loading
        
    Returns:
        DataLoader yang dikonfigurasi
    """
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
    """
    Fungsi collate kustom untuk batch dengan jumlah label bervariasi
    
    Args:
        batch: List tuple (gambar, label)
        
    Returns:
        Tuple tensor (gambar, label) yang telah di-pad
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    
    # Pad label ke panjang maksimum dalam batch
    max_labels = max(label.shape[0] for label in labels)
    padded_labels = torch.zeros((len(labels), max_labels, 5))
    
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            padded_labels[i, :label.shape[0]] = label
            
    return imgs, padded_labels