# File: handlers/evaluation_data_loader.py
# Author: Alfrida Sabar
# Deskripsi: DataLoader khusus untuk evaluasi model dengan support berbagai kondisi pengujian

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

class EvaluationDataset(Dataset):
    """Dataset untuk evaluasi model dengan support berbagai kondisi pengujian"""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        conditions: str = 'position',  # 'position' atau 'lighting'
        transform=None
    ):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.conditions = conditions
        self.transform = transform
        
        # Load semua file gambar
        self.image_files = list(self.data_path.glob('*.jpg'))
        self.label_files = [
            self.data_path.parent / 'labels' / f"{img.stem}.txt"
            for img in self.image_files
        ]
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load gambar
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing sesuai kondisi
        if self.conditions == 'lighting':
            img = self._adjust_lighting(img)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalisasi
        img = img.astype(np.float32) / 255.0
        
        # Load labels
        label_path = self.label_files[idx]
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    labels.append([float(x) for x in line.strip().split()])
                    
        # Convert ke tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        label_tensor = torch.tensor(labels) if labels else torch.zeros((0, 5))
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor, label_tensor
        
    def _adjust_lighting(self, img: np.ndarray) -> np.ndarray:
        """Adjust lighting untuk skenario pengujian pencahayaan"""
        if np.random.random() < 0.5:
            # Simulasi low light
            alpha = np.random.uniform(0.5, 0.8)
            beta = np.random.uniform(-50, -20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        else:
            # Simulasi bright light
            alpha = np.random.uniform(1.2, 1.5)
            beta = np.random.uniform(20, 50)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
        return img

def create_evaluation_dataloader(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    conditions: str = 'position'
) -> DataLoader:
    """
    Buat DataLoader untuk evaluasi
    Args:
        data_path: Path ke direktori dataset
        batch_size: Ukuran batch
        num_workers: Jumlah worker untuk data loading
        conditions: Kondisi pengujian ('position' atau 'lighting')
    Returns:
        DataLoader untuk evaluasi
    """
    dataset = EvaluationDataset(
        data_path=data_path,
        conditions=conditions
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )