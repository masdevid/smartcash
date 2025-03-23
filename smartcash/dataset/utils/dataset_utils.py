"""
File: smartcash/dataset/utils/dataset_utils.py
Deskripsi: Utilitas umum untuk operasi dataset
"""

import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.layer_config_interface import ILayerConfigManager
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_constants import (
    IMG_EXTENSIONS, DEFAULT_SPLITS, DEFAULT_SPLIT_RATIOS, DEFAULT_RANDOM_SEED
)

class DatasetUtils:
    """Utilitas umum untuk operasi dataset."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None, layer_config: Optional[ILayerConfigManager] = None):
        """
        Inisialisasi DatasetUtils.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            layer_config: ILayerConfigManager instance (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_utils")
        
        # Setup layer config
        if config:
            self.layer_config = layer_config or get_layer_config()
            self.active_layers = config.get('layers', self.layer_config.get_layer_names())
            
            # Buat mapping class ID ke layer/nama
            self.class_to_layer, self.class_to_name = {}, {}
            
            for layer_name in self.layer_config.get_layer_names():
                layer_config = self.layer_config.get_layer_config(layer_name)
                for i, cls_id in enumerate(layer_config['class_ids']):
                    self.class_to_layer[cls_id] = layer_name
                    if i < len(layer_config['classes']):
                        self.class_to_name[cls_id] = layer_config['classes'][i]
    
    def get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Path ke direktori split
        """
        # Normalisasi nama split
        split = 'valid' if split in ('val', 'validation') else split
        
        # Cek config atau gunakan default
        split_paths = self.config.get('data', {}).get('local', {})
        if split in split_paths:
            return Path(split_paths[split])
            
        # Fallback ke path default
        return self.data_dir / split
    
    def get_class_name(self, cls_id: int) -> str:
        """
        Dapatkan nama kelas dari ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama kelas
        """
        if not hasattr(self, 'class_to_name'):
            return f"Class-{cls_id}"
            
        if cls_id in self.class_to_name:
            return self.class_to_name[cls_id]
            
        # Cari di layer config
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                idx = layer_config['class_ids'].index(cls_id)
                if idx < len(layer_config['classes']):
                    # Tambahkan ke cache untuk akses lebih cepat
                    self.class_to_name[cls_id] = layer_config['classes'][idx]
                    return layer_config['classes'][idx]
                    
        return f"Class-{cls_id}"
    
    def get_layer_from_class(self, cls_id: int) -> Optional[str]:
        """
        Dapatkan nama layer dari ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama layer atau None jika tidak ditemukan
        """
        if not hasattr(self, 'class_to_layer'):
            return None
            
        if cls_id in self.class_to_layer:
            return self.class_to_layer[cls_id]
            
        # Cari di layer config
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                # Tambahkan ke cache untuk akses lebih cepat
                self.class_to_layer[cls_id] = layer_name
                return layer_name
                
        return None
    
    def find_image_files(self, directory: Union[str, Path], with_labels: bool = True) -> List[Path]:
        """
        Cari file gambar dalam direktori.
        
        Args:
            directory: Direktori yang akan dicari
            with_labels: Apakah hanya file dengan label yang dikembalikan
            
        Returns:
            Daftar path file gambar
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.warning(f"⚠️ Direktori tidak ditemukan: {dir_path}")
            return []
        
        # Cari semua file gambar
        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(list(dir_path.glob(ext)))
        
        if not with_labels:
            return image_files
        
        # Tentukan direktori label
        labels_dir = None
        if (dir_path / 'labels').exists():
            labels_dir = dir_path / 'labels'
        elif 'images' in dir_path.parts and (dir_path.parent / 'labels').exists():
            labels_dir = dir_path.parent / 'labels'
        
        if not labels_dir:
            return image_files
            
        # Filter hanya file dengan label
        return [img for img in image_files if (labels_dir / f"{img.stem}.txt").exists()]
    
    def get_random_sample(self, items: List, sample_size: int, seed: int = DEFAULT_RANDOM_SEED) -> List:
        """
        Ambil sampel acak dari list.
        
        Args:
            items: List item yang akan diambil sampelnya
            sample_size: Jumlah sampel yang diinginkan
            seed: Seed untuk random
            
        Returns:
            List sampel
        """
        if sample_size <= 0 or sample_size >= len(items):
            return items
            
        random.seed(seed)
        return random.sample(items, sample_size)
    
    def load_image(self, image_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Baca gambar dari file.
        
        Args:
            image_path: Path ke file gambar
            target_size: Ukuran target (opsional)
            
        Returns:
            Array NumPy berisi gambar
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if target_size:
                img = cv2.resize(img, target_size)
                
            return img
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {image_path}: {str(e)}")
            
            # Return dummy image jika target_size disediakan
            height, width = target_size[1], target_size[0] if target_size else (640, 640)
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def parse_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        Parse file label YOLO.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List berisi data bounding box dan kelas
        """
        if not label_path.exists():
            return []
        
        bboxes = []
        try:
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:5])
                            
                            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0.001 and h > 0.001):
                                continue
                            
                            bbox_data = {'class_id': cls_id, 'bbox': [x, y, w, h], 'line_idx': i}
                            
                            if hasattr(self, 'class_to_name'):
                                bbox_data.update({
                                    'class_name': self.get_class_name(cls_id),
                                    'layer': self.get_layer_from_class(cls_id) or 'unknown'
                                })
                            
                            bboxes.append(bbox_data)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.debug(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return bboxes
    
    def get_available_layers(self, label_path: Path) -> List[str]:
        """
        Dapatkan layer yang tersedia dalam file label.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List layer yang tersedia
        """
        if not hasattr(self, 'active_layers'):
            return []
        
        available_layers = set()
        for bbox in self.parse_yolo_label(label_path):
            if 'layer' in bbox and bbox['layer'] != 'unknown' and bbox['layer'] in self.active_layers:
                available_layers.add(bbox['layer'])
        
        return list(available_layers)
    
    def get_split_statistics(self, splits: List[str] = DEFAULT_SPLITS) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dasar untuk semua split dataset.
        
        Args:
            splits: List split yang akan diperiksa
            
        Returns:
            Dictionary berisi statistik per split
        """
        stats = {}
        
        for split in splits:
            split_path = self.get_split_path(split)
            images_dir, labels_dir = split_path / 'images', split_path / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                stats[split] = {'images': 0, 'labels': 0, 'status': 'missing'}
                continue
            
            # Hitung file
            image_count = sum(len(list(images_dir.glob(ext))) for ext in IMG_EXTENSIONS)
            label_count = len(list(labels_dir.glob('*.txt')))
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'status': 'valid' if image_count > 0 and label_count > 0 else 'empty'
            }
        
        self.logger.info(
            f"📊 Statistik dataset:\n"
            f"   • Train: {stats.get('train', {}).get('images', 0)} gambar, {stats.get('train', {}).get('labels', 0)} label\n"
            f"   • Valid: {stats.get('valid', {}).get('images', 0)} gambar, {stats.get('valid', {}).get('labels', 0)} label\n"
            f"   • Test: {stats.get('test', {}).get('images', 0)} gambar, {stats.get('test', {}).get('labels', 0)} label"
        )
        
        return stats
    
    def backup_directory(self, source_dir: Union[str, Path], suffix: Optional[str] = None) -> Optional[Path]:
        """
        Buat backup direktori.
        
        Args:
            source_dir: Direktori yang akan di-backup
            suffix: Suffix untuk nama direktori backup (opsional)
            
        Returns:
            Path ke direktori backup atau None jika gagal
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            self.logger.warning(f"⚠️ Direktori sumber tidak ditemukan: {source_path}")
            return None
        
        suffix = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = source_path.parent
        backup_path = parent_dir / f"{source_path.name}_backup_{suffix}"
        
        # Terapkan penomoran jika path sudah ada
        i = 1
        while backup_path.exists():
            backup_path = parent_dir / f"{source_path.name}_backup_{suffix}_{i}"
            i += 1
        
        try:
            shutil.copytree(source_path, backup_path)
            self.logger.success(f"✅ Direktori berhasil dibackup ke: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup {source_path}: {str(e)}")
            return None
    
    def move_invalid_files(self, source_dir: Union[str, Path], target_dir: Union[str, Path], 
                         file_list: List[Path]) -> Dict[str, int]:
        """
        Pindahkan file ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: Daftar file yang akan dipindahkan
            
        Returns:
            Statistik pemindahan
        """
        stats = {'moved': 0, 'skipped': 0, 'errors': 0}
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_list:
            try:
                rel_path = file_path.relative_to(source_dir)
                dest_path = target_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    stats['moved'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                self.logger.error(f"❌ Gagal memindahkan {file_path}: {str(e)}")
                stats['errors'] += 1
        
        return stats