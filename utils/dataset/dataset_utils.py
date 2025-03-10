# File: smartcash/utils/dataset/dataset_utils.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas umum untuk operasi dataset SmartCash

import os, cv2, shutil, random, numpy as np, collections
from tqdm.auto import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config

# Konstanta
IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
DEFAULT_SPLITS = ['train', 'valid', 'test']
DEFAULT_SPLIT_RATIOS = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
DEFAULT_RANDOM_SEED = 42

class DatasetUtils:
    """Utilitas umum untuk operasi dataset SmartCash."""

    def __init__(self, config: Optional[Dict] = None, data_dir: Optional[str] = None, logger=None):
        self.config = config or {}
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_utils")
        self._cache = {}
        
        if config:
            self.layer_config = get_layer_config()
            self.active_layers = config.get('layers', self.layer_config.get_layer_names())
            self.class_to_layer, self.class_to_name = {}, {}
            
            for layer_name in self.layer_config.get_layer_names():
                layer_config = self.layer_config.get_layer_config(layer_name)
                for i, cls_id in enumerate(layer_config['class_ids']):
                    self.class_to_layer[cls_id] = layer_name
                    if i < len(layer_config['classes']):
                        self.class_to_name[cls_id] = layer_config['classes'][i]
        
        self.logger.info(f"🔧 DatasetUtils diinisialisasi untuk: {self.data_dir}")
    
    def get_split_path(self, split: str) -> Path:
        """Dapatkan path untuk split dataset tertentu."""
        split = 'valid' if split in ('val', 'validation') else split
        if self.config:
            split_paths = self.config.get('data', {}).get('local', {})
            if split in split_paths: return Path(split_paths[split])
        return self.data_dir / split
    
    def get_class_name(self, cls_id: int) -> str:
        """Dapatkan nama kelas dari ID kelas."""
        if not hasattr(self, 'class_to_name'): return f"Class-{cls_id}"
        if cls_id in self.class_to_name: return self.class_to_name[cls_id]
        
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                idx = layer_config['class_ids'].index(cls_id)
                if idx < len(layer_config['classes']):
                    return layer_config['classes'][idx]
        return f"Class-{cls_id}"
    
    def get_layer_from_class(self, cls_id: int) -> Optional[str]:
        """Dapatkan nama layer dari ID kelas."""
        if not hasattr(self, 'class_to_layer'): return None
        if cls_id in self.class_to_layer: return self.class_to_layer[cls_id]
        
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                self.class_to_layer[cls_id] = layer_name
                return layer_name
        return None
    
    def find_image_files(self, directory: Union[str, Path], with_labels: bool = True) -> List[Path]:
        """Cari file gambar dalam direktori."""
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.warning(f"⚠️ Direktori tidak ditemukan: {dir_path}")
            return []
        
        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(list(dir_path.glob(ext)))
        
        if not with_labels: return image_files
        
        labels_dir = None
        if (dir_path / 'labels').exists(): labels_dir = dir_path / 'labels'
        elif 'images' in dir_path.parts[-1]: labels_dir = dir_path.parent / 'labels'
        
        if not labels_dir: return image_files
        return [img for img in image_files if (labels_dir / f"{img.stem}.txt").exists()]
    
    def get_random_sample(self, items: List, sample_size: int, seed: int = DEFAULT_RANDOM_SEED) -> List:
        """Ambil sampel acak dari list."""
        if sample_size <= 0 or sample_size >= len(items): return items
        random.seed(seed)
        return random.sample(items, sample_size)
    
    def load_image(self, image_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Baca gambar dari file."""
        try:
            img = cv2.imread(str(image_path))
            if img is None: raise ValueError(f"Gagal membaca gambar: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if target_size: img = cv2.resize(img, target_size)
            return img
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {image_path}: {str(e)}")
            height, width = target_size[1], target_size[0] if target_size else (640, 640)
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def parse_yolo_label(self, label_path: Path, parse_callback: Optional[Callable] = None) -> List[Dict]:
        """Parse file label YOLO."""
        if not label_path.exists(): return []
        
        bboxes = []
        try:
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:5])
                            
                            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0.001 and h > 0.001): continue
                            
                            bbox_data = {'class_id': cls_id, 'bbox': [x, y, w, h], 'line_idx': i}
                            
                            if hasattr(self, 'class_to_name'):
                                bbox_data.update({
                                    'class_name': self.get_class_name(cls_id),
                                    'layer': self.get_layer_from_class(cls_id) or 'unknown'
                                })
                            
                            if parse_callback:
                                bbox_data = parse_callback(bbox_data, line, i)
                                if not bbox_data: continue
                            
                            bboxes.append(bbox_data)
                        except (ValueError, IndexError): continue
        except Exception as e:
            self.logger.debug(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return bboxes
    
    def get_available_layers(self, label_path: Path) -> List[str]:
        """Dapatkan layer yang tersedia dalam file label."""
        if not hasattr(self, 'active_layers'): return []
        
        available_layers = set()
        for bbox in self.parse_yolo_label(label_path):
            if 'layer' in bbox and bbox['layer'] != 'unknown' and bbox['layer'] in self.active_layers:
                available_layers.add(bbox['layer'])
        
        return list(available_layers)
    
    def get_split_statistics(self, splits: List[str] = DEFAULT_SPLITS) -> Dict[str, Dict[str, int]]:
        """Dapatkan statistik dasar untuk semua split dataset."""
        self.logger.info("📊 Mengumpulkan statistik dataset")
        
        stats = {}
        for split in splits:
            split_path = self.get_split_path(split)
            images_dir, labels_dir = split_path / 'images', split_path / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                stats[split] = {'images': 0, 'labels': 0, 'status': 'missing'}
                continue
            
            image_count = sum(len(list(images_dir.glob(ext))) for ext in IMG_EXTENSIONS)
            label_count = len(list(labels_dir.glob('*.txt')))
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'status': 'valid' if image_count > 0 and label_count > 0 else 'empty'
            }
        
        log_lines = ["📊 Ringkasan dataset:"] + [
            f"   • {split.capitalize()}: {s.get('images', 0)} gambar, {s.get('labels', 0)} label"
            for split, s in stats.items()
        ]
        
        self.logger.info("\n".join(log_lines))
        return stats
    
    def backup_directory(self, source_dir: Union[str, Path], suffix: Optional[str] = None) -> Optional[Path]:
        """Buat backup direktori."""
        source_path = Path(source_dir)
        if not source_path.exists():
            self.logger.warning(f"⚠️ Direktori sumber tidak ditemukan: {source_path}")
            return None
        
        suffix = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = source_path.parent
        backup_path = parent_dir / f"{source_path.name}_backup_{suffix}"
        
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
        """Pindahkan file ke direktori target."""
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
                else: stats['skipped'] += 1
            except Exception as e:
                self.logger.error(f"❌ Gagal memindahkan {file_path}: {str(e)}")
                stats['errors'] += 1
        
        return stats
    
    def copy_dataset_files(self, source_files: List[Tuple[Path, Path]], target_dir: Path, 
                          use_symlinks: bool = False, desc: str = "Copying files") -> int:
        """Salin file-file dataset (gambar dan label)."""
        (target_dir / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        with tqdm(source_files, desc=desc) as pbar:
            for img_path, label_path in pbar:
                target_img = target_dir / 'images' / img_path.name
                target_label = target_dir / 'labels' / label_path.name
                
                try:
                    if use_symlinks:
                        if not target_img.exists(): target_img.symlink_to(img_path.resolve())
                        if not target_label.exists(): target_label.symlink_to(label_path.resolve())
                    else:
                        if not target_img.exists(): shutil.copy2(img_path, target_img)
                        if not target_label.exists(): shutil.copy2(label_path, target_label)
                    copied_count += 1
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal menyalin {img_path.name}: {str(e)}")
        
        return copied_count
    
    def split_dataset(self, split_ratios: Dict[str, float] = DEFAULT_SPLIT_RATIOS, 
                     source_dir: Optional[Path] = None, 
                     stratify_by_class: bool = True,
                     random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, int]:
        """Pecah dataset menjadi train/valid/test."""
        if abs(sum(split_ratios.values()) - 1.0) > 1e-10:
            raise ValueError(f"Total rasio harus 1.0, didapat: {sum(split_ratios.values())}")
            
        random.seed(random_seed)
        source_dir = source_dir or self.data_dir
        
        if not ((source_dir / 'images').exists() and (source_dir / 'labels').exists()):
            raise ValueError(f"Source dir harus berisi subdirektori images/ dan labels/: {source_dir}")
        
        self.logger.info(f"📊 Memecah dataset dengan rasio: {split_ratios}")
        valid_files = []
        
        for img_path in self.find_image_files(source_dir / 'images', True):
            label_path = source_dir / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_files.append((img_path, label_path))
        
        if not valid_files:
            self.logger.error("❌ Tidak ada file valid dengan pasangan gambar/label")
            return {split: 0 for split in split_ratios.keys()}
        
        self.logger.info(f"🔍 Ditemukan {len(valid_files)} file valid dengan pasangan gambar/label")
        
        # Stratify jika diminta
        files_by_class = {}
        if stratify_by_class:
            for img_path, label_path in valid_files:
                bboxes = self.parse_yolo_label(label_path)
                main_class = bboxes[0]['class_id'] if bboxes else 'unknown'
                
                if main_class not in files_by_class:
                    files_by_class[main_class] = []
                files_by_class[main_class].append((img_path, label_path))
            
            self.logger.info("📊 Distribusi kelas:" + "".join(
                f"\n   • Kelas {self.get_class_name(cls) if isinstance(cls, int) else str(cls)}: {len(files)} sampel"
                for cls, files in files_by_class.items()
            ))
        else:
            files_by_class = {'all': valid_files}
        
        # Bagi file per split
        split_files = {split: [] for split in split_ratios.keys()}
        
        for _, files in files_by_class.items():
            random.shuffle(files)
            n_total = len(files)
            
            # Alokasi file secara berurutan
            start_idx = 0
            for split, ratio in split_ratios.items():
                n_split = int(n_total * ratio)
                end_idx = start_idx + n_split
                split_files[split].extend(files[start_idx:end_idx])
                start_idx = end_idx
        
        self.logger.info("📊 Pembagian dataset:" + "".join(
            f"\n   • {split.capitalize()}: {len(files)} sampel" 
            for split, files in split_files.items()
        ))
        
        # Salin file
        splits_count = {}
        for split, files in split_files.items():
            if files:
                target_dir = self.get_split_path(split)
                splits_count[split] = self.copy_dataset_files(files, target_dir, desc=f"Copying to {split}")
        
        self.logger.success(f"✅ Pemecahan dataset selesai")
        return splits_count