"""
File: smartcash/dataset/utils/split/stratifier.py
Deskripsi: Implementasi strategi stratifikasi dataset berdasarkan kelas atau layer
"""

import random
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DEFAULT_RANDOM_SEED, DEFAULT_SPLIT_RATIOS


class DatasetStratifier:
    """Stratifikasi dataset berdasarkan berbagai kriteria untuk split yang seimbang."""
    
    def __init__(self, config: Dict, data_dir: str = None, logger=None):
        """
        Inisialisasi DatasetStratifier.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_stratifier")
        
        # Setup layer config
        self.layer_config = get_layer_config()
        
        self.logger.info(f"📊 DatasetStratifier diinisialisasi")
    
    def stratify_by_class(
        self, 
        files: List[Tuple[Path, Path]], 
        class_ratios: Optional[Dict[int, float]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED
    ) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Stratifikasi dataset berdasarkan kelas.
        
        Args:
            files: List pasangan file (gambar, label)
            class_ratios: Rasio distribusi kelas dalam bentuk {class_id: ratio}
            random_seed: Seed untuk random
            
        Returns:
            Dictionary berisi file terstratifikasi {'train': [...], 'valid': [...], 'test': [...]}
        """
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Default ratios
        if class_ratios is None:
            class_ratios = DEFAULT_SPLIT_RATIOS
            
        # Kelompokkan file berdasarkan kelas
        files_by_class = self._group_files_by_class(files)
        
        self.logger.info(f"📊 Stratifikasi dataset berdasarkan kelas dengan rasio: {class_ratios}")
        
        # Periksa distribusi kelas
        for cls, cls_files in files_by_class.items():
            self.logger.info(f"   • Kelas {cls}: {len(cls_files)} sampel")
            
        # Inisialisasi hasil
        result = {'train': [], 'valid': [], 'test': []}
        
        # Proses setiap kelas
        for cls, cls_files in files_by_class.items():
            # Acak file
            random.shuffle(cls_files)
            
            # Hitung jumlah untuk setiap split
            num_train = int(len(cls_files) * class_ratios['train'])
            num_valid = int(len(cls_files) * class_ratios['valid'])
            
            # Bagi file
            result['train'].extend(cls_files[:num_train])
            result['valid'].extend(cls_files[num_train:num_train + num_valid])
            result['test'].extend(cls_files[num_train + num_valid:])
                
        # Log hasil
        self.logger.info(
            f"📊 Hasil stratifikasi:\n"
            f"   • Train: {len(result['train'])} sampel\n"
            f"   • Valid: {len(result['valid'])} sampel\n"
            f"   • Test: {len(result['test'])} sampel"
        )
        
        return result
    
    def stratify_by_layer(
        self, 
        files: List[Tuple[Path, Path]], 
        layer_ratios: Optional[Dict[str, Dict[str, float]]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED
    ) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Stratifikasi dataset berdasarkan layer.
        
        Args:
            files: List pasangan file (gambar, label)
            layer_ratios: Rasio distribusi layer per split: {layer: {'train': 0.7, 'valid': 0.2, 'test': 0.1}}
            random_seed: Seed untuk random
            
        Returns:
            Dictionary berisi file terstratifikasi {'train': [...], 'valid': [...], 'test': [...]}
        """
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Default rasio
        default_ratios = DEFAULT_SPLIT_RATIOS
        
        # Gunakan default jika layer_ratios tidak disediakan
        if layer_ratios is None:
            layer_ratios = {layer: default_ratios.copy() for layer in self.layer_config.get_layer_names()}
            
        # Kelompokkan file berdasarkan layer
        files_by_layer = self._group_files_by_layer(files)
        
        self.logger.info(f"📊 Stratifikasi dataset berdasarkan layer")
        
        # Periksa distribusi layer
        for layer, layer_files in files_by_layer.items():
            self.logger.info(f"   • Layer {layer}: {len(layer_files)} sampel")
            
        # Inisialisasi hasil, menggunakan set untuk menghindari duplikat
        result_sets = {'train': set(), 'valid': set(), 'test': set()}
        
        # Proses setiap layer
        for layer, layer_files in files_by_layer.items():
            # Dapatkan rasio untuk layer ini
            ratios = layer_ratios.get(layer, default_ratios)
            
            # Acak file
            random.shuffle(layer_files)
            
            # Hitung jumlah untuk setiap split
            num_train = int(len(layer_files) * ratios['train'])
            num_valid = int(len(layer_files) * ratios['valid'])
            
            # Tambahkan file ke set
            for i, f in enumerate(layer_files):
                if i < num_train:
                    result_sets['train'].add(f)
                elif i < num_train + num_valid:
                    result_sets['valid'].add(f)
                else:
                    result_sets['test'].add(f)
                    
        # Konversi set ke list
        result = {k: list(v) for k, v in result_sets.items()}
                
        # Log hasil
        self.logger.info(
            f"📊 Hasil stratifikasi:\n"
            f"   • Train: {len(result['train'])} sampel\n"
            f"   • Valid: {len(result['valid'])} sampel\n"
            f"   • Test: {len(result['test'])} sampel"
        )
        
        return result
    
    def stratify_by_count(
        self, 
        files: List[Tuple[Path, Path]], 
        train_count: int, 
        valid_count: int, 
        test_count: Optional[int] = None,
        random_seed: int = DEFAULT_RANDOM_SEED
    ) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Stratifikasi dataset berdasarkan jumlah sampel per split.
        
        Args:
            files: List pasangan file (gambar, label)
            train_count: Jumlah sampel untuk train
            valid_count: Jumlah sampel untuk valid
            test_count: Jumlah sampel untuk test (opsional, default=sisanya)
            random_seed: Seed untuk random
            
        Returns:
            Dictionary berisi file terstratifikasi {'train': [...], 'valid': [...], 'test': [...]}
        """
        # Set random seed
        random.seed(random_seed)
        
        # Acak file
        files_copy = files.copy()
        random.shuffle(files_copy)
        
        # Validasi jumlah
        total_files = len(files_copy)
        
        if train_count + valid_count > total_files:
            self.logger.warning(f"⚠️ Jumlah sampel yang diminta ({train_count + valid_count}) melebihi total file ({total_files})")
            train_count = min(train_count, int(total_files * 0.7))
            valid_count = min(valid_count, int(total_files * 0.2))
            
        # Tentukan test_count jika tidak disediakan
        if test_count is None:
            test_count = total_files - train_count - valid_count
        else:
            # Pastikan total tidak melebihi jumlah file
            if train_count + valid_count + test_count > total_files:
                test_count = total_files - train_count - valid_count
        
        self.logger.info(
            f"📊 Stratifikasi dataset berdasarkan jumlah:\n"
            f"   • Train: {train_count} sampel\n"
            f"   • Valid: {valid_count} sampel\n"
            f"   • Test: {test_count} sampel\n"
            f"   • Total: {total_files} sampel"
        )
        
        # Bagi file
        result = {
            'train': files_copy[:train_count],
            'valid': files_copy[train_count:train_count + valid_count],
            'test': files_copy[train_count + valid_count:train_count + valid_count + test_count]
        }
        
        return result
    
    def _group_files_by_class(self, files: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Kelompokkan file berdasarkan kelas utama dalam file label.
        
        Args:
            files: List pasangan file (gambar, label)
            
        Returns:
            Dictionary berisi file per kelas
        """
        from smartcash.dataset.utils.dataset_utils import DatasetUtils
        utils = DatasetUtils(self.config, str(self.data_dir), self.logger)
        
        files_by_class = defaultdict(list)
        
        for img_path, label_path in files:
            # Dapatkan kelas dari file label
            bbox_data = utils.parse_yolo_label(label_path)
            if not bbox_data:
                continue
                
            # Gunakan kelas pertama sebagai kelas utama
            main_class = bbox_data[0]['class_id']
            # Dapatkan nama kelas jika tersedia
            if 'class_name' in bbox_data[0]:
                main_class = bbox_data[0]['class_name']
                
            files_by_class[main_class].append((img_path, label_path))
            
        return dict(files_by_class)
    
    def _group_files_by_layer(self, files: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Kelompokkan file berdasarkan layer dalam file label.
        
        Args:
            files: List pasangan file (gambar, label)
            
        Returns:
            Dictionary berisi file per layer
        """
        from smartcash.dataset.utils.dataset_utils import DatasetUtils
        utils = DatasetUtils(self.config, str(self.data_dir), self.logger)
        
        files_by_layer = defaultdict(list)
        
        for img_path, label_path in files:
            # Dapatkan layer dari file label
            available_layers = utils.get_available_layers(label_path)
            
            if not available_layers:
                continue
                
            # File dapat berada di multiple layer
            for layer in available_layers:
                files_by_layer[layer].append((img_path, label_path))
            
        return dict(files_by_layer)