# File: handlers/data_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk dataset management dan augmentasi data lokal

import os
from typing import Dict, Optional, Tuple
import yaml
import shutil
from pathlib import Path
from utils.logger import SmartCashLogger

class DataHandler:
    """Handler untuk mengelola dataset lokal hasil download dari Roboflow"""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str = "data",
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi dataset"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['dataset']
    
    def setup_dataset_structure(self) -> None:
        """Setup struktur folder dataset"""
        self.logger.start("Menyiapkan struktur folder dataset...")
        
        try:
            # Buat direktori utama jika belum ada
            self.data_dir.mkdir(exist_ok=True)
            
            # Buat subdirektori untuk train, valid, test
            for split in ['train', 'valid', 'test']:
                split_dir = self.data_dir / split
                split_dir.mkdir(exist_ok=True)
                
                # Buat subdirektori images dan labels
                (split_dir / 'images').mkdir(exist_ok=True)
                (split_dir / 'labels').mkdir(exist_ok=True)
            
            self.logger.success("Struktur folder dataset berhasil disiapkan ✨")
            
        except Exception as e:
            self.logger.error(f"Gagal menyiapkan struktur folder: {str(e)}")
            raise e
    
    def get_dataset_stats(self) -> Dict:
        """Mendapatkan statistik dataset"""
        stats = {}
        
        try:
            for split in ['train', 'valid', 'test']:
                split_dir = self.data_dir / split
                
                n_images = len(list((split_dir / 'images').glob('*')))
                n_labels = len(list((split_dir / 'labels').glob('*')))
                
                stats[split] = {
                    'images': n_images,
                    'labels': n_labels
                }
                
                self.logger.data(
                    f"📊 {split.capitalize()}: "
                    f"{n_images} images, {n_labels} labels"
                )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Gagal mendapatkan statistik: {str(e)}")
            raise e
    
    def verify_dataset(self) -> bool:
        """Verifikasi struktur dan integritas dataset"""
        self.logger.info("Memverifikasi dataset...")
        
        try:
            is_valid = True
            
            for split in ['train', 'valid', 'test']:
                split_dir = self.data_dir / split
                
                # Cek keberadaan direktori
                if not (split_dir / 'images').exists() or \
                   not (split_dir / 'labels').exists():
                    self.logger.warning(
                        f"Direktori images/labels tidak ditemukan di {split}"
                    )
                    is_valid = False
                    continue
                
                # Cek jumlah file
                n_images = len(list((split_dir / 'images').glob('*')))
                n_labels = len(list((split_dir / 'labels').glob('*')))
                
                if n_images != n_labels:
                    self.logger.warning(
                        f"Jumlah images ({n_images}) tidak sama dengan "
                        f"jumlah labels ({n_labels}) di {split}"
                    )
                    is_valid = False
                
                # Cek nama file
                image_names = {f.stem for f in (split_dir / 'images').glob('*')}
                label_names = {f.stem for f in (split_dir / 'labels').glob('*')}
                
                if image_names != label_names:
                    self.logger.warning(
                        f"Ada ketidaksesuaian nama file di {split}"
                    )
                    is_valid = False
            
            if is_valid:
                self.logger.success("Dataset valid dan siap digunakan! ✨")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Validasi dataset gagal: {str(e)}")
            return False
    
    def get_data_paths(self) -> Tuple[str, str, str]:
        """Mendapatkan path untuk train, validation dan test set"""
        return (
            str(self.data_dir / 'train'),
            str(self.data_dir / 'valid'), 
            str(self.data_dir / 'test')
        )
    
    def get_class_names(self) -> list:
        """Mendapatkan daftar kelas (denominasi rupiah)"""
        return self.config['classes']