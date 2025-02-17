# File: handlers/roboflow_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk download dan setup dataset dari Roboflow API

import os
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path
from roboflow import Roboflow
from utils.logger import SmartCashLogger

class RoboflowHandler:
    """Handler untuk mengelola dataset dari Roboflow API"""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str = "data",
        api_key: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Roboflow API key tidak ditemukan. "
                "Mohon set ROBOFLOW_API_KEY environment variable"
            )
            
        self.rf = Roboflow(api_key=self.api_key)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi dataset"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['dataset']
    
    def download_dataset(self, format: str = "yolov5") -> str:
        """Download dataset dari Roboflow"""
        self.logger.start(
            f"Mendownload dataset dari Roboflow...\n"
            f"Project: {self.config['project']}\n"
            f"Version: {self.config['version']}"
        )
        
        try:
            # Get project
            project = self.rf.workspace(self.config['workspace'])\
                            .project(self.config['project'])
            
            # Download versi dataset yang sesuai
            dataset = project.version(self.config['version'])\
                           .download(format)
            
            self.logger.success(
                f"Dataset berhasil diunduh ke {dataset.location} 📥"
            )
            
            return dataset.location
            
        except Exception as e:
            self.logger.error(f"Gagal download dataset: {str(e)}")
            raise e
    
    def export_to_local(self, roboflow_dir: str) -> None:
        """Export dataset Roboflow ke struktur folder lokal"""
        self.logger.start("Mengexport dataset ke struktur folder lokal...")
        
        try:
            rf_path = Path(roboflow_dir)
            
            # Pastikan direktori tujuan ada
            self.data_dir.mkdir(exist_ok=True)
            
            # Copy untuk setiap split dataset
            for split in ['train', 'valid', 'test']:
                # Source dan target paths
                src_dir = rf_path / split
                target_dir = self.data_dir / split
                
                # Buat direktori target
                target_dir.mkdir(exist_ok=True)
                (target_dir / 'images').mkdir(exist_ok=True)
                (target_dir / 'labels').mkdir(exist_ok=True)
                
                # Copy files
                for item in ['images', 'labels']:
                    src = src_dir / item
                    if src.exists():
                        for file in src.glob('*'):
                            target = target_dir / item / file.name
                            if not target.exists():
                                self.logger.info(
                                    f"Copying {split}/{item}/{file.name}..."
                                )
                                target.write_bytes(file.read_bytes())
                
            self.logger.success(
                "Dataset berhasil diexport ke struktur folder lokal ✨"
            )
            
        except Exception as e:
            self.logger.error(f"Gagal mengexport dataset: {str(e)}")
            raise e
    
    def pull_dataset(self) -> Tuple[str, str, str]:
        """Download dan setup dataset dari Roboflow"""
        try:
            # Download dari Roboflow
            roboflow_dir = self.download_dataset()
            
            # Export ke struktur folder lokal
            self.export_to_local(roboflow_dir)
            
            # Return paths
            return (
                str(self.data_dir / 'train'),
                str(self.data_dir / 'valid'),
                str(self.data_dir / 'test')
            )
            
        except Exception as e:
            self.logger.error("Gagal melakukan pull dataset")
            raise e
    
    def get_dataset_info(self) -> Dict:
        """Mendapatkan informasi dataset dari Roboflow"""
        try:
            project = self.rf.workspace(self.config['workspace'])\
                            .project(self.config['project'])
            
            version = project.version(self.config['version'])
            
            info = {
                'name': project.name,
                'version': version.version,
                'created': version.created,
                'classes': self.config['classes'],
                'splits': {
                    'train': version.train_count,
                    'valid': version.valid_count,
                    'test': version.test_count
                }
            }
            
            self.logger.data(
                f"Dataset Info:\n"
                f"Nama: {info['name']}\n"
                f"Versi: {info['version']}\n"
                f"Created: {info['created']}\n"
                f"Train: {info['splits']['train']} images\n"
                f"Valid: {info['splits']['valid']} images\n"
                f"Test: {info['splits']['test']} images"
            )
            
            return info
            
        except Exception as e:
            self.logger.error(f"Gagal mendapatkan info dataset: {str(e)}")
            raise e