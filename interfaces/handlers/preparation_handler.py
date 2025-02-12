# File: src/interfaces/handlers/preparation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi persiapan dataset

from pathlib import Path
from typing import Dict
from interfaces.handlers.base_handler import BaseHandler
import yaml         

class DataPreparationHandler(BaseHandler):
    """Handler for dataset preparation operations"""
    def __init__(self, config):
        super().__init__(config)
        self.required_structure = {
            'train': ['images', 'labels'],
            'val': ['images', 'labels'],
            'test': ['images', 'labels']
        }
        
    def prepare_dataset(self) -> bool:
        """Prepare dataset structure and configuration"""
        self.logger.info("🔄 Memulai persiapan dataset...")
        
        try:
            # Create directories
            created = self._create_directories()
            if not created:
                return False
                
            # Create configuration
            created = self._create_config()
            if not created:
                return False
                
            self.log_operation("Persiapan dataset", "success")
            return True
            
        except Exception as e:
            self.log_operation("Persiapan dataset", "failed", str(e))
            return False
            
    def _create_directories(self) -> bool:
        """Create required directory structure"""
        try:
            for split, subdirs in self.required_structure.items():
                for subdir in subdirs:
                    path = self.rupiah_dir / split / subdir
                    if not self.validate_directory(path):
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Gagal membuat struktur direktori: {str(e)}")
            return False
            
    def _create_config(self) -> bool:
        """Create dataset configuration file"""
        try:
            config = {
                'path': str(self.rupiah_dir),
                'train': str(self.rupiah_dir / 'train'),
                'val': str(self.rupiah_dir / 'val'),
                'test': str(self.rupiah_dir / 'test'),
                'nc': 7,  # jumlah kelas
                'names': ['100k', '10k', '1k', '20k', '2k', '50k', '5k']
            }
            
            config_path = self.rupiah_dir / 'rupiah.yaml'
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Gagal membuat konfigurasi: {str(e)}")
            return False
            
    def validate_preparation(self) -> Dict:
        """Validate dataset preparation"""
        missing = []
        
        for split, subdirs in self.required_structure.items():
            for subdir in subdirs:
                path = self.rupiah_dir / split / subdir
                if not path.exists():
                    missing.append(f"{split}/{subdir}")
                    
        return {
            'valid': len(missing) == 0,
            'missing': missing
        }