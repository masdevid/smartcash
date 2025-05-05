"""
File: smartcash/common/layer_config.py
Deskripsi: Modul untuk manajemen konfigurasi layer deteksi dengan implementasi interface
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from smartcash.common.interfaces.layer_config_interface import ILayerConfigManager

class LayerConfigManager(ILayerConfigManager):
    """
    Manager konfigurasi layer deteksi untuk memastikan konsistensi definisi layer
    dan kelas yang digunakan di berbagai bagian aplikasi.
    
    Menyediakan akses terpusat ke konfigurasi layer dan class mapping,
    dengan dukungan untuk penyimpanan dan pemulihan konfigurasi.
    
    Implementasi dari ILayerConfigManager interface.
    """
    
    # Singleton instance
    _instance = None
    
    # Default layer configuration
    DEFAULT_LAYER_CONFIG = {
        'banknote': {
            'name': 'banknote',
            'description': 'Deteksi uang kertas utuh',
            'classes': ['001', '002', '005', '010', '020', '050', '100'],
            'class_ids': [0, 1, 2, 3, 4, 5, 6],
            'threshold': 0.25,
            'enabled': True
        },
        'nominal': {
            'name': 'nominal',
            'description': 'Deteksi area nominal',
            'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
            'class_ids': [7, 8, 9, 10, 11, 12, 13],
            'threshold': 0.3,
            'enabled': True
        },
        'security': {
            'name': 'security',
            'description': 'Deteksi fitur keamanan',
            'classes': ['l3_sign', 'l3_text', 'l3_thread'],
            'class_ids': [14, 15, 16],
            'threshold': 0.35,
            'enabled': True
        }
    }
    
    def __new__(cls, *args, **kwargs):
        """Implementasi singleton untuk memastikan hanya ada satu instance."""
        if cls._instance is None: cls._instance = super(LayerConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi manager konfigurasi layer.
        
        Args:
            config: Optional konfigurasi layer
            config_path: Optional path ke file konfigurasi
            logger: Optional logger untuk mencatat aktivitas
        """
        if hasattr(self, '_initialized') and self._initialized: return
        self.logger = logger
        self.config = config or (self.load_config(config_path) and self.config if config_path and os.path.exists(config_path) else self.DEFAULT_LAYER_CONFIG)
        self._initialized = True
        
        if self.logger:
            self.logger.info(f"✅ Layer config manager diinisialisasi dengan {len(self.config)} layer")
    
    def get_layer_config(self, layer_name: str) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi untuk layer tertentu.
        
        Args:
            layer_name: Nama layer
            
        Returns:
            Dict konfigurasi layer atau dict kosong jika tidak ditemukan
        """
        return self.config.get(layer_name, {})
    
    def get_layer_names(self) -> List[str]:
        """
        Dapatkan daftar nama semua layer yang tersedia.
        
        Returns:
            List nama layer
        """
        return list(self.config.keys())
    
    def get_enabled_layers(self) -> List[str]:
        """Dapatkan daftar nama layer yang diaktifkan."""
        return [name for name, conf in self.config.items() if conf.get('enabled', True)]
    
    def get_class_map(self) -> Dict[int, str]:
        """
        Dapatkan mapping class_id ke class_name untuk semua layer.
        
        Returns:
            Dict berisi mapping class_id ke class_name
        """
        class_map = {}
        for layer_name, layer_config in self.config.items():
            classes = layer_config.get('classes', [])
            class_ids = layer_config.get('class_ids', [])
            
            for i, class_id in enumerate(class_ids):
                if i < len(classes):
                    class_map[class_id] = classes[i]
        
        return class_map
    
    def get_all_class_ids(self) -> List[int]:
        """
        Dapatkan semua class_id dari semua layer.
        
        Returns:
            List class_id
        """
        class_ids = []
        for layer_name, layer_config in self.config.items():
            class_ids.extend(layer_config.get('class_ids', []))
        return sorted(class_ids)
    
    def get_layer_for_class_id(self, class_id: int) -> Optional[str]:
        """
        Dapatkan nama layer untuk class_id tertentu.
        
        Args:
            class_id: ID kelas yang dicari
            
        Returns:
            Nama layer atau None jika tidak ditemukan
        """
        for layer_name, layer_config in self.config.items():
            if class_id in layer_config.get('class_ids', []):
                return layer_name
        return None
    
    def load_config(self, config_path: str) -> bool:
        """
        Muat konfigurasi dari file.
        
        Args:
            config_path: Path ke file konfigurasi (yaml atau json)
            
        Returns:
            Boolean yang menunjukkan keberhasilan load
        """
        try:
            path = Path(config_path)
            
            if not path.exists():
                if self.logger:
                    self.logger.warning(f"⚠️ File konfigurasi {config_path} tidak ditemukan")
                return False
                
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            else:
                if self.logger:
                    self.logger.warning(f"⚠️ Format file tidak didukung: {path.suffix}")
                return False
            
            # Check if config contains layer info
            if 'layers' in config:
                self.config = config['layers']
            else:
                self.config = config
                
            if self.logger:
                self.logger.info(f"✅ Konfigurasi layer dimuat dari {config_path}")
                
            return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Gagal memuat konfigurasi layer: {str(e)}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Simpan konfigurasi ke file.
        
        Args:
            config_path: Path untuk menyimpan file konfigurasi
            
        Returns:
            Boolean yang menunjukkan keberhasilan save
        """
        try:
            path = Path(config_path)
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                with open(path, 'w') as f:
                    yaml.dump({'layers': self.config}, f, default_flow_style=False)
            elif path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump({'layers': self.config}, f, indent=2)
            else:
                # Default to yaml
                with open(str(path) + '.yaml', 'w') as f:
                    yaml.dump({'layers': self.config}, f, default_flow_style=False)
                    
            if self.logger:
                self.logger.info(f"✅ Konfigurasi layer disimpan ke {config_path}")
                
            return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Gagal menyimpan konfigurasi layer: {str(e)}")
            return False
    
    def update_layer_config(self, layer_name: str, config: Dict) -> bool:
        """
        Update konfigurasi layer tertentu.
        
        Args:
            layer_name: Nama layer
            config: Konfigurasi baru
            
        Returns:
            Boolean yang menunjukkan keberhasilan update
        """
        try:
            # Make sure layer exists
            if layer_name not in self.config:
                self.config[layer_name] = {}
                
            # Update config
            self.config[layer_name].update(config)
            
            if self.logger:
                self.logger.info(f"✅ Konfigurasi layer {layer_name} diperbarui")
                
            return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Gagal memperbarui konfigurasi layer: {str(e)}")
            return False
    
    def set_layer_enabled(self, layer_name: str, enabled: bool) -> bool:
        """
        Aktifkan atau nonaktifkan layer.
        
        Args:
            layer_name: Nama layer
            enabled: Status enabled
            
        Returns:
            Boolean yang menunjukkan keberhasilan set
        """
        if layer_name in self.config:
            self.config[layer_name]['enabled'] = enabled
            return True
        return False
    
    def validate_class_ids(self) -> Dict:
        """
        Validasi class_id untuk memastikan tidak ada duplikat atau gap.
        
        Returns:
            Dict berisi hasil validasi
        """
        all_class_ids = []
        duplicates = []
        gaps = []
        
        # Collect all class IDs
        for layer_name, layer_config in self.config.items():
            class_ids = layer_config.get('class_ids', [])
            for class_id in class_ids:
                if class_id in all_class_ids:
                    duplicates.append((class_id, layer_name))
                all_class_ids.append(class_id)
        
        # Check for gaps
        all_class_ids = sorted(set(all_class_ids))
        if all_class_ids:
            min_id = min(all_class_ids)
            max_id = max(all_class_ids)
            for i in range(min_id, max_id + 1):
                if i not in all_class_ids:
                    gaps.append(i)
        
        # Results
        return {
            'valid': not duplicates and not gaps,
            'duplicates': duplicates,
            'gaps': gaps,
            'count': len(set(all_class_ids))
        }

# Global function to get an instance of LayerConfigManager
def get_layer_config(config_path: Optional[str] = None, logger: Optional[Any] = None) -> ILayerConfigManager:
    """
    Fungsi global untuk mendapatkan instance LayerConfigManager.
    
    Args:
        config_path: Optional path ke file konfigurasi
        logger: Optional logger
        
    Returns:
        Instance ILayerConfigManager (singleton)
    """
    return LayerConfigManager(config_path=config_path, logger=logger)