# File: smartcash/utils/layer_config_manager.py
# Author: Alfrida Sabar
# Deskripsi: Modul untuk pengelolaan konfigurasi layer secara terpusat untuk menghindari redundansi

from typing import Dict, List, Optional, Union, Set
import yaml
from pathlib import Path
import json
import os
from functools import lru_cache

from smartcash.utils.logger import SmartCashLogger

class LayerConfigManager:
    """
    Pengelola konfigurasi layer terpusat untuk SmartCash.
    Menghilangkan duplikasi konfigurasi layer di seluruh aplikasi.
    Mendukung loading dari file yaml atau default builtin.
    """
    
    # Konfigurasi default jika tidak ada file eksternal
    DEFAULT_CONFIG = {
        'banknote': {
            'classes': ['001', '002', '005', '010', '020', '050', '100'],
            'class_ids': list(range(7))  # 0-6
        },
        'nominal': {
            'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
            'class_ids': list(range(7, 14))  # 7-13
        },
        'security': {
            'classes': ['l3_sign', 'l3_text', 'l3_thread'],
            'class_ids': list(range(14, 17))  # 14-16
        }
    }
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implementasi singleton untuk memastikan konfigurasi konsisten."""
        if cls._instance is None:
            cls._instance = super(LayerConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pengelola konfigurasi layer.
        
        Args:
            config_path: Path ke file konfigurasi layer (yaml/json)
            logger: Logger kustom
        """
        # Cek apakah sudah diinisialisasi (untuk singleton)
        if self._initialized:
            return
            
        self.logger = logger or SmartCashLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        
        # Load konfigurasi
        self.layer_config = self._load_config()
        self._class_to_layer_map = self._build_class_to_layer_map()
        
        self._initialized = True
        
        self.logger.info(f"✅ Konfigurasi layer dimuat: {len(self.layer_config)} layer tersedia")
    
    def _load_config(self) -> Dict:
        """
        Load konfigurasi layer dari file atau gunakan default.
        
        Returns:
            Dict konfigurasi layer
        """
        if not self.config_path or not self.config_path.exists():
            self.logger.info("ℹ️ Menggunakan konfigurasi layer default")
            return self.DEFAULT_CONFIG
            
        try:
            ext = self.config_path.suffix.lower()
            
            if ext == '.yaml' or ext == '.yml':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif ext == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                self.logger.warning(f"⚠️ Format file konfigurasi tidak didukung: {ext}")
                return self.DEFAULT_CONFIG
                
            # Validasi konfigurasi
            if self._validate_config(config):
                self.logger.success(f"✅ Konfigurasi berhasil dimuat dari {self.config_path}")
                return config
            else:
                self.logger.warning("⚠️ Konfigurasi tidak valid, menggunakan default")
                return self.DEFAULT_CONFIG
                
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            return self.DEFAULT_CONFIG
    
    def _validate_config(self, config: Dict) -> bool:
        """
        Validasi struktur konfigurasi.
        
        Args:
            config: Konfigurasi yang akan divalidasi
            
        Returns:
            Boolean hasil validasi
        """
        if not isinstance(config, dict):
            return False
            
        for layer_name, layer_config in config.items():
            # Setiap layer harus memiliki 'classes' dan 'class_ids'
            if not isinstance(layer_config, dict):
                return False
                
            if 'classes' not in layer_config or 'class_ids' not in layer_config:
                return False
                
            if not isinstance(layer_config['classes'], list) or not isinstance(layer_config['class_ids'], list):
                return False
                
            # Jumlah kelas dan class_ids harus sama
            if len(layer_config['classes']) != len(layer_config['class_ids']):
                return False
        
        return True
    
    def _build_class_to_layer_map(self) -> Dict[int, str]:
        """
        Buat mapping dari class_id ke layer name.
        
        Returns:
            Dict mapping {class_id: layer_name}
        """
        class_to_layer = {}
        
        for layer_name, layer_config in self.layer_config.items():
            for class_id in layer_config['class_ids']:
                class_to_layer[class_id] = layer_name
                
        return class_to_layer
    
    def get_layer_config(self, layer_name: Optional[str] = None) -> Dict:
        """
        Dapatkan konfigurasi untuk layer tertentu atau semua layer.
        
        Args:
            layer_name: Nama layer (jika None, kembalikan semua)
            
        Returns:
            Dict konfigurasi layer
        """
        if layer_name is None:
            return self.layer_config
            
        if layer_name in self.layer_config:
            return self.layer_config[layer_name]
            
        self.logger.warning(f"⚠️ Layer tidak ditemukan: {layer_name}")
        return {}
    
    def get_all_class_ids(self) -> List[int]:
        """
        Dapatkan semua class_id dari semua layer.
        
        Returns:
            List semua class_id
        """
        class_ids = []
        
        for layer_config in self.layer_config.values():
            class_ids.extend(layer_config['class_ids'])
            
        return sorted(class_ids)
    
    def get_all_classes(self) -> List[str]:
        """
        Dapatkan semua class name dari semua layer.
        
        Returns:
            List semua class name
        """
        classes = []
        
        for layer_config in self.layer_config.values():
            classes.extend(layer_config['classes'])
            
        return classes
    
    def get_layer_for_class_id(self, class_id: int) -> Optional[str]:
        """
        Dapatkan nama layer untuk class_id tertentu.
        
        Args:
            class_id: ID kelas yang dicari
            
        Returns:
            Nama layer atau None jika tidak ditemukan
        """
        return self._class_to_layer_map.get(class_id)
    
    def get_class_name(self, class_id: int) -> Optional[str]:
        """
        Dapatkan nama kelas berdasarkan class_id.
        
        Args:
            class_id: ID kelas
            
        Returns:
            Nama kelas atau None jika tidak ditemukan
        """
        layer_name = self.get_layer_for_class_id(class_id)
        
        if not layer_name:
            return None
            
        layer_config = self.layer_config[layer_name]
        class_ids = layer_config['class_ids']
        classes = layer_config['classes']
        
        try:
            idx = class_ids.index(class_id)
            return classes[idx]
        except (ValueError, IndexError):
            return None
    
    @lru_cache(maxsize=32)
    def get_total_classes(self) -> int:
        """
        Dapatkan total jumlah kelas dari semua layer.
        
        Returns:
            Total jumlah kelas
        """
        return len(self.get_all_class_ids())
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Simpan konfigurasi layer ke file.
        
        Args:
            output_path: Path output (default: self.config_path)
            
        Returns:
            Boolean sukses/gagal
        """
        output_path = Path(output_path) if output_path else self.config_path
        
        if not output_path:
            self.logger.error("❌ Path output tidak ditentukan")
            return False
            
        try:
            # Buat direktori jika belum ada
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            ext = output_path.suffix.lower()
            
            if ext == '.yaml' or ext == '.yml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.layer_config, f, default_flow_style=False)
            elif ext == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.layer_config, f, indent=2)
            else:
                self.logger.warning(f"⚠️ Format file tidak didukung: {ext}")
                return False
                
            self.logger.success(f"✅ Konfigurasi berhasil disimpan ke {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan konfigurasi: {str(e)}")
            return False
            
    def get_layer_names(self) -> List[str]:
        """
        Dapatkan daftar nama layer yang tersedia.
        
        Returns:
            List nama layer
        """
        return list(self.layer_config.keys())
        
    def filter_classes_by_layers(self, active_layers: List[str]) -> Set[int]:
        """
        Filter class_ids berdasarkan layer yang aktif.
        
        Args:
            active_layers: List nama layer yang aktif
            
        Returns:
            Set class_ids yang termasuk dalam layer aktif
        """
        active_class_ids = set()
        
        for layer in active_layers:
            if layer in self.layer_config:
                active_class_ids.update(self.layer_config[layer]['class_ids'])
                
        return active_class_ids

# Fungsi helper untuk mendapatkan instance
def get_layer_config(config_path: Optional[Union[str, Path]] = None) -> LayerConfigManager:
    """
    Dapatkan instance LayerConfigManager.
    
    Args:
        config_path: Path ke file konfigurasi (opsional)
        
    Returns:
        Instance LayerConfigManager
    """
    return LayerConfigManager(config_path)
