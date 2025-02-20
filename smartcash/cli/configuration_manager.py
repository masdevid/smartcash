# File: smartcash/cli/configuration_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manajemen konfigurasi yang cerdas dengan pembersihan otomatis dan pengambilan konfigurasi terbaru

import os
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

class ConfigurationManager:
    """
    Manajemen konfigurasi yang cerdas untuk proyek SmartCash.
    """
    
    def __init__(
        self, 
        base_config_path: str, 
        config_dir: Optional[str] = None,
        max_config_age_days: int = 30,
        max_config_files: int = 1
    ):
        """
        Inisialisasi Configuration Manager.
        """
        # Definisikan default config paling awal
        self.default_config: Dict[str, Any] = {
            'data_source': None,
            'detection_mode': None,
            'backbone': None,
            'layers': ['banknote'],
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping_patience': 10
            },
            'roboflow': {
                'workspace': None,
                'project': None,
                'version': None
            }
        }
        
        # Inisialisasi atribut konfigurasi
        self.base_config_path: Path = Path(base_config_path)
        self.config_dir: Path = Path(config_dir or self.base_config_path.parent)
        self.max_config_age: timedelta = timedelta(days=max_config_age_days)
        self.max_config_files: int = max_config_files
        
        # Buat direktori jika tidak ada
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi atribut konfigurasi
        self.base_config: Dict[str, Any] = self._load_base_config()
        self.current_config: Dict[str, Any] = self._load_latest_config() or self.base_config.copy()
        self.config_history: List[Dict[str, Any]] = []
        
        # Lakukan pembersihan saat inisialisasi
        self._cleanup_old_configs()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """
        Muat konfigurasi dasar dengan penanganan kesalahan.
        """
        try:
            # Coba memuat konfigurasi dari file
            if self.base_config_path.exists():
                with open(self.base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f) or {}
                
                # Gabungkan default dengan konfigurasi yang dimuat
                return self._deep_merge(self.default_config, base_config)
            
            return self.default_config
        
        except Exception as e:
            print(f"❌ Gagal memuat konfigurasi dasar: {e}")
            return self.default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Melakukan merge rekursif untuk konfigurasi.
        """
        merged = base.copy()
        for key, value in update.items():
            if isinstance(value, dict):
                # Jika kunci sudah ada dan bernilai dict, lakukan merge rekursif
                merged[key] = self._deep_merge(merged.get(key, {}), value)
            else:
                # Jika bukan dict, langsung timpa atau tambahkan
                merged[key] = value
        return merged
    
    def _load_latest_config(self) -> Optional[Dict[str, Any]]:
        """
        Muat konfigurasi training terbaru.
        """
        try:
            # Temukan file konfigurasi training terbaru
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Kembalikan konfigurasi terbaru jika ada
            if config_files:
                with open(config_files[0], 'r') as f:
                    latest_config = yaml.safe_load(f) or {}
                
                # Merge dengan base config untuk memastikan semua kunci ada
                return self._deep_merge(self.base_config, latest_config)
            
            return None
        
        except Exception as e:
            print(f"❌ Gagal memuat konfigurasi terbaru: {str(e)}")
            return None
    
    def _cleanup_old_configs(self):
        """
        Bersihkan file konfigurasi lama dengan debugging detail.
        """
        try:
            # Temukan semua file konfigurasi
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Batasi jumlah file yang disimpan
            files_to_keep = config_files[:self.max_config_files]
            files_to_remove = config_files[self.max_config_files:]
            
            # Hapus file yang berlebihan
            for file in files_to_remove:
                try:
                    file.unlink()
                except PermissionError:
                    print(f"❌ Izin ditolak untuk menghapus {file.name}")
                except FileNotFoundError:
                    print(f"❌ File tidak ditemukan: {file.name}")
                except Exception as remove_error:
                    print(f"❌ Gagal menghapus {file.name}: {remove_error}")
            
        except Exception as e:
            print(f"❌ Gagal membersihkan konfigurasi: {str(e)}")

    def save(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Simpan konfigurasi saat ini dengan timestamp dan bersihkan file lama.
        """
        # Bersihkan file konfigurasi sebelum menyimpan
        self._cleanup_old_configs()
        
        config = config or self.current_config
        
        # Generate nama file dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.config_dir / f"train_config_{timestamp}.yaml"
        
        # Simpan konfigurasi
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file


    
    def update(self, key: str, value: Any):
        """
        Perbarui konfigurasi dengan mendukung akses bertingkat.
        
        Args:
            key (str): Kunci yang akan diperbarui (mendukung dot notation)
            value (Any): Nilai baru
        """
        keys = key.split('.')
        config = self.current_config
        
        # Navigasi hingga kunci terakhir
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        
        # Perbarui nilai
        config[keys[-1]] = value
        
        # Catat riwayat
        self.config_history.append(self.current_config.copy())
    
    def save(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Simpan konfigurasi saat ini dengan timestamp dan bersihkan file lama.
        """
        # Bersihkan file konfigurasi sebelum menyimpan
        self._cleanup_old_configs()
        
        config = config or self.current_config
        
        # Generate nama file dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.config_dir / f"train_config_{timestamp}.yaml"
        
        # Simpan konfigurasi
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_file