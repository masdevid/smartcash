# File: smartcash/cli/configuration_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manajemen konfigurasi dengan fitur pembersihan dan pelacakan versi

import os
import yaml
import shutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

class ConfigurationManager:
    """
    Manajemen konfigurasi yang cerdas untuk proyek SmartCash.
    
    Fitur utama:
    - Load konfigurasi terbaru
    - Manajemen riwayat konfigurasi
    - Pembersihan file konfigurasi lama
    """
    
    def __init__(
        self, 
        base_config_path: str, 
        config_dir: Optional[str] = None,
        max_config_age_days: int = 30,
        max_config_files: int = 10
    ):
        """
        Inisialisasi Configuration Manager.
        
        Args:
            base_config_path (str): Path ke file konfigurasi dasar
            config_dir (str, optional): Direktori untuk menyimpan konfigurasi
            max_config_age_days (int): Umur maksimal file konfigurasi (dalam hari)
            max_config_files (int): Jumlah maksimal file konfigurasi yang disimpan
        """
        self.base_config_path = Path(base_config_path)
        self.config_dir = Path(config_dir or self.base_config_path.parent)
        self.max_config_age = timedelta(days=max_config_age_days)
        self.max_config_files = max_config_files
        
        # Buat direktori jika tidak ada
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Variabel untuk menyimpan konfigurasi
        self.base_config = self._load_base_config()
        self.current_config = self.base_config.copy()
        self.config_history = []
        
    def _load_base_config(self) -> Dict[str, Any]:
        """
        Muat konfigurasi dasar dengan penanganan kesalahan.
        
        Returns:
            Dict[str, Any]: Konfigurasi dasar dengan default values
        """
        try:
            with open(self.base_config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
            
            # Default konfigurasi jika tidak ada
            default_config = {
                'data_source': None,
                'detection_mode': None,
                'backbone': None,
                'layers': ['banknote'],
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'early_stopping_patience': 10
                }
            }
            
            # Gabungkan default dengan konfigurasi yang dimuat
            return self._deep_merge(default_config, base_config)
        
        except Exception as e:
            print(f"❌ Gagal memuat konfigurasi dasar: {e}")
            return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Melakukan merge rekursif untuk konfigurasi.
        
        Args:
            base (Dict): Konfigurasi dasar
            update (Dict): Konfigurasi untuk diperbarui
        
        Returns:
            Dict: Konfigurasi yang digabungkan
        """
        merged = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def load_latest_config(self) -> Dict[str, Any]:
        """
        Muat konfigurasi training terbaru.
        
        Returns:
            Dict[str, Any]: Konfigurasi training terbaru
        """
        try:
            # Temukan file konfigurasi training terbaru
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Jika tidak ada konfigurasi, gunakan base config
            if not config_files:
                return self.base_config.copy()
            
            # Muat konfigurasi terbaru
            with open(config_files[0], 'r') as f:
                latest_config = yaml.safe_load(f) or {}
            
            # Merge dengan base config untuk memastikan semua kunci ada
            return self._deep_merge(self.base_config, latest_config)
        
        except Exception as e:
            print(f"❌ Gagal memuat konfigurasi terbaru: {e}")
            return self.base_config.copy()
    
    def save(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Simpan konfigurasi saat ini dengan timestamp.
        
        Args:
            config (Dict[str, Any], optional): Konfigurasi yang akan disimpan
        
        Returns:
            Path: Path file konfigurasi yang disimpan
        """
        config = config or self.current_config
        
        # Generate nama file dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.config_dir / f"train_config_{timestamp}.yaml"
        
        # Simpan konfigurasi
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Cleanup konfigurasi lama
        self._cleanup_old_configs()
        
        return config_file
    
    def _cleanup_old_configs(self):
        """
        Bersihkan file konfigurasi lama.
        
        - Hapus file yang lebih tua dari max_config_age
        - Batasi jumlah file konfigurasi
        """
        try:
            # Temukan semua file konfigurasi
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime
            )
            
            now = datetime.now()
            
            # Hapus file konfigurasi yang terlalu tua
            for config_file in config_files[:-self.max_config_files]:
                file_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                if now - file_mtime > self.max_config_age:
                    config_file.unlink()
                    print(f"🗑️ Menghapus konfigurasi lama: {config_file.name}")
        
        except Exception as e:
            print(f"❌ Gagal membersihkan konfigurasi: {e}")
    
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
    
    def reset(self):
        """
        Reset konfigurasi ke kondisi awal.
        """
        self.current_config = self.base_config.copy()
        self.config_history.clear()
    
    def rollback(self, steps: int = 1):
        """
        Kembalikan konfigurasi ke versi sebelumnya.
        
        Args:
            steps (int): Jumlah langkah untuk kembali
        """
        if 0 < steps <= len(self.config_history):
            self.current_config = self.config_history[-steps]
            self.config_history = self.config_history[:-steps]
        else:
            print("❌ Tidak ada konfigurasi untuk dikembalikan")