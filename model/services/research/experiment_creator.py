"""
File: smartcash/model/services/research/experiment_creator.py
Deskripsi: Komponen untuk membuat dan mengelola konfigurasi eksperimen
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.model.config import ModelConfig
from smartcash.model.config.experiment_config import ExperimentConfig


class ExperimentCreator:
    """
    Komponen untuk membuat dan mengelola konfigurasi eksperimen.
    
    Bertanggung jawab untuk:
    - Membuat eksperimen baru
    - Menyimpan konfigurasi eksperimen
    - Mengelola direktori eksperimen
    """
    
    def __init__(
        self,
        base_dir: str,
        base_config: Optional[ModelConfig] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment creator.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan hasil eksperimen
            base_config: Konfigurasi model dasar (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or get_logger()
        self.base_config = base_config or ModelConfig()
        
        # Daftar eksperimen aktif
        self.active_experiments = {}
        
        self.logger.debug(f"🧪 ExperimentCreator diinisialisasi (base_dir={self.base_dir})")
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        config_overrides: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> ExperimentConfig:
        """
        Buat eksperimen baru.
        
        Args:
            name: Nama eksperimen
            description: Deskripsi eksperimen
            config_overrides: Override untuk konfigurasi dasar
            tags: Tag untuk kategorisasi eksperimen
            
        Returns:
            ExperimentConfig untuk eksperimen baru
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        # Buat direktori eksperimen
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Buat konfigurasi eksperimen
        experiment_config = ExperimentConfig(
            name=name,
            base_config=self.base_config,
            experiment_dir=str(experiment_dir),
            description=description,
            tags=tags or []
        )
        
        # Terapkan override konfigurasi jika ada
        if config_overrides:
            for key, value in config_overrides.items():
                experiment_config.parameters[key] = value
        
        # Simpan ke active_experiments
        self.active_experiments[experiment_id] = experiment_config
        
        self.logger.info(f"✅ Eksperimen baru dibuat: {name} (ID: {experiment_id})")
        return experiment_config
    
    def create_experiment_group(
        self,
        name: str,
        group_type: str
    ) -> Dict[str, Any]:
        """
        Buat grup eksperimen untuk perbandingan atau tuning.
        
        Args:
            name: Nama grup eksperimen
            group_type: Tipe grup (comparison, tuning, dll)
            
        Returns:
            Dictionary dengan informasi grup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_id = f"{group_type}_{name}_{timestamp}"
        group_dir = self.base_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📂 Grup eksperimen dibuat: {group_id}")
        
        return {
            "id": group_id,
            "dir": group_dir,
            "name": name,
            "timestamp": timestamp,
            "type": group_type
        }
    
    def get_experiment_config(
        self,
        experiment_id: str
    ) -> Optional[ExperimentConfig]:
        """
        Dapatkan konfigurasi eksperimen berdasarkan ID.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            ExperimentConfig untuk eksperimen atau None jika tidak ditemukan
        """
        # Cek di active_experiments terlebih dahulu
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        
        # Coba load dari direktori
        experiment_dir = self.base_dir / experiment_id
        if experiment_dir.exists() and (experiment_dir / "config.yaml").exists():
            try:
                config = ExperimentConfig(name="loaded", experiment_dir=str(experiment_dir))
                return config
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
        
        return None