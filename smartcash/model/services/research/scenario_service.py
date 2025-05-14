"""
File: smartcash/model/services/research/scenario_service.py
Deskripsi: Layanan untuk manajemen skenario penelitian pada model SmartCash
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager

class ScenarioService:
    """
    Layanan untuk manajemen skenario penelitian pada model SmartCash.
    
    Bertanggung jawab untuk:
    - Membuat skenario penelitian baru
    - Mengelola konfigurasi skenario
    - Menjalankan eksperimen dalam skenario
    - Menganalisis hasil skenario
    """
    
    def __init__(
        self,
        scenario_name: Optional[str] = None,
        base_dir: str = '/content/scenarios',
        logger = None
    ):
        """
        Inisialisasi ScenarioService.
        
        Args:
            scenario_name: Nama skenario (opsional)
            base_dir: Direktori dasar untuk menyimpan skenario
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("scenario_service")
        self.config_manager = get_config_manager()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.scenario_name = scenario_name
        self.scenario_dir = None
        self.config = {}
        
        if scenario_name:
            self.load_scenario(scenario_name)
    
    def create_scenario(
        self,
        name: str,
        description: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Membuat skenario penelitian baru.
        
        Args:
            name: Nama skenario
            description: Deskripsi skenario
            config: Konfigurasi skenario
            
        Returns:
            Dict berisi informasi skenario
        """
        # Validasi nama skenario
        if not name or not isinstance(name, str):
            raise ValueError("Nama skenario harus berupa string yang valid")
        
        # Buat direktori skenario
        self.scenario_name = name
        self.scenario_dir = self.base_dir / name
        self.scenario_dir.mkdir(exist_ok=True)
        
        # Buat metadata skenario
        metadata = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'experiments': [],
            'status': 'created'
        }
        
        # Simpan metadata
        with open(self.scenario_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Simpan konfigurasi
        self.config = config
        with open(self.scenario_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        self.logger.info(f"✅ Skenario '{name}' berhasil dibuat di {self.scenario_dir}")
        return metadata
    
    def load_scenario(self, name: str) -> Dict[str, Any]:
        """
        Muat skenario yang sudah ada.
        
        Args:
            name: Nama skenario
            
        Returns:
            Dict berisi metadata skenario
            
        Raises:
            FileNotFoundError: Jika skenario tidak ditemukan
        """
        scenario_dir = self.base_dir / name
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Skenario '{name}' tidak ditemukan di {self.base_dir}")
        
        # Muat metadata
        try:
            with open(scenario_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat metadata skenario '{name}': {str(e)}")
            raise
        
        # Muat konfigurasi
        try:
            with open(scenario_dir / 'config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi skenario '{name}': {str(e)}")
            raise
        
        # Update state
        self.scenario_name = name
        self.scenario_dir = scenario_dir
        
        self.logger.info(f"✅ Skenario '{name}' berhasil dimuat dari {scenario_dir}")
        return metadata
    
    def list_scenarios(self) -> List[Dict[str, Any]]:
        """
        Daftar semua skenario yang tersedia.
        
        Returns:
            List berisi metadata skenario
        """
        scenarios = []
        
        for scenario_dir in self.base_dir.glob('*'):
            if not scenario_dir.is_dir():
                continue
                
            metadata_file = scenario_dir / 'metadata.json'
            if not metadata_file.exists():
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    metadata['path'] = str(scenario_dir)
                    scenarios.append(metadata)
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat metadata dari {scenario_dir}: {str(e)}")
        
        return scenarios
    
    def add_experiment(
        self,
        experiment_id: str,
        experiment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tambahkan eksperimen ke skenario.
        
        Args:
            experiment_id: ID eksperimen
            experiment_config: Konfigurasi eksperimen
            
        Returns:
            Dict berisi metadata skenario yang diperbarui
        """
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        # Muat metadata
        metadata_file = self.scenario_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Tambahkan eksperimen
        experiment_info = {
            'id': experiment_id,
            'added_at': datetime.now().isoformat(),
            'status': 'added'
        }
        
        # Cek apakah eksperimen sudah ada
        for i, exp in enumerate(metadata['experiments']):
            if exp['id'] == experiment_id:
                metadata['experiments'][i] = experiment_info
                break
        else:
            metadata['experiments'].append(experiment_info)
        
        # Simpan metadata
        metadata['updated_at'] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Simpan konfigurasi eksperimen
        experiments_dir = self.scenario_dir / 'experiments'
        experiments_dir.mkdir(exist_ok=True)
        
        with open(experiments_dir / f'{experiment_id}.yaml', 'w') as f:
            yaml.dump(experiment_config, f)
        
        self.logger.info(f"✅ Eksperimen '{experiment_id}' berhasil ditambahkan ke skenario '{self.scenario_name}'")
        return metadata
    
    def get_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi eksperimen.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            Dict berisi konfigurasi eksperimen
        """
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        experiments_dir = self.scenario_dir / 'experiments'
        config_file = experiments_dir / f'{experiment_id}.yaml'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Konfigurasi eksperimen '{experiment_id}' tidak ditemukan")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perbarui status eksperimen.
        
        Args:
            experiment_id: ID eksperimen
            status: Status baru ('running', 'completed', 'failed')
            results: Hasil eksperimen (opsional)
            
        Returns:
            Dict berisi metadata skenario yang diperbarui
        """
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        # Muat metadata
        metadata_file = self.scenario_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Perbarui status eksperimen
        experiment_found = False
        for i, exp in enumerate(metadata['experiments']):
            if exp['id'] == experiment_id:
                metadata['experiments'][i]['status'] = status
                metadata['experiments'][i]['updated_at'] = datetime.now().isoformat()
                experiment_found = True
                break
        
        if not experiment_found:
            raise ValueError(f"Eksperimen '{experiment_id}' tidak ditemukan dalam skenario")
        
        # Simpan metadata
        metadata['updated_at'] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Simpan hasil eksperimen jika ada
        if results:
            results_dir = self.scenario_dir / 'results'
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / f'{experiment_id}.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        self.logger.info(f"✅ Status eksperimen '{experiment_id}' diperbarui menjadi '{status}'")
        return metadata
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Dapatkan hasil eksperimen.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            Dict berisi hasil eksperimen
        """
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        results_dir = self.scenario_dir / 'results'
        results_file = results_dir / f'{experiment_id}.json'
        
        if not results_file.exists():
            raise FileNotFoundError(f"Hasil eksperimen '{experiment_id}' tidak ditemukan")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Bandingkan hasil beberapa eksperimen.
        
        Args:
            experiment_ids: List ID eksperimen
            metrics: List metrik yang akan dibandingkan
            
        Returns:
            DataFrame berisi perbandingan eksperimen
        """
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                results = self.get_experiment_results(exp_id)
                
                # Ekstrak metrik yang diminta
                exp_metrics = {}
                for metric in metrics:
                    if metric in results:
                        exp_metrics[metric] = results[metric]
                    else:
                        exp_metrics[metric] = None
                
                exp_metrics['experiment_id'] = exp_id
                comparison_data.append(exp_metrics)
                
            except FileNotFoundError:
                self.logger.warning(f"⚠️ Hasil eksperimen '{exp_id}' tidak ditemukan")
        
        # Buat DataFrame
        if comparison_data:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame(columns=['experiment_id'] + metrics)
    
    def export_scenario(self, output_path: Optional[str] = None) -> str:
        """
        Ekspor skenario ke file ZIP.
        
        Args:
            output_path: Path untuk file output (opsional)
            
        Returns:
            Path ke file ZIP
        """
        import shutil
        
        if not self.scenario_dir:
            raise ValueError("Skenario belum dimuat atau dibuat")
        
        # Tentukan output path
        if not output_path:
            output_path = str(self.base_dir / f"{self.scenario_name}_export.zip")
        
        # Buat ZIP file
        shutil.make_archive(
            output_path.replace('.zip', ''),
            'zip',
            self.scenario_dir
        )
        
        self.logger.info(f"✅ Skenario '{self.scenario_name}' berhasil diekspor ke {output_path}")
        return output_path
