# File: smartcash/handlers/model/integration/experiment_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan ExperimentTracker

from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.experiment_tracker import ExperimentTracker

class ExperimentAdapter:
    """
    Adapter untuk integrasi dengan ExperimentTracker.
    Menyediakan antarmuka yang konsisten untuk tracking eksperimen.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi experiment adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("experiment_adapter")
        
        # Set default experiment name berdasarkan config
        self.experiment_name = config.get('experiment', {}).get('name', "default_experiment")
        
        # Set output directory berdasarkan config
        self.output_dir = config.get('experiment', {}).get('output_dir', "runs/train/experiments")
        
        # Lazy initialization untuk experiment tracker
        self._experiment_tracker = None
        
        self.logger.info(f"🧪 ExperimentAdapter diinisialisasi untuk {self.experiment_name}")
    
    @property
    def experiment_tracker(self) -> ExperimentTracker:
        """Lazy-loaded experiment tracker."""
        if self._experiment_tracker is None:
            self._experiment_tracker = ExperimentTracker(
                experiment_name=self.experiment_name,
                output_dir=self.output_dir,
                logger=self.logger
            )
        return self._experiment_tracker
    
    def set_experiment_name(self, experiment_name: str) -> None:
        """
        Set nama eksperimen baru.
        
        Args:
            experiment_name: Nama eksperimen
        """
        self.experiment_name = experiment_name
        self._experiment_tracker = None  # Reset tracker untuk inisialisasi ulang dengan nama baru
        self.logger.info(f"🧪 Experiment name diubah ke {experiment_name}")
    
    def start_experiment(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Mulai eksperimen baru.
        
        Args:
            config: Konfigurasi eksperimen (opsional, gunakan config utama jika None)
        """
        self.experiment_tracker.start_experiment(config or self.config)
    
    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log metrik per epoch.
        
        Args:
            epoch: Nomor epoch
            train_loss: Training loss
            val_loss: Validation loss
            lr: Learning rate (opsional)
            additional_metrics: Metrik tambahan (opsional)
        """
        self.experiment_tracker.log_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            additional_metrics=additional_metrics
        )
    
    def end_experiment(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Akhiri eksperimen dan simpan hasil.
        
        Args:
            final_metrics: Metrik final (opsional)
        """
        self.experiment_tracker.end_experiment(final_metrics)
    
    def plot_metrics(self, save_to_file: bool = True) -> Any:
        """
        Plot metrik eksperimen.
        
        Args:
            save_to_file: Flag untuk simpan plot ke file
            
        Returns:
            Plot figure
        """
        return self.experiment_tracker.plot_metrics(save_to_file)
    
    def generate_report(self) -> str:
        """
        Generate laporan eksperimen.
        
        Returns:
            Path ke laporan
        """
        return self.experiment_tracker.generate_report()
    
    @staticmethod
    def list_experiments(output_dir: Optional[str] = None) -> List[str]:
        """
        Daftar semua eksperimen yang tersedia.
        
        Args:
            output_dir: Direktori eksperimen (opsional)
            
        Returns:
            List nama eksperimen
        """
        if output_dir is None:
            output_dir = "runs/train/experiments"
            
        return ExperimentTracker.list_experiments(output_dir)
    
    @staticmethod
    def compare_experiments(
        experiment_names: List[str], 
        output_dir: Optional[str] = None,
        save_to_file: bool = True
    ) -> Any:
        """
        Bandingkan beberapa eksperimen.
        
        Args:
            experiment_names: List nama eksperimen
            output_dir: Direktori eksperimen (opsional)
            save_to_file: Flag untuk simpan hasil ke file
            
        Returns:
            Plot figure
        """
        if output_dir is None:
            output_dir = "runs/train/experiments"
            
        return ExperimentTracker.compare_experiments(
            experiment_names=experiment_names,
            output_dir=output_dir,
            save_to_file=save_to_file
        )