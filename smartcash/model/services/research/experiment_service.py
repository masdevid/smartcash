"""
File: smartcash/model/services/research/experiment_service.py
Deskripsi: Facade untuk layanan eksperimen penelitian model
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.model.config import ModelConfig
from smartcash.model.config.experiment_config import ExperimentConfig
from smartcash.model.services.training import TrainingService
from smartcash.model.services.evaluation import EvaluationService
from smartcash.model.services.research.scenario_service import ScenarioService

# Import komponen-komponen eksperimen
from smartcash.model.services.research.experiment_creator import ExperimentCreator
from smartcash.model.services.research.experiment_runner import ExperimentRunner
from smartcash.model.services.research.experiment_analyzer import ExperimentAnalyzer
from smartcash.model.services.research.parameter_tuner import ParameterTuner
from smartcash.model.services.research.comparison_runner import ComparisonRunner


class ExperimentService:
    """
    Facade untuk layanan eksperimen penelitian model SmartCash.
    
    Menggabungkan berbagai komponen untuk:
    - Membuat dan mengelola eksperimen
    - Menjalankan eksperimen tunggal dan perbandingan
    - Analisis dan visualisasi hasil
    - Tuning parameter
    """
    
    def __init__(
        self,
        base_dir: str = "runs/experiments",
        config: Optional[ModelConfig] = None,
        logger: Optional[Any] = None,
    ):
        """
        Inisialisasi experiment service.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan hasil eksperimen
            config: Konfigurasi model dasar (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.logger = logger or get_logger()
        
        # Persiapkan direktori
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Konfigurasi dasar
        self.config = config or ModelConfig()
        
        # Layanan dasar
        training_service = TrainingService(logger=self.logger)
        evaluation_service = EvaluationService(logger=self.logger)
        scenario_service = ScenarioService(logger=self.logger)
        
        # Komponen eksperimen
        self.experiment_creator = ExperimentCreator(
            base_dir=base_dir,
            base_config=self.config,
            logger=self.logger
        )
        
        self.experiment_runner = ExperimentRunner(
            base_dir=base_dir,
            training_service=training_service,
            evaluation_service=evaluation_service,
            logger=self.logger
        )
        
        self.experiment_analyzer = ExperimentAnalyzer(
            base_dir=base_dir,
            logger=self.logger
        )
        
        self.parameter_tuner = ParameterTuner(
            base_dir=base_dir,
            experiment_runner=self.experiment_runner,
            experiment_creator=self.experiment_creator,
            logger=self.logger
        )
        
        self.comparison_runner = ComparisonRunner(
            base_dir=base_dir,
            experiment_runner=self.experiment_runner,
            experiment_creator=self.experiment_creator,
            logger=self.logger
        )
        
        self.logger.info(f"🧪 ExperimentService diinisialisasi (base_dir={self.base_dir})")
    
    # ----- Metode Facade untuk Eksperimen Dasar -----
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        config_overrides: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> ExperimentConfig:
        """Facade: Buat eksperimen baru."""
        return self.experiment_creator.create_experiment(name, description, config_overrides, tags)
    
    def run_experiment(
        self,
        experiment: Union[str, ExperimentConfig],
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        model_type: str = "efficient_basic",
        callbacks: List[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Facade: Jalankan eksperimen tunggal."""
        return self.experiment_runner.run_experiment(
            experiment,
            dataset_path,
            epochs,
            batch_size,
            learning_rate,
            model_type,
            callbacks,
            **kwargs
        )
    
    # ----- Metode Facade untuk Perbandingan -----
    
    def run_comparison_experiment(
        self,
        name: str,
        dataset_path: str,
        models_to_compare: List[str],
        epochs: int = 10,
        batch_size: int = 16,
        **kwargs
    ) -> Dict[str, Any]:
        """Facade: Jalankan eksperimen perbandingan antar model."""
        return self.comparison_runner.run_comparison_experiment(
            name,
            dataset_path,
            models_to_compare,
            epochs,
            batch_size,
            **kwargs
        )
    
    # ----- Metode Facade untuk Parameter Tuning -----
    
    def run_parameter_tuning(
        self,
        name: str,
        dataset_path: str,
        model_type: str,
        param_grid: Dict[str, List],
        **kwargs
    ) -> Dict[str, Any]:
        """Facade: Jalankan eksperimen tuning parameter."""
        return self.parameter_tuner.run_parameter_tuning(
            name,
            dataset_path,
            model_type,
            param_grid,
            **kwargs
        )
    
    # ----- Metode Facade untuk Analisis -----
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Facade: Dapatkan hasil eksperimen."""
        return self.experiment_analyzer.get_experiment_results(experiment_id)
    
    def list_experiments(self, filter_tags: List[str] = None) -> pd.DataFrame:
        """Facade: Dapatkan daftar semua eksperimen."""
        return self.experiment_analyzer.list_experiments(filter_tags)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """Facade: Bandingkan beberapa eksperimen."""
        return self.experiment_analyzer.compare_experiments(experiment_ids, metrics)
    
    def generate_experiment_report(
        self,
        experiment_id: str,
        include_plots: bool = True
    ) -> str:
        """Facade: Generate report untuk eksperimen."""
        return self.experiment_analyzer.generate_experiment_report(experiment_id, include_plots)