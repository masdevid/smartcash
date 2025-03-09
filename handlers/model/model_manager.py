# File: smartcash/handlers/model/model_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama model sebagai facade untuk semua komponen model

import torch
import numpy as np
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError

class ModelManager:
    """
    Manager utama model sebagai facade.
    Menyembunyikan kompleksitas implementasi dan meningkatkan usability.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None
    ):
        """Inisialisasi model manager."""
        self.config = config
        self.logger = logger or get_logger("model_manager")
        self.colab_mode = self._detect_colab() if colab_mode is None else colab_mode
        
        # Dictionary untuk lazy-loaded components
        self._components = {}
        
        self.logger.info(
            f"🔧 ModelManager diinisialisasi (Colab: {self.colab_mode}, "
            f"Output: {self.config.get('output_dir', 'runs/train')})"
        )
    
    def _detect_colab(self) -> bool:
        """Deteksi apakah running di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _get_component(self, component_id: str, factory_func) -> Any:
        """
        Dapatkan komponen dengan lazy initialization.
        
        Args:
            component_id: ID unik untuk komponen
            factory_func: Fungsi factory untuk membuat komponen
            
        Returns:
            Komponen yang diminta
        """
        if component_id not in self._components:
            self._components[component_id] = factory_func()
        return self._components[component_id]
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        return self._get_component('model_factory', lambda: self._create_model_factory())
    
    def _create_model_factory(self):
        from smartcash.handlers.model.core.model_factory import ModelFactory
        return ModelFactory(self.config, self.logger)
    
    @property
    def trainer(self):
        """Lazy-loaded model trainer."""
        return self._get_component('trainer', lambda: self._create_trainer())
    
    def _create_trainer(self):
        from smartcash.handlers.model.core.model_trainer import ModelTrainer
        return ModelTrainer(self.config, self.logger)
    
    @property
    def evaluator(self):
        """Lazy-loaded model evaluator."""
        return self._get_component('evaluator', lambda: self._create_evaluator())
    
    def _create_evaluator(self):
        from smartcash.handlers.model.core.model_evaluator import ModelEvaluator
        return ModelEvaluator(self.config, self.logger)
    
    @property
    def predictor(self):
        """Lazy-loaded model predictor."""
        return self._get_component('predictor', lambda: self._create_predictor())
    
    def _create_predictor(self):
        from smartcash.handlers.model.core.model_predictor import ModelPredictor
        return ModelPredictor(self.config, self.logger)
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        return self._get_component('experiment_manager', lambda: self._create_experiment_manager())
    
    def _create_experiment_manager(self):
        from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager
        return ExperimentManager(self.config, self.logger)
    
    @property
    def optimizer_factory(self):
        """Lazy-loaded optimizer factory."""
        return self._get_component('optimizer_factory', lambda: self._create_optimizer_factory())
    
    def _create_optimizer_factory(self):
        from smartcash.handlers.model.core.optimizer_factory import OptimizerFactory
        return OptimizerFactory(self.config, self.logger)
    
    @property
    def checkpoint_adapter(self):
        """Lazy-loaded checkpoint adapter."""
        return self._get_component('checkpoint_adapter', lambda: self._create_checkpoint_adapter())
    
    def _create_checkpoint_adapter(self):
        from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
        return CheckpointAdapter(self.config, self.logger)
    
    @property
    def metrics_adapter(self):
        """Lazy-loaded metrics adapter."""
        return self._get_component('metrics_adapter', lambda: self._create_metrics_adapter())
    
    def _create_metrics_adapter(self):
        from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter
        return MetricsAdapter(self.logger, self.config)
    
    @property
    def environment_adapter(self):
        """Lazy-loaded environment adapter."""
        return self._get_component('environment_adapter', lambda: self._create_environment_adapter())
    
    def _create_environment_adapter(self):
        from smartcash.handlers.model.integration.environment_adapter import EnvironmentAdapter
        return EnvironmentAdapter(self.config, self.logger)
    
    @property
    def experiment_adapter(self):
        """Lazy-loaded experiment adapter."""
        return self._get_component('experiment_adapter', lambda: self._create_experiment_adapter())
    
    def _create_experiment_adapter(self):
        from smartcash.handlers.model.integration.experiment_adapter import ExperimentAdapter
        return ExperimentAdapter(self.config, self.logger)
    
    @property
    def exporter_adapter(self):
        """Lazy-loaded exporter adapter."""
        return self._get_component('exporter_adapter', lambda: self._create_exporter_adapter())
    
    def _create_exporter_adapter(self):
        from smartcash.handlers.model.integration.exporter_adapter import ExporterAdapter
        return ExporterAdapter(self.config, self.logger)
    
    # ==== Core Functionality ====
    
    def create_model(self, backbone_type=None, **kwargs) -> torch.nn.Module:
        """Buat model baru dengan konfigurasi tertentu."""
        return self.model_factory.create_model(backbone_type=backbone_type, **kwargs)
    
    def load_model(self, checkpoint_path, **kwargs) -> Tuple[torch.nn.Module, Dict]:
        """Load model dari checkpoint."""
        return self.model_factory.load_model(checkpoint_path, **kwargs)
    
    def train(self, train_loader, val_loader, model=None, **kwargs) -> Dict:
        """Train model dengan dataset yang diberikan."""
        # Setup observers untuk Colab jika perlu
        self._setup_colab_observers(kwargs)
        return self.trainer.train(train_loader=train_loader, val_loader=val_loader, model=model, **kwargs)
    
    def evaluate(self, test_loader, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """Evaluasi model pada test dataset."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        return self.evaluator.evaluate(test_loader=test_loader, model=model, checkpoint_path=checkpoint_path, **kwargs)
    
    def predict(self, images, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """Prediksi dengan model."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        return self.predictor.predict(images=images, model=model, checkpoint_path=checkpoint_path, **kwargs)
    
    def predict_on_video(self, video_path, model=None, checkpoint_path=None, **kwargs) -> str:
        """Prediksi pada video dengan visualisasi hasil."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        return self.predictor.predict_on_video(video_path=video_path, model=model, **kwargs)
    
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs) -> Dict:
        """Bandingkan beberapa backbone dengan kondisi yang sama."""
        # Setup Colab observer jika perlu
        self._setup_colab_observers(kwargs)
        return self.experiment_manager.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
    
    def setup_environment(self, use_drive=True, create_symlinks=True) -> Dict[str, Any]:
        """Setup environment project."""
        return self.environment_adapter.setup_project_environment(
            use_drive=use_drive,
            create_symlinks=create_symlinks
        )
    
    def export_model(self, model=None, checkpoint_path=None, format='torchscript', **kwargs) -> Optional[str]:
        """Export model ke format untuk deployment."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        # Export model sesuai format
        if format.lower() == 'torchscript':
            return self.exporter_adapter.export_to_torchscript(model=model, **kwargs)
        elif format.lower() == 'onnx':
            return self.exporter_adapter.export_to_onnx(model=model, **kwargs)
        else:
            self.logger.error(f"❌ Format export '{format}' tidak didukung")
            return None
    
    def track_experiment(self, experiment_name=None, config=None) -> None:
        """Mulai tracking eksperimen baru."""
        if experiment_name:
            self.experiment_adapter.set_experiment_name(experiment_name)
        self.experiment_adapter.start_experiment(config or self.config)
        self.logger.info(f"🧪 Mulai tracking eksperimen: {self.experiment_adapter.experiment_name}")
    
    def log_experiment_metrics(self, epoch, train_loss, val_loss, lr=None, additional_metrics=None) -> None:
        """Log metrik eksperimen."""
        self.experiment_adapter.log_metrics(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss, 
            lr=lr, additional_metrics=additional_metrics
        )
    
    def end_experiment(self, final_metrics=None) -> str:
        """Akhiri eksperimen dan generate laporan."""
        self.experiment_adapter.end_experiment(final_metrics)
        return self.experiment_adapter.generate_report()
    
    def compare_experiments(self, experiment_names=None, save_to_file=True) -> Any:
        """Bandingkan beberapa eksperimen."""
        if experiment_names is None:
            experiment_names = self.experiment_adapter.list_experiments()
        
        if not experiment_names:
            self.logger.warning("⚠️ Tidak ada eksperimen untuk dibandingkan")
            return None
        
        return self.experiment_adapter.compare_experiments(
            experiment_names=experiment_names,
            save_to_file=save_to_file
        )
    
    # ==== Helper Methods ====
    
    def _setup_colab_observers(self, kwargs):
        """Setup observer untuk Colab jika perlu."""
        if self.colab_mode and 'observers' not in kwargs:
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            kwargs['observers'] = [ColabObserver(self.logger)]
        elif self.colab_mode and isinstance(kwargs.get('observers', []), list):
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            # Tambahkan ColabObserver jika belum ada
            if not any(isinstance(obs, ColabObserver) for obs in kwargs['observers']):
                kwargs['observers'].append(ColabObserver(self.logger))
    
    def _ensure_model_or_checkpoint(self, model, checkpoint_path):
        """Pastikan ada model atau checkpoint path."""
        if model is None and checkpoint_path is None:
            checkpoint_path = self.checkpoint_adapter.find_best_checkpoint()
            if checkpoint_path is None:
                raise ModelError("Tidak ada model yang diberikan, dan tidak ada checkpoint yang ditemukan")
        
        if model is None and checkpoint_path is not None:
            model, _ = self.load_model(checkpoint_path)
            checkpoint_path = None  # Sudah di-load, tidak perlu lagi
        
        return model, checkpoint_path