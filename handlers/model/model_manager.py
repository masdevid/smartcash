# File: smartcash/handlers/model/model_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama model sebagai facade untuk semua komponen model (diringkas)

import torch
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
            f"🔧 ModelManager diinisialisasi (Colab: {self.colab_mode})"
        )
    
    def _detect_colab(self) -> bool:
        """Deteksi apakah running di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _get_component(self, component_id: str, factory_func) -> Any:
        """Dapatkan komponen dengan lazy initialization."""
        if component_id not in self._components:
            self._components[component_id] = factory_func()
        return self._components[component_id]
    
    # ===== Component Properties =====
    
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
    def checkpoint_adapter(self):
        """Lazy-loaded checkpoint adapter."""
        return self._get_component('checkpoint_adapter', lambda: self._create_checkpoint_adapter())
    
    def _create_checkpoint_adapter(self):
        from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
        return CheckpointAdapter(self.config, self.logger)
    
    # ===== Core Functionality =====
    
    def create_model(self, backbone_type=None, **kwargs) -> torch.nn.Module:
        """Buat model baru dengan konfigurasi tertentu."""
        return self.model_factory.create_model(backbone_type=backbone_type, **kwargs)
    
    def load_model(self, checkpoint_path, **kwargs) -> Tuple[torch.nn.Module, Dict]:
        """Load model dari checkpoint."""
        return self.model_factory.load_model(checkpoint_path, **kwargs)
    
    def train(self, train_loader, val_loader, model=None, **kwargs) -> Dict:
        """Train model dengan dataset yang diberikan."""
        return self.trainer.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            model=model, 
            **kwargs
        )
    
    def evaluate(self, test_loader, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """Evaluasi model pada test dataset."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.evaluator.evaluate(
            test_loader=test_loader, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict(self, images, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """Prediksi dengan model."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict(
            images=images, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict_on_video(self, video_path, model=None, checkpoint_path=None, **kwargs) -> str:
        """Prediksi pada video dengan visualisasi hasil."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict_on_video(
            video_path=video_path, 
            model=model,
            **kwargs
        )
    
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs) -> Dict:
        """Bandingkan beberapa backbone dengan kondisi yang sama."""
        from smartcash.handlers.model.model_experiments import ModelExperiments
        
        experiments = ModelExperiments(self.config, self.logger)
        return experiments.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
    
    def export_model(self, model=None, checkpoint_path=None, format='torchscript', **kwargs) -> Optional[str]:
        """Export model ke format untuk deployment."""
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        # Import ExporterAdapter secara lazy
        from smartcash.handlers.model.integration.exporter_adapter import ExporterAdapter
        exporter = ExporterAdapter(self.config, self.logger)
        
        # Export model sesuai format
        if format.lower() == 'torchscript':
            return exporter.export_to_torchscript(model=model, **kwargs)
        elif format.lower() == 'onnx':
            return exporter.export_to_onnx(model=model, **kwargs)
        else:
            self.logger.error(f"❌ Format export '{format}' tidak didukung")
            return None
    
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