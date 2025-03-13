# File: smartcash/handlers/model/model_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama model sebagai facade, direfaktor untuk konsistensi dan DRY

import torch
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.core.model_component import ModelComponent

class ModelManager(ModelComponent):
    """
    Manager utama model sebagai facade.
    Menyembunyikan kompleksitas implementasi dan meningkatkan usability.
    Direfaktor untuk menggunakan ModelComponent base class.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None
    ):
        """
        Inisialisasi model manager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            colab_mode: Flag untuk mode Colab (opsional, auto-detect jika None)
        """
        super().__init__(config, logger, "model_manager")
        self.colab_mode = self._is_running_in_colab() if colab_mode is None else colab_mode
    
    def process(self, *args, **kwargs) -> Any:
        """
        Proses default adalah create_model.
        
        Returns:
            Model yang dibuat
        """
        return self.create_model(*args, **kwargs)
    
    # ===== Lazy-loaded Component Properties =====
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        from smartcash.handlers.model.core.model_factory import ModelFactory
        return self.get_component('model_factory', lambda: ModelFactory(self.config, self.logger))
    
    @property
    def trainer(self):
        """Lazy-loaded model trainer."""
        from smartcash.handlers.model.core.model_trainer import ModelTrainer
        return self.get_component('trainer', lambda: ModelTrainer(self.config, self.logger))
    
    @property
    def evaluator(self):
        """Lazy-loaded model evaluator."""
        from smartcash.handlers.model.core.model_evaluator import ModelEvaluator
        return self.get_component('evaluator', lambda: ModelEvaluator(self.config, self.logger))
    
    @property
    def predictor(self):
        """Lazy-loaded model predictor."""
        from smartcash.handlers.model.core.model_predictor import ModelPredictor
        return self.get_component('predictor', lambda: ModelPredictor(self.config, self.logger))
    
    @property
    def checkpoint_adapter(self):
        """Lazy-loaded checkpoint adapter."""
        from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
        return self.get_component('checkpoint_adapter', lambda: CheckpointAdapter(self.config, self.logger))
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager
        return self.get_component('experiment_manager', lambda: ExperimentManager(self.config, self.logger))
    
    # ===== Core Functionality =====
    
    def create_model(self, backbone_type=None, **kwargs) -> torch.nn.Module:
        """
        Buat model baru dengan konfigurasi tertentu.
        
        Args:
            backbone_type: Tipe backbone (optional)
            **kwargs: Parameter tambahan untuk model
            
        Returns:
            Model yang siap digunakan
        """
        return self.model_factory.create_model(backbone_type=backbone_type, **kwargs)
    
    def load_model(self, checkpoint_path, **kwargs) -> Tuple[torch.nn.Module, Dict]:
        """
        Load model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            **kwargs: Parameter tambahan
            
        Returns:
            Tuple (Model yang dimuat, Metadata checkpoint)
        """
        return self.model_factory.load_model(checkpoint_path, **kwargs)
    
    def train(self, train_loader, val_loader, model=None, **kwargs) -> Dict:
        """
        Train model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model untuk dilatih (opsional, akan dibuat jika None)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil training
        """
        return self.trainer.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            model=model, 
            **kwargs
        )
    
    def evaluate(self, test_loader, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil evaluasi
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.evaluator.evaluate(
            test_loader=test_loader, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict(self, images, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """
        Prediksi dengan model.
        
        Args:
            images: Input gambar untuk prediksi
            model: Model untuk prediksi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil prediksi
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict(
            images=images, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict_on_video(self, video_path, model=None, checkpoint_path=None, **kwargs) -> str:
        """
        Prediksi pada video dengan visualisasi hasil.
        
        Args:
            video_path: Path ke file video
            model: Model untuk prediksi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke video hasil
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict_on_video(
            video_path=video_path, 
            model=model,
            **kwargs
        )
    
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs) -> Dict:
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        return self.experiment_manager.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
    
    def export_model(self, model=None, checkpoint_path=None, format='torchscript', **kwargs) -> Optional[str]:
        """
        Export model ke format untuk deployment.
        
        Args:
            model: Model untuk export (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            format: Format export ('torchscript', 'onnx')
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke model yang diexport
        """
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
        """
        Pastikan ada model atau checkpoint path.
        
        Args:
            model: Model yang sudah ada (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            
        Returns:
            Tuple (model, checkpoint_path)
            
        Raises:
            ModelError: Jika tidak ada model dan tidak ada checkpoint
        """
        if model is None and checkpoint_path is None:
            checkpoint_path = self.checkpoint_adapter.find_best_checkpoint()
            if checkpoint_path is None:
                raise ModelError("Tidak ada model yang diberikan, dan tidak ada checkpoint yang ditemukan")
        
        if model is None and checkpoint_path is not None:
            model, _ = self.load_model(checkpoint_path)
            checkpoint_path = None  # Sudah di-load, tidak perlu lagi
        
        return model, checkpoint_path