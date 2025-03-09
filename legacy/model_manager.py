# File: smartcash/handlers/model_manager.py
# Author: Alfrida Sabar
# Deskripsi: Entry point untuk model training, evaluasi, dan prediksi

import torch
from typing import Dict, Optional, List, Union, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.model.model_factory import ModelFactory
from smartcash.handlers.model.optimizer_factory import OptimizerFactory
from smartcash.handlers.model.model_trainer import ModelTrainer
from smartcash.handlers.model.model_evaluator import ModelEvaluator
from smartcash.handlers.model.model_predictor import ModelPredictor
from smartcash.handlers.model.model_experiments import ModelExperiments
from smartcash.handlers.checkpoint import CheckpointManager

class ModelManager:
    """
    Entry point utama untuk semua operasi terkait model di SmartCash.
    
    ModelManager adalah facade yang menyediakan akses terstruktur ke berbagai 
    komponen model yang telah direfaktorisasi, menjadikannya entry point 
    tunggal untuk training, evaluasi, dan prediksi.
    """
    
    def __init__(
        self,
        config: Dict,
        config_path: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi model manager.
        
        Args:
            config: Konfigurasi model dan training
            config_path: Path ke file konfigurasi (opsional)
            logger: Custom logger (opsional)
        """
        self.config = config
        self.config_path = config_path
        self.logger = logger or get_logger("model_manager")
        
        # Setup checkpoint handler
        checkpoints_dir = config.get('output_dir', 'runs/train') + '/weights'
        self.checkpoint_manager = CheckpointManager(
            output_dir=checkpoints_dir,
            logger=self.logger
        )
        
        # Inisialisasi komponen
        self.model_factory = ModelFactory(config, self.logger)
        self.optimizer_factory = OptimizerFactory(config, self.logger)
        self.trainer = ModelTrainer(config, self.logger, self.checkpoint_manager)
        self.evaluator = ModelEvaluator(config, self.logger, self.checkpoint_manager)
        self.predictor = ModelPredictor(config, self.logger, self.checkpoint_manager)
        self.experiments = ModelExperiments(config, self.logger)
        
        self.logger.info(
            f"🚀 ModelManager diinisialisasi sebagai entry point untuk semua operasi model\n"
            f"   • Backbone: {config.get('model', {}).get('backbone', 'efficientnet')}\n"
            f"   • Checkpoints: {checkpoints_dir}"
        )
    
    def create_model(self, backbone_type: Optional[str] = None):
        """
        Buat model dengan backbone tertentu.
        
        Args:
            backbone_type: Tipe backbone (opsional)
            
        Returns:
            Model yang dibuat
        """
        return self.model_factory.create_model(backbone_type)
    
    def train(self, train_loader, val_loader, **kwargs):
        """
        Lakukan training model.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            **kwargs: Parameter tambahan untuk trainer
            
        Returns:
            Hasil training
        """
        return self.trainer.train(train_loader, val_loader, **kwargs)
    
    def evaluate(self, test_loader, **kwargs):
        """
        Evaluasi model pada dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            **kwargs: Parameter tambahan untuk evaluator
            
        Returns:
            Hasil evaluasi
        """
        return self.evaluator.evaluate(test_loader, **kwargs)
    
    def predict(self, images, **kwargs):
        """
        Lakukan prediksi dengan model.
        
        Args:
            images: Input gambar untuk prediksi
            **kwargs: Parameter tambahan untuk predictor
            
        Returns:
            Hasil prediksi
        """
        return self.predictor.predict(images, **kwargs)
    
    def run_experiment(self, scenario, train_loader, val_loader, test_loader):
        """
        Jalankan eksperimen model.
        
        Args:
            scenario: Konfigurasi eksperimen
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            
        Returns:
            Hasil eksperimen
        """
        return self.experiments.run_experiment(scenario, train_loader, val_loader, test_loader)
    
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader):
        """
        Bandingkan performa beberapa backbone.
        
        Args:
            backbones: List backbone yang akan dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            
        Returns:
            Hasil perbandingan
        """
        return self.experiments.compare_backbones(backbones, train_loader, val_loader, test_loader)
    
    def tune_hyperparameters(self, param_grid, train_loader, val_loader, test_loader, **kwargs):
        """
        Lakukan hyperparameter tuning.
        
        Args:
            param_grid: Grid parameter yang akan dituning
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil tuning
        """
        return self.experiments.hyperparameter_tuning(param_grid, train_loader, val_loader, test_loader, **kwargs)
    
    def load_model(self, checkpoint_path=None, device=None):
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint (opsional)
            device: Device untuk model (opsional)
            
        Returns:
            Model yang dimuat
        """
        return self.predictor.load_model(checkpoint_path, device)
    
    def list_checkpoints(self):
        """
        Dapatkan daftar checkpoint yang tersedia.
        
        Returns:
            Daftar checkpoint
        """
        return self.checkpoint_handler.list_checkpoints()
    
    def __repr__(self):
        return f"ModelManager(backbone={self.config.get('model', {}).get('backbone', 'efficientnet')})"