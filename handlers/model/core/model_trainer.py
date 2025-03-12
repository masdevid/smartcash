# File: smartcash/handlers/model/core/model_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk melatih model dengan implementasi minimal

import torch
import time
from typing import Dict, Optional, Any, List
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError, TrainingError
from smartcash.utils.training import TrainingPipeline

class ModelTrainer(ModelComponent):
    """Komponen untuk melatih model dengan pendekatan standar."""
    
    def _initialize(self):
        """Inisialisasi parameter training default."""
        cfg = self.config.get('training', {})
        self.defaults = {
            'epochs': cfg.get('epochs', 30),
            'batch_size': cfg.get('batch_size', 16),
            'early_stopping': cfg.get('early_stopping_patience', 10) > 0,
            'patience': cfg.get('early_stopping_patience', 10),
            'save_every': cfg.get('save_period', 5)
        }
        self.in_colab = self._is_colab()
        
    def process(self, train_loader, val_loader, **kwargs):
        """Alias untuk train()."""
        return self.train(train_loader, val_loader, **kwargs)
    
    def train(
        self,
        train_loader,
        val_loader,
        model=None,
        checkpoint_path=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Latih model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model yang akan dilatih (opsional)
            checkpoint_path: Path untuk resume training
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil training
        """
        start_time = time.time()
        
        try:
            # Siapkan model
            model = self._prepare_model(model, checkpoint_path)
            device = kwargs.get('device') or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Buat training pipeline
            pipeline = TrainingPipeline(config=self.config, model_handler=model, logger=self.logger)
            
            # Parameter training
            train_params = self._prepare_params(kwargs)
            
            # Jalankan training
            results = pipeline.train(
                dataloaders={'train': train_loader, 'val': val_loader},
                **train_params
            )
          
            
            # Log hasil
            duration = time.time() - start_time
            h, m = int(duration // 3600), int((duration % 3600) // 60)
            self.logger.success(
                f"✅ Training selesai dalam {h}h {m}m\n"
                f"   • Epochs: {results.get('epoch', 0)}/{train_params.get('num_epochs', 0)}\n"
                f"   • Best val loss: {results.get('best_val_loss', 'N/A'):.4f}"
            )
            
            results['execution_time'] = duration
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error training: {str(e)}")
            raise TrainingError(f"Gagal training: {str(e)}")
    
    def _prepare_model(self, model, checkpoint_path):
        """Persiapkan model untuk training."""
        if model is None:
            if checkpoint_path:
                self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
                model, _ = self.model_factory.load_model(checkpoint_path)
            else:
                self.logger.info("🔄 Membuat model baru")
                model = self.model_factory.create_model()
        return model
    
    def _prepare_params(self, kwargs):
        """Siapkan parameter training dengan default values."""
        params = {
            'num_epochs': kwargs.get('epochs', self.defaults['epochs']),
            'early_stopping': kwargs.get('early_stopping', self.defaults['early_stopping']),
            'early_stopping_patience': kwargs.get('early_stopping_patience', self.defaults['patience']),
            'save_every': kwargs.get('save_every', self.defaults['save_every']),
            'resume_from_checkpoint': kwargs.get('checkpoint_path'),
            'device': kwargs.get('device')
        }
        
        # Tambahkan optimizer dan scheduler jika ada
        if 'optimizer' in kwargs:
            params['optimizer'] = kwargs['optimizer']
        if 'scheduler' in kwargs:
            params['scheduler'] = kwargs['scheduler']
            
        # Tambahkan semua parameter lain dari kwargs
        for k, v in kwargs.items():
            if k not in params:
                params[k] = v
                
            return params