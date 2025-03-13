# File: smartcash/handlers/model/core/model_trainer.py
# Deskripsi: Komponen untuk melatih model dengan dependency injection

import torch
import time
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

from smartcash.exceptions.base import ModelError, TrainingError
from smartcash.utils.training import TrainingPipeline
from smartcash.handlers.model.core.component_base import ComponentBase

class ModelTrainer(ComponentBase):
    """Komponen untuk melatih model dengan dependency injection."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional = None,
        model_factory = None
    ):
        """
        Inisialisasi model trainer.
        
        Args:
            config: Konfigurasi model dan training
            logger: Logger kustom (opsional)
            model_factory: ModelFactory instance (opsional)
        """
        super().__init__(config, logger, "model_trainer")
        
        # Dependencies
        self.model_factory = model_factory
        
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
        
        # Output directory
        self.output_dir = self.create_output_dir("weights")
    
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
            
            # Setup optimizer dan scheduler
            optimizer, scheduler = self._setup_optimizer_scheduler(model, **kwargs)
            
            # Buat training pipeline
            pipeline = TrainingPipeline(
                config=self.config, 
                model=model, 
                optimizer=optimizer,
                scheduler=scheduler,
                logger=self.logger
            )
            
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
            if checkpoint_path and self.model_factory:
                self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
                model, _ = self.model_factory.load_model(checkpoint_path)
            elif self.model_factory:
                self.logger.info("🔄 Membuat model baru")
                model = self.model_factory.create_model()
            else:
                raise ModelError("Model tidak tersedia dan model factory tidak diberikan")
        return model
    
    def _setup_optimizer_scheduler(self, model, **kwargs):
        """Setup optimizer dan scheduler untuk training."""
        # Ambil parameter dari config atau kwargs
        cfg = self.config.get('training', {})
        optimizer_type = kwargs.get('optimizer_type', cfg.get('optimizer', 'adam')).lower()
        lr = kwargs.get('lr', cfg.get('lr0', 0.001))
        weight_decay = kwargs.get('weight_decay', cfg.get('weight_decay', 0.0005))
        momentum = kwargs.get('momentum', cfg.get('momentum', 0.9))
        
        # Buat optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"⚠️ Optimizer {optimizer_type} tidak dikenal, menggunakan Adam")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            
        # Buat scheduler
        scheduler_type = kwargs.get('scheduler_type', cfg.get('scheduler', 'cosine')).lower()
        epochs = kwargs.get('epochs', cfg.get('epochs', 30))
        
        if scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', cfg.get('patience', 3)),
                verbose=True
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', cfg.get('lr_step_size', 10)),
                gamma=kwargs.get('gamma', cfg.get('lr_gamma', 0.1))
            )
        else:  # default to cosine
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
            
        return optimizer, scheduler
    
    def _prepare_params(self, kwargs):
        """Siapkan parameter training dengan default values."""
        params = {
            'num_epochs': kwargs.get('epochs', self.defaults['epochs']),
            'early_stopping': kwargs.get('early_stopping', self.defaults['early_stopping']),
            'early_stopping_patience': kwargs.get('early_stopping_patience', self.defaults['patience']),
            'save_every': kwargs.get('save_every', self.defaults['save_every']),
            'save_dir': kwargs.get('save_dir', str(self.output_dir)),
            'device': kwargs.get('device'),
            'resume_from_checkpoint': kwargs.get('checkpoint_path')
        }
        
        # Tambahkan semua parameter lain dari kwargs
        for k, v in kwargs.items():
            if k not in params:
                params[k] = v
                
        return params