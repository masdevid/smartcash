# File: smartcash/handlers/model/model_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk proses training model menggunakan komponen utils/training

import os
import time
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any, Tuple
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.training import TrainingPipeline
from smartcash.utils.training.training_callbacks import TrainingCallback
from smartcash.handlers.checkpoint import CheckpointManager
from smartcash.handlers.model.model_factory import ModelFactory
from smartcash.handlers.model.optimizer_factory import OptimizerFactory

class ModelTrainer:
    """
    Handler untuk proses training model dengan menggunakan
    komponen dari utils/training yang telah direfaktor.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Inisialisasi model trainer.
        
        Args:
            config: Konfigurasi training
            logger: Custom logger (opsional)
            checkpoint_manager: Handler untuk manajemen checkpoint (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup factories
        self.model_factory = ModelFactory(config, logger)
        self.optimizer_factory = OptimizerFactory(config, logger)
        
        # Setup checkpoint handler
        if checkpoint_manager is None:
            checkpoints_dir = config.get('output_dir', 'runs/train') + '/weights'
            self.checkpoint_manager = CheckpointManager(
                output_dir=checkpoints_dir,
                logger=self.logger
            )
        else:
            self.checkpoint_manager = checkpoint_manager
            
        # Training pipeline
        self.training_pipeline = None
        
        # Track hasil training
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }
        
        self.logger.info(f"🔧 ModelTrainer diinisialisasi")
        
    def _initialize_training_pipeline(self, model: torch.nn.Module) -> None:
        """
        Inisialisasi TrainingPipeline dari utils/training.
        
        Args:
            model: Model yang akan dilatih
        """
        # Buat optimizer
        optimizer = self.optimizer_factory.create_optimizer(model)
        
        # Buat model handler for training pipeline
        model_handler = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': None  # Akan diset oleh TrainingPipeline
        }
        
        # Buat TrainingPipeline
        self.training_pipeline = TrainingPipeline(
            config=self.config,
            model_handler=model_handler,
            logger=self.logger
        )
        
        # Register custom callbacks
        self._register_training_callbacks()
    
    def _register_training_callbacks(self) -> None:
        """
        Mendaftarkan custom callbacks untuk training pipeline.
        """
        if self.training_pipeline is None:
            return
            
        # Callback untuk menyimpan history training
        def on_epoch_end(epoch, metrics, **kwargs):
            # Update training history
            self.training_history['train_loss'].append(metrics.get('train_loss', 0))
            self.training_history['val_loss'].append(metrics.get('val_loss', 0))
            
            # Simpan learning rate
            if 'learning_rate' in metrics:
                self.training_history['learning_rates'].append(metrics['learning_rate'])
            
            # Track metrik lainnya
            for key, value in metrics.items():
                if key not in ['train_loss', 'val_loss', 'learning_rate']:
                    if key not in self.training_history['metrics']:
                        self.training_history['metrics'][key] = []
                    self.training_history['metrics'][key].append(value)
                    
            # Plot progress jika epoch adalah kelipatan 5 atau epoch terakhir
            if (epoch + 1) % 5 == 0 or epoch == self.config.get('training', {}).get('epochs', 30) - 1:
                self._plot_training_progress()
        
        # Daftarkan callback
        self.training_pipeline.register_callback('epoch_end', on_epoch_end)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        resume_from: Optional[str] = None
    ) -> Dict:
        """
        Jalankan training loop untuk model menggunakan TrainingPipeline.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            epochs: Jumlah epoch (jika None, ambil dari config)
            model: Model yang akan dilatih (jika None, buat model baru)
            device: Device untuk training
            resume_from: Path checkpoint untuk resume training
            
        Returns:
            Dict berisi hasil training dan path ke checkpoint terbaik
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = epochs or self.config.get('training', {}).get('epochs', 30)
        
        # Reset history training
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }
        
        # Buat atau load model
        if model is None:
            if resume_from:
                # Load model dari checkpoint
                checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
                checkpoint_config = checkpoint.get('config', {})
                
                # Dapatkan informasi backbone dari checkpoint
                backbone = checkpoint_config.get('model', {}).get('backbone', 
                        self.config.get('model', {}).get('backbone', 'efficientnet'))
                
                # Buat model baru dengan konfigurasi yang sama dengan checkpoint
                model = self.model_factory.create_model(backbone_type=backbone)
                
                # Muat state_dict
                model.load_state_dict(checkpoint['model_state_dict'])
                
                self.logger.info(f"📋 Melanjutkan training dari checkpoint {resume_from}")
            else:
                model = self.model_factory.create_model()
            
        # Pindahkan model ke device
        model = model.to(device)
        
        # Inisialisasi training pipeline
        self._initialize_training_pipeline(model)
        
        # Jalankan training
        self.logger.info(
            f"🚀 Memulai training:\n"
            f"   • Epochs: {epochs}\n"
            f"   • Device: {device}\n"
            f"   • Early stopping patience: {self.config.get('training', {}).get('early_stopping_patience', 10)}"
        )
        
        # Siapkan dataloaders
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Training
        training_start = time.time()
        results = self.training_pipeline.train(
            dataloaders=dataloaders,
            epochs=epochs,
            resume_from_checkpoint=resume_from
        )
        
        # Hitung total waktu training
        total_time = time.time() - training_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.success(
            f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"   • Best val loss: {results.get('best_val_loss', 'N/A')}\n"
            f"   • Best checkpoint: {results.get('best_checkpoint_path', 'N/A')}"
        )
        
        # Gabungkan informasi training_history dengan results
        final_results = {
            **results,
            'history': self.training_history,
            'total_time': total_time
        }
        
        return final_results
    
    def _plot_training_progress(self) -> None:
        """Plot dan simpan grafik progres training."""
        try:
            # Buat direktori output jika belum ada
            output_dir = Path(self.config.get('output_dir', 'runs/train'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot training/validation loss
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.training_history['train_loss'], label='Train Loss')
            ax.plot(self.training_history['val_loss'], label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True)
            
            # Simpan plot
            fig.savefig(output_dir / 'training_loss.png')
            plt.close(fig)
            
            # Plot learning rate jika tersedia
            if self.training_history['learning_rates']:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.training_history['learning_rates'])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.grid(True)
                ax.set_yscale('log')
                
                # Simpan plot
                fig.savefig(output_dir / 'learning_rate.png')
                plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat plot training: {str(e)}")
            
    def resume_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        checkpoint_path: str,
        epochs: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Lanjutkan training dari checkpoint tertentu.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            checkpoint_path: Path ke checkpoint
            epochs: Jumlah epoch tambahan (jika None, ambil dari config)
            device: Device untuk training
            
        Returns:
            Dict berisi hasil training
        """
        return self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            resume_from=checkpoint_path
        )