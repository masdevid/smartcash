"""
File: smartcash/utils/training/training_pipeline.py
Author: Alfrida Sabar
Deskripsi: Pipeline training yang teroptimasi dengan dukungan untuk berbagai callback dan metrik
"""

import os
import torch
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any, Tuple

from smartcash.utils.training.training_callbacks import TrainingCallbacks
from smartcash.utils.training.training_metrics import TrainingMetrics
from smartcash.utils.training.training_epoch import TrainingEpoch
from smartcash.utils.training.validation_epoch import ValidationEpoch

class TrainingPipeline:
    """
    Pipeline utama untuk proses training yang teroptimasi dengan dukungan:
    - Callback system untuk event tracking
    - Resume training dari checkpoint
    - Early stopping
    - Manajemen memory yang lebih baik
    """
    
    def __init__(
        self,
        config: Dict,
        model_handler: Any,
        data_manager: Any = None,
        logger: Any = None
    ):
        """
        Inisialisasi training pipeline.
        
        Args:
            config: Dictionary konfigurasi
            model_handler: Instance dari ModelHandler
            data_manager: Instance dari DataManager (opsional)
            logger: Logger untuk mencatat aktivitas
        """
        self.config = config
        self.model_handler = model_handler
        self.data_manager = data_manager
        self.logger = logger
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tracking flags
        self._stop_training = False
        self._training_active = False
        
        # Komponen terkait training
        self.metrics = TrainingMetrics(logger)
        self.callbacks = TrainingCallbacks(logger)
        self.train_epoch = TrainingEpoch(logger)
        self.val_epoch = ValidationEpoch(logger)
        
        # Siapkan output dir
        self.output_dir = Path(config.get('output_dir', 'runs/train'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup custom logging
        self._setup_logging()
        
        if self.logger:
            self.log_message("🚀 Training pipeline siap")
    
    def _setup_logging(self):
        """Setup custom logging untuk pipeline."""
        # Jika logger tidak diberikan, buat logger dummy
        if self.logger is None:
            class DummyLogger:
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
                def success(self, msg): print(f"SUCCESS: {msg}")
                
            self.logger = DummyLogger()
    
    def log_message(self, msg):
        """Fungsi umum untuk logging yang dapat menangani berbagai jenis logger."""
        if hasattr(self.logger, 'info'):
            self.logger.info(msg)
        elif hasattr(self.logger, 'log_info'):
            self.logger.log_info(msg)
        else:
            print(f"INFO: {msg}")
    
    def log_warning(self, msg):
        """Fungsi umum untuk warning yang dapat menangani berbagai jenis logger."""
        if hasattr(self.logger, 'warning'):
            self.logger.warning(msg)
        elif hasattr(self.logger, 'log_warning'):
            self.logger.log_warning(msg)
        else:
            print(f"WARNING: {msg}")
    
    def log_error(self, msg):
        """Fungsi umum untuk error yang dapat menangani berbagai jenis logger."""
        if hasattr(self.logger, 'error'):
            self.logger.error(msg)
        elif hasattr(self.logger, 'log_error'):
            self.logger.log_error(msg)
        else:
            print(f"ERROR: {msg}")
    
    def log_success(self, msg):
        """Fungsi umum untuk success yang dapat menangani berbagai jenis logger."""
        if hasattr(self.logger, 'success'):
            self.logger.success(msg)
        elif hasattr(self.logger, 'log_success'):
            self.logger.log_success(msg)
        else:
            print(f"SUCCESS: {msg}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback function untuk event tertentu.
        
        Args:
            event: Nama event ('batch_end', 'epoch_end', etc.)
            callback: Fungsi callback
        """
        self.callbacks.register(event, callback)
    
    def stop_training(self) -> None:
        """Set flag untuk menghentikan training."""
        self._stop_training = True
        self.log_message("⏹️ Sinyal stop training diterima, menghentikan di akhir epoch")
    
    def get_training_status(self) -> Dict:
        """
        Dapatkan status training saat ini.
        
        Returns:
            Dictionary berisi status training
        """
        if self._training_active:
            # Hitung progres training
            if len(self.metrics.get_history('train_loss')) > 0:
                current_epoch = len(self.metrics.get_history('train_loss'))
                total_epochs = self.config.get('training', {}).get('epochs', 30)
                progress = (current_epoch / total_epochs) * 100
                
                # Hitung ETA
                if current_epoch > 0 and hasattr(self, 'epoch_times') and len(self.epoch_times) > 0:
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    remaining_epochs = total_epochs - current_epoch
                    eta_seconds = avg_epoch_time * remaining_epochs
                    
                    if eta_seconds < 60:
                        eta = f"{eta_seconds:.1f} detik"
                    elif eta_seconds < 3600:
                        eta = f"{eta_seconds/60:.1f} menit"
                    else:
                        eta = f"{eta_seconds/3600:.1f} jam"
                else:
                    eta = "Menghitung..."
                
                # Dapatkan best validation loss
                val_losses = self.metrics.get_history('val_loss')
                best_val_loss = min(val_losses) if val_losses else float('inf')
                
                return {
                    'status': 'training',
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'progress': progress,
                    'best_val_loss': best_val_loss,
                    'estimated_time_remaining': eta,
                    'message': 'Training aktif'
                }
            else:
                return {
                    'status': 'training',
                    'message': 'Training baru dimulai'
                }
        else:
            return {
                'status': 'idle',
                'message': 'Pipeline siap'
            }
    
    def update_dataloaders(self, batch_size: int = None, num_workers: int = None) -> Dict:
        """
        Update dataloader dengan batch size atau num_workers yang baru.
        
        Args:
            batch_size: Batch size baru (opsional)
            num_workers: Jumlah workers baru (opsional)
            
        Returns:
            Dictionary berisi dataloader yang diperbarui
        """
        if self.data_manager is None:
            self.log_warning("⚠️ Data manager tidak tersedia untuk update dataloader")
            return {}
        
        # Ambil parameter dari config jika tidak disediakan
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 16)
        
        if num_workers is None:
            num_workers = self.config.get('model', {}).get('workers', 4)
        
        # Perbarui config
        if 'training' not in self.config:
            self.config['training'] = {}
        self.config['training']['batch_size'] = batch_size
        
        if 'model' not in self.config:
            self.config['model'] = {}
        self.config['model']['workers'] = num_workers
        
        # Buat dataloader baru
        dataloaders = {
            'train': self.data_manager.get_train_loader(
                batch_size=batch_size,
                num_workers=num_workers
            ),
            'val': self.data_manager.get_val_loader(
                batch_size=batch_size,
                num_workers=num_workers
            ),
            'test': self.data_manager.get_test_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
        }
        
        self.log_message(f"🔄 Dataloader diperbarui: batch_size={batch_size}, num_workers={num_workers}")
        return dataloaders
    
    def train(
        self,
        dataloaders: Optional[Dict] = None,
        resume_from_checkpoint: Optional[str] = None,
        save_every: int = 5,
        epochs: Optional[int] = None
    ) -> Dict:
        """
        Jalankan training loop.
        
        Args:
            dataloaders: Dictionary berisi dataloaders ('train', 'val')
            resume_from_checkpoint: Path ke checkpoint untuk melanjutkan training
            save_every: Simpan checkpoint setiap n epoch
            epochs: Jumlah epoch (jika None, ambil dari config)
            
        Returns:
            Dictionary berisi hasil training
        """
        self._stop_training = False
        self._training_active = True
        
        # Setup parameter
        if epochs is None:
            epochs = self.config.get('training', {}).get('epochs', 30)
        
        # Gunakan dataloader default jika tidak disediakan
        if dataloaders is None:
            dataloaders = self._setup_default_dataloaders()
        
        # Validasi dataloaders
        if 'train' not in dataloaders or 'val' not in dataloaders:
            self.log_error("❌ Dataloaders tidak lengkap, harus ada 'train' dan 'val'")
            return {'success': False, 'message': 'Dataloaders tidak lengkap'}
        
        # Siapkan model
        try:
            # Persiapkan model, optimizer dan scheduler
            model, optimizer, scheduler, start_epoch = self._setup_model_components(
                resume_from_checkpoint
            )
            
            # Setup early stopping
            early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)
            best_val_loss = float('inf')
            early_stopping_counter = 0
            best_model_state = None
            
            # Track epoch times untuk estimasi
            self.epoch_times = []
            
            # Mulai training loop
            self.log_message(f"🚀 Memulai training untuk {epochs} epochs")
            
            for epoch in range(start_epoch, epochs):
                if self._stop_training:
                    self.log_message("⏹️ Training dihentikan oleh user")
                    break
                
                epoch_start_time = time.time()
                
                # Training phase
                model.train()
                train_loss = self.train_epoch.run(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    train_loader=dataloaders['train'],
                    device=self.device,
                    callbacks=self.callbacks
                )
                
                # Validation phase
                model.eval()
                val_loss, val_metrics = self.val_epoch.run(
                    epoch=epoch,
                    model=model,
                    val_loader=dataloaders['val'],
                    device=self.device,
                    callbacks=self.callbacks
                )
                
                # Update learning rate scheduler
                self._update_scheduler(scheduler, val_loss)
                
                # Dapatkan learning rate terkini
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update training history
                self.metrics.update_history('train_loss', train_loss)
                self.metrics.update_history('val_loss', val_loss)
                self.metrics.update_history('learning_rates', current_lr)
                
                # Track metrik lainnya
                for key, value in val_metrics.items():
                    self.metrics.update_history(key, value)
                
                # Hitung waktu epoch
                epoch_time = time.time() - epoch_start_time
                self.epoch_times.append(epoch_time)
                
                # Log hasil epoch
                self.log_message(
                    f"📊 Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Log metrics ke file CSV
                self.metrics.log_to_csv(
                    epoch=epoch,
                    metrics={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': current_lr,
                        **val_metrics
                    },
                    output_dir=self.output_dir
                )
                
                # Trigger callback dengan metrics
                self.callbacks.trigger(
                    event='epoch_end',
                    epoch=epoch,
                    metrics={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': current_lr,
                        **val_metrics
                    }
                )
                
                # Simpan checkpoint
                early_stopping_updated = self._handle_checkpoints(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    save_every=save_every,
                    best_val_loss=best_val_loss
                )
                
                # Update early stopping state
                if early_stopping_updated:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_counter >= early_stopping_patience:
                    self.log_message(
                        f"⏹️ Early stopping diaktifkan setelah {early_stopping_counter} "
                        f"epoch tanpa peningkatan"
                    )
                    break
                
                # Memory cleanup after each epoch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Selesai training
            self._training_active = False
            self.callbacks.trigger(
                event='training_end',
                epochs_trained=epoch - start_epoch + 1,
                best_val_loss=best_val_loss,
                training_history=self.metrics.get_all_history()
            )
            
            self.log_success(
                f"✅ Training selesai: {epoch - start_epoch + 1} epochs, "
                f"Best val loss: {best_val_loss:.4f}"
            )
            
            # Kembalikan hasil training
            return {
                'success': True,
                'epochs_trained': epoch - start_epoch + 1,
                'best_val_loss': best_val_loss,
                'training_history': self.metrics.get_all_history(),
                'early_stopped': early_stopping_counter >= early_stopping_patience
            }
            
        except KeyboardInterrupt:
            self.log_warning("⚠️ Training dihentikan oleh keyboard interrupt")
            self._training_active = False
            return {
                'success': False,
                'message': 'Training dihentikan oleh keyboard interrupt',
                'training_history': self.metrics.get_all_history()
            }
        except Exception as e:
            self.log_error(f"❌ Error saat training: {str(e)}")
            self._training_active = False
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error training: {str(e)}',
                'training_history': self.metrics.get_all_history()
            }
    
    def _setup_default_dataloaders(self) -> Dict:
        """Setup dataloader default"""
        if self.data_manager is None:
            self.log_error("❌ Tidak ada data_manager tersedia")
            return {}
            
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        num_workers = self.config.get('model', {}).get('workers', 4)
        
        return {
            'train': self.data_manager.get_train_loader(
                batch_size=batch_size,
                num_workers=num_workers
            ),
            'val': self.data_manager.get_val_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
        }
    
    def _setup_model_components(self, resume_from_checkpoint: Optional[str] = None) -> Tuple:
        """Setup model, optimizer, scheduler, dan epoch awal"""
        # Buat model atau resume dari checkpoint
        if resume_from_checkpoint:
            self.log_message(f"🔄 Melanjutkan training dari checkpoint: {resume_from_checkpoint}")
            model, checkpoint = self.model_handler.load_model(resume_from_checkpoint)
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            # Restore training history jika tersedia
            if 'training_history' in checkpoint:
                self.metrics.restore_history(checkpoint['training_history'])
        else:
            model = self.model_handler.create_model()
            start_epoch = 0
        
        # Pindahkan model ke device
        model = model.to(self.device)
        
        # Buat optimizer dan scheduler
        optimizer = self.model_handler.get_optimizer(model)
        scheduler = self.model_handler.get_scheduler(optimizer)
        
        return model, optimizer, scheduler, start_epoch
    
    def _update_scheduler(self, scheduler, val_loss: float) -> None:
        """Update learning rate scheduler"""
        if scheduler is None:
            return
            
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
    
    def _handle_checkpoints(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        val_loss: float,
        val_metrics: Dict,
        save_every: int,
        best_val_loss: float
    ) -> bool:
        """Handle checkpoint saving and early stopping update"""
        is_best = val_loss < best_val_loss
        
        # Simpan checkpoint secara periodik atau jika terbaik
        if is_best or epoch % save_every == 0:
            checkpoint_info = self.model_handler.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'loss': val_loss,
                    **val_metrics,
                    'best_val_loss': best_val_loss if not is_best else val_loss
                },
                is_best=is_best,
                training_history=self.metrics.get_all_history()
            )
            
            self.callbacks.trigger(
                event='checkpoint_saved',
                epoch=epoch,
                checkpoint_info=checkpoint_info,
                is_best=is_best
            )
        
        return is_best