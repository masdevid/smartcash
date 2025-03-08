"""
File: smartcash/utils/training_pipeline.py
Author: Alfrida Sabar
Deskripsi: Pipeline training yang teroptimasi dengan dukungan callback, 
           logging yang lebih baik, dan manajemen checkpoint.
           Implementasi sederhana tanpa ketergantungan TensorBoard.
"""

import os
import time
import torch
import yaml
import numpy as np
import csv
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any, Tuple
import threading
import gc

class TrainingPipeline:
    """
    Pipeline untuk proses training yang teroptimasi dengan dukungan:
    - Callback system untuk event tracking
    - Resume training dari checkpoint
    - Early stopping
    - Learning rate schedulers
    - Metric tracking dan visualisasi CSV
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
        
        # Setup tracking metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }
        
        # Setup callbacks
        self.callbacks = {
            'batch_end': [],
            'epoch_end': [],
            'training_end': [],
            'validation_end': [],
            'checkpoint_saved': []
        }
        
        # Siapkan output dir
        self.output_dir = Path(config.get('output_dir', 'runs/train'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup custom logging
        self._setup_logging()
        
        if self.logger:
            self.log_message("🚀 Training pipeline siap")
    
    def _setup_logging(self):
        """Setup custom logging sebagai pengganti TensorBoard."""
        # Jika logger tidak diberikan, buat logger dummy
        if self.logger is None:
            class DummyLogger:
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
                def success(self, msg): print(f"SUCCESS: {msg}")
                
            self.logger = DummyLogger()
            
        # Buat direktori log jika belum ada
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Simpan metrics ke file CSV
        self.metrics_file = log_dir / 'metrics.csv'
        
        # Flag untuk mencatat header
        self.header_written = False
    
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
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            self.log_warning(f"⚠️ Event tidak didukung: {event}")
    
    def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """
        Jalankan callback untuk event tertentu.
        
        Args:
            event: Nama event
            **kwargs: Argumen untuk dikirimkan ke callback
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    self.log_warning(f"⚠️ Error pada callback {event}: {str(e)}")
    
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
            if len(self.training_history['train_loss']) > 0:
                current_epoch = len(self.training_history['train_loss'])
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
                val_losses = self.training_history['val_loss']
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
    
    def _log_metrics_to_csv(self, epoch: int, metrics: Dict) -> None:
        """
        Catat metrics ke file CSV.
        
        Args:
            epoch: Epoch saat ini
            metrics: Metrics yang akan dicatat
        """
        # Persiapkan data
        data = {'epoch': epoch}
        data.update(metrics)
        
        # Tentukan apakah file sudah ada
        file_exists = self.metrics_file.exists()
        
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                
                # Tulis header jika file baru
                if not file_exists:
                    writer.writeheader()
                
                # Tulis data
                writer.writerow(data)
        except Exception as e:
            self.log_warning(f"⚠️ Gagal menulis metrics ke CSV: {str(e)}")
    
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
            if self.data_manager is None:
                self.log_error("❌ Tidak ada dataloaders dan data_manager tersedia")
                return {'success': False, 'message': 'Tidak ada dataloaders tersedia'}
            
            batch_size = self.config.get('training', {}).get('batch_size', 16)
            num_workers = self.config.get('model', {}).get('workers', 4)
            
            dataloaders = {
                'train': self.data_manager.get_train_loader(
                    batch_size=batch_size,
                    num_workers=num_workers
                ),
                'val': self.data_manager.get_val_loader(
                    batch_size=batch_size,
                    num_workers=num_workers
                )
            }
        
        # Validasi dataloaders
        if 'train' not in dataloaders or 'val' not in dataloaders:
            self.log_error("❌ Dataloaders tidak lengkap, harus ada 'train' dan 'val'")
            return {'success': False, 'message': 'Dataloaders tidak lengkap'}
        
        # Siapkan model
        try:
            # Buat model atau resume dari checkpoint
            if resume_from_checkpoint:
                self.log_message(f"🔄 Melanjutkan training dari checkpoint: {resume_from_checkpoint}")
                model, checkpoint = self.model_handler.load_model(resume_from_checkpoint)
                start_epoch = checkpoint.get('epoch', 0) + 1
                
                # Restore training history jika tersedia
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']
            else:
                model = self.model_handler.create_model()
                start_epoch = 0
            
            # Pindahkan model ke device
            model = model.to(self.device)
            
            # Buat optimizer dan scheduler
            optimizer = self.model_handler.get_optimizer(model)
            scheduler = self.model_handler.get_scheduler(optimizer)
            
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
                train_loss = self._train_epoch(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    train_loader=dataloaders['train']
                )
                
                # Validation phase
                model.eval()
                val_loss, val_metrics = self._validate_epoch(
                    epoch=epoch,
                    model=model,
                    val_loader=dataloaders['val']
                )
                
                # Update learning rate scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                # Dapatkan learning rate terkini
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update training history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['learning_rates'].append(current_lr)
                
                # Track metrik lainnya
                for key, value in val_metrics.items():
                    if key not in self.training_history['metrics']:
                        self.training_history['metrics'][key] = []
                    self.training_history['metrics'][key].append(value)
                
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
                self._log_metrics_to_csv(
                    epoch=epoch,
                    metrics={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': current_lr,
                        **val_metrics
                    }
                )
                
                # Trigger callback dengan metrics
                self._trigger_callbacks(
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
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Simpan checkpoint secara periodik atau jika terbaik
                if is_best or epoch % save_every == 0 or epoch == epochs - 1:
                    checkpoint_info = self.model_handler.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics={
                            'loss': val_loss,
                            **val_metrics,
                            'best_val_loss': best_val_loss
                        },
                        is_best=is_best,
                        training_history=self.training_history
                    )
                    
                    self._trigger_callbacks(
                        event='checkpoint_saved',
                        epoch=epoch,
                        checkpoint_info=checkpoint_info,
                        is_best=is_best
                    )
                
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
            self._trigger_callbacks(
                event='training_end',
                epochs_trained=epoch - start_epoch + 1,
                best_val_loss=best_val_loss,
                training_history=self.training_history
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
                'training_history': self.training_history,
                'early_stopped': early_stopping_counter >= early_stopping_patience
            }
            
        except KeyboardInterrupt:
            self.log_warning("⚠️ Training dihentikan oleh keyboard interrupt")
            self._training_active = False
            return {
                'success': False,
                'message': 'Training dihentikan oleh keyboard interrupt',
                'training_history': self.training_history
            }
        except Exception as e:
            self.log_error(f"❌ Error saat training: {str(e)}")
            self._training_active = False
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error training: {str(e)}',
                'training_history': self.training_history
            }
    
    def _train_epoch(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Jalankan satu epoch training.
        
        Args:
            epoch: Nomor epoch
            model: Model yang akan ditraining
            optimizer: Optimizer
            train_loader: Dataloader untuk training
            
        Returns:
            Rata-rata loss untuk epoch ini
        """
        total_loss = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Handle berbagai format data
            if isinstance(data, dict):
                # Format multilayer dataset
                images = data['image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in data['targets'].items()}
            elif isinstance(data, tuple) and len(data) == 2:
                # Format (images, targets)
                images, targets = data
                images = images.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                self.log_warning(f"⚠️ Format data tidak didukung: {type(data)}")
                continue
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            if hasattr(model, 'compute_loss'):
                # Jika model memiliki metode compute_loss sendiri
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
            else:
                # Gunakan loss function default
                if isinstance(predictions, dict) and isinstance(targets, dict):
                    # Multilayer predictions
                    loss = 0
                    for layer_name in predictions:
                        if layer_name in targets:
                            layer_pred = predictions[layer_name]
                            layer_target = targets[layer_name]
                            loss += torch.nn.functional.mse_loss(layer_pred, layer_target)
                else:
                    # Single layer predictions
                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(predictions, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss tracking
            total_loss += loss.item()
            batch_count += 1
            
            # Trigger batch end callback
            self._trigger_callbacks(
                event='batch_end',
                epoch=epoch,
                batch=batch_idx,
                loss=loss.item(),
                batch_size=images.size(0)
            )
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / max(1, batch_count)  # Hindari division by zero
    
    def _validate_epoch(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict]:
        """
        Jalankan satu epoch validasi.
        
        Args:
            epoch: Nomor epoch
            model: Model yang akan divalidasi
            val_loader: Dataloader untuk validasi
            
        Returns:
            Tuple (rata-rata validation loss, dict metrik)
        """
        total_loss = 0
        batch_count = 0
        val_metrics = {}
        
        # Kumpulkan semua prediksi dan target untuk menghitung metrik
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # Handle berbagai format data
                if isinstance(data, dict):
                    # Format multilayer dataset
                    images = data['image'].to(self.device)
                    targets = {k: v.to(self.device) for k, v in data['targets'].items()}
                elif isinstance(data, tuple) and len(data) == 2:
                    # Format (images, targets)
                    images, targets = data
                    images = images.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)
                    elif isinstance(targets, dict):
                        targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    continue
                
                # Forward pass
                predictions = model(images)
                
                # Compute loss
                if hasattr(model, 'compute_loss'):
                    # Jika model memiliki metode compute_loss sendiri
                    loss_dict = model.compute_loss(predictions, targets)
                    loss = loss_dict['total_loss']
                    
                    # Tambahkan component losses ke metrics
                    for k, v in loss_dict.items():
                        if k != 'total_loss':
                            if k not in val_metrics:
                                val_metrics[k] = 0
                            val_metrics[k] += v.item()
                else:
                    # Gunakan loss function default
                    if isinstance(predictions, dict) and isinstance(targets, dict):
                        # Multilayer predictions
                        loss = 0
                        for layer_name in predictions:
                            if layer_name in targets:
                                layer_pred = predictions[layer_name]
                                layer_target = targets[layer_name]
                                loss += torch.nn.functional.mse_loss(layer_pred, layer_target)
                    else:
                        # Single layer predictions
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(predictions, targets)
                
                # Update loss tracking
                total_loss += loss.item()
                batch_count += 1
                
                # Collect predictions dan targets untuk metrics
                if not isinstance(targets, dict):
                    try:
                        pred_classes = predictions.argmax(dim=1).cpu().numpy()
                        if isinstance(targets, torch.Tensor) and targets.dim() > 1:
                            true_classes = targets.argmax(dim=1).cpu().numpy()
                        else:
                            true_classes = targets.cpu().numpy()
                        
                        all_predictions.extend(pred_classes)
                        all_targets.extend(true_classes)
                    except:
                        pass
        
        # Calculate average loss
        avg_loss = total_loss / max(1, batch_count)
        
        # Average component metrics
        for k in val_metrics:
            val_metrics[k] /= max(1, batch_count)
        
        # Calculate additional metrics if we have predictions and targets
        if len(all_targets) > 0 and len(all_predictions) > 0:
            # Try to calculate precision, recall, f1, etc.
            try:
                from sklearn.metrics import precision_recall_fscore_support, accuracy_score
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='weighted', zero_division=0
                )
                
                accuracy = accuracy_score(all_targets, all_predictions)
                
                val_metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                })
            except ImportError:
                self.log_warning("⚠️ scikit-learn tidak ditemukan, metrics lanjutan tidak dihitung")
            except Exception as e:
                self.log_warning(f"⚠️ Error saat menghitung metrics: {str(e)}")
        
        # Trigger validation end callback
        self._trigger_callbacks(
            event='validation_end',
            epoch=epoch,
            val_loss=avg_loss,
            metrics=val_metrics
        )
        
        return avg_loss, val_metrics