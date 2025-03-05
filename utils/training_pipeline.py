# File: smartcash/utils/training_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline training yang dioptimalkan untuk Google Colab dengan dukungan checkpointing dan visualisasi

import os
import time
import gc
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.utils.model_checkpoint import StatelessCheckpointSaver
from smartcash.utils.memory_optimizer import MemoryOptimizer

# Tambahkan import untuk mengatasi masalah TensorBoard
try:
    import tensorflow as tf
except ImportError:
    tf = None

class TrainingPipeline:
    """Pipeline training SmartCash dengan optimasi untuk Google Colab"""
    
    def __init__(
        self, 
        config, 
        model_manager=None, 
        data_manager=None, 
        logger=None
    ):
        self.config = config
        
        # Setup logger dengan mode yang lebih aman
        if logger is None:
            # Buat logger kustom jika tidak disediakan
            logger = logging.getLogger('training_pipeline')
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        
        self.logger = SmartCashLogger("training_pipeline") if isinstance(logger, SmartCashLogger) else logger
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Menyimpan referensi ke manager
        self.model_manager = model_manager
        self.data_manager = data_manager
        
        # Setup direktori output
        self.output_dir = Path(config.get('output_dir', 'runs/train'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard dengan penanganan error
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Gunakan SummaryWriter dengan penanganan error
        try:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception as e:
            # Fallback ke logging biasa jika TensorBoard gagal
            self.logger.warning(f"❌ Gagal menginisialisasi TensorBoard: {str(e)}")
            self.writer = None
        
        # Setup metrik untuk tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        # Setup memory optimizer
        self.memory_optimizer = MemoryOptimizer(logger=self.logger)
        
        # Status training
        self.is_training = False
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
    
    def _safe_log(self, log_level, message):
        """
        Logging yang aman dengan berbagai metode
        
        Args:
            log_level: Level logging (info, warning, error, etc)
            message: Pesan yang akan di-log
        """
        try:
            # Coba dengan SmartCashLogger jika tersedia
            if hasattr(self.logger, 'log'):
                self.logger.log(log_level, 'info', message)
            # Fallback ke logging standar
            elif hasattr(self.logger, log_level):
                getattr(self.logger, log_level)(message)
            else:
                # Fallback paling akhir
                logging.log(
                    getattr(logging, log_level.upper(), logging.INFO), 
                    message
                )
        except Exception as e:
            # Logging terakhir dengan print jika semua cara gagal
            print(f"Logging error: {e}. Original message: {message}")
    
    def _log_metrics(self, epoch, avg_train_loss, avg_val_loss):
        """
        Log metrik dengan aman, termasuk ke TensorBoard jika tersedia
        """
        # Log ke console
        self._safe_log('info', 
            f"⏱️ Epoch [{epoch+1}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        
        # Log ke TensorBoard jika tersedia
        if self.writer:
            try:
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            except Exception as e:
                self._safe_log('warning', f"Gagal menulis ke TensorBoard: {str(e)}")
    
    def train(
        self, 
        model=None, 
        dataloaders=None, 
        resume_from_checkpoint=None,
        epochs=None,
        save_every=1
    ):
        """
        Training dengan Early Stopping dan Checkpoint Saving
        
        Args:
            model: Model yang akan dilatih (jika None, akan diambil dari model_manager)
            dataloaders: Dictionary berisi dataloader train dan val (jika None, akan diambil dari data_manager)
            resume_from_checkpoint: Path ke checkpoint untuk melanjutkan training (opsional)
            epochs: Jumlah epochs (jika None, akan diambil dari config)
            save_every: Interval untuk menyimpan checkpoint
            
        Returns:
            Dictionary berisi hasil training
        """
        # Setup model dan dataloaders jika belum ada
        model = model or self.model_manager.get_model()
        dataloaders = dataloaders or self.data_manager.get_dataloaders()
        
        # Setup optimizer
        optimizer = self.model_manager.get_optimizer(model)
        
        # Setup scheduler
        scheduler = self.model_manager.get_scheduler(optimizer)
        
        # Setup early stopping
        patience = self.config.get('training', {}).get('early_stopping_patience', 5)
        early_stopping = EarlyStopping(patience=patience, mode='min')
        
        # Hyperparameters
        epochs = epochs or self.config.get('training', {}).get('epochs', 30)
        
        # Resume training jika diperlukan
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('loss', float('inf'))
            self.logger.info(f"📂 Melanjutkan training dari epoch {start_epoch} dengan loss {self.best_val_loss:.4f}")
        
        self.logger.info(f"🚀 Memulai training untuk {epochs} epochs")
        self.logger.info(f"🖥️ Device: {self.device}")
        
        # Mulai timer
        self.training_start_time = time.time()
        self.is_training = True
        self.current_epoch = start_epoch
        
        # Implementasi mixed precision untuk efisiensi memori
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training loop
        try:
            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                model.train()
                train_loss = 0.0
                
                train_progress = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} [Train]")
                
                # Track batch untuk mengosongkan CUDA cache setiap interval tertentu
                batch_counter = 0
                
                for batch_idx, (images, targets) in enumerate(train_progress):
                    batch_counter += 1
                    
                    # Move to device
                    images = images.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)
                    
                    # Forward pass dengan mixed precision
                    if scaler:
                        with torch.cuda.amp.autocast():
                            predictions = model(images)
                            loss_dict = model.compute_loss(predictions, targets)
                            loss = loss_dict['total_loss']
                        
                        # Backward dan optimize dengan scaler
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard forward pass
                        predictions = model(images)
                        loss_dict = model.compute_loss(predictions, targets)
                        loss = loss_dict['total_loss']
                        
                        # Backward dan optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    train_progress.set_postfix({'loss': loss.item()})
                    
                    # Bersihkan cache GPU setiap 10 batch
                    if torch.cuda.is_available() and batch_counter % 10 == 0:
                        torch.cuda.empty_cache()
                
                # Calculate average training loss
                avg_train_loss = train_loss / len(dataloaders['train'])
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                val_progress = tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{epochs} [Val]")
                
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(val_progress):
                        # Move to device
                        images = images.to(self.device)
                        if isinstance(targets, torch.Tensor):
                            targets = targets.to(self.device)
                        
                        # Forward pass
                        predictions = model(images)
                        
                        # Compute loss
                        loss_dict = model.compute_loss(predictions, targets)
                        loss = loss_dict['total_loss']
                        
                        # Update metrics
                        val_loss += loss.item()
                        val_progress.set_postfix({'loss': loss.item()})
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(dataloaders['val'])
                
                # Update scheduler berdasarkan validasi loss
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
                
                # Log metrics
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
                
                # Track metrics history
                self.metrics_history['train_loss'].append(avg_train_loss)
                self.metrics_history['val_loss'].append(avg_val_loss)
                self.metrics_history['epochs'].append(epoch)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start
                
                # Log epoch results
                self.logger.info(
                    f"⏱️ Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Check if this is the best model so far
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss
                    self.logger.success(f"🏆 Validasi loss terbaik: {self.best_val_loss:.4f}")
                
                # Save checkpoint berdasarkan interval atau jika best
                if epoch % save_every == 0 or is_best or epoch == epochs - 1:
                    checkpoint_paths = self.model_manager.save_checkpoint(
                        model=model,
                        epoch=epoch,
                        loss=avg_val_loss,
                        is_best=is_best
                    )
                
                # Early stopping check
                if early_stopping(avg_val_loss):
                    self.logger.warning(f"⚠️ Early stopping setelah epoch {epoch+1}")
                    break
                    
                # Plot progress setelah setiap epoch
                if (epoch + 1) % save_every == 0:
                    self._plot_progress()
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Training selesai
            training_time = time.time() - self.training_start_time
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.logger.success(
                f"✅ Training selesai dalam "
                f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            )
            
            # Final plot
            self._plot_training_curves()
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Training dihentikan oleh pengguna")
            # Simpan checkpoint terakhir jika dihentikan
            self.model_manager.save_checkpoint(
                model=model,
                epoch=self.current_epoch,
                loss=float('inf'),  # Nanti akan diupdate jika ada
                is_best=False
            )
        except Exception as e:
            self.logger.error(f"❌ Error selama training: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            # Tutup TensorBoard writer
            self.writer.close()
        
        return {
            'model': model,
            'best_val_loss': self.best_val_loss,
            'training_time': time.time() - self.training_start_time,
            'epochs_completed': len(self.metrics_history['epochs']),
            'metrics_history': self.metrics_history
        }
    
    def _plot_progress(self):
        """Plot kurva loss untuk monitoring selama training"""
        if len(self.metrics_history['epochs']) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics_history['epochs'], self.metrics_history['train_loss'], label='Training Loss', marker='o')
            plt.plot(self.metrics_history['epochs'], self.metrics_history['val_loss'], label='Validation Loss', marker='o')
            plt.title('Kurva Loss Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight best epoch
            best_epoch_idx = np.argmin(self.metrics_history['val_loss'])
            best_epoch = self.metrics_history['epochs'][best_epoch_idx]
            best_loss = self.metrics_history['val_loss'][best_epoch_idx]
            plt.scatter([best_epoch], [best_loss], c='red', s=100, zorder=5)
            plt.annotate(f'Best: {best_loss:.4f}', 
                         (best_epoch, best_loss),
                         xytext=(5, 5), 
                         textcoords='offset points')
            
            plt.tight_layout()
            plt.show()
            
    def _plot_training_curves(self):
        """Plot kurva loss training dan validasi di akhir training"""
        plt.figure(figsize=(12, 8))
        
        # Subplot untuk loss
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics_history['epochs'], self.metrics_history['train_loss'], label='Training Loss', marker='o')
        plt.plot(self.metrics_history['epochs'], self.metrics_history['val_loss'], label='Validation Loss', marker='o')
        plt.title('Kurva Loss Training dan Validasi')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight best epoch
        if self.metrics_history['val_loss']:
            best_epoch_idx = np.argmin(self.metrics_history['val_loss'])
            best_epoch = self.metrics_history['epochs'][best_epoch_idx]
            best_loss = self.metrics_history['val_loss'][best_epoch_idx]
            plt.scatter([best_epoch], [best_loss], c='red', s=100, zorder=5)
            plt.annotate(f'Best: {best_loss:.4f}', 
                         (best_epoch, best_loss),
                         xytext=(5, -15), 
                         textcoords='offset points')
        
        # Subplot untuk learning rate jika tersedia
        if hasattr(self, 'learning_rates') and self.learning_rates:
            plt.subplot(2, 1, 2)
            plt.plot(self.metrics_history['epochs'], self.learning_rates, label='Learning Rate', marker='s')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Simpan plot
        plot_path = self.output_dir / 'loss_curve.png'
        plt.savefig(plot_path)
        plt.show()
        
        self.logger.info(f"📊 Kurva loss disimpan di {plot_path}")
        
        return plot_path
        
    def copy_checkpoints_to_drive(self, drive_path='/content/drive/MyDrive/SmartCash/weights'):
        """
        Salin checkpoint ke Google Drive untuk backup
        
        Args:
            drive_path: Path tujuan di Google Drive
        """
        if not os.path.exists('/content/drive'):
            self.logger.warning("⚠️ Google Drive tidak terpasang. Gunakan command 'drive.mount(\"/content/drive\")' terlebih dahulu.")
            return False
            
        try:
            # Buat direktori tujuan jika belum ada
            os.makedirs(drive_path, exist_ok=True)
            
            # Cari semua checkpoint
            checkpoint_dir = self.output_dir / 'weights'
            
            if not checkpoint_dir.exists():
                self.logger.warning(f"⚠️ Direktori checkpoint tidak ditemukan: {checkpoint_dir}")
                return False
                
            checkpoints = list(checkpoint_dir.glob('*.pth'))
            
            if not checkpoints:
                self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan")
                return False
                
            # Salin semua checkpoint
            import shutil
            copied_count = 0
            
            for checkpoint in tqdm(checkpoints, desc="📂 Menyalin checkpoint ke Drive"):
                dest_path = os.path.join(drive_path, checkpoint.name)
                # Cek apakah sudah ada dan lebih baru
                if not os.path.exists(dest_path) or os.path.getmtime(checkpoint) > os.path.getmtime(dest_path):
                    shutil.copy2(checkpoint, dest_path)
                    copied_count += 1
            
            self.logger.success(f"✅ {copied_count} checkpoint berhasil disalin ke {drive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return False
    
    def get_training_status(self):
        """Dapatkan status training saat ini"""
        if not self.is_training:
            return {
                'status': 'idle',
                'message': 'Training belum dimulai atau sudah selesai'
            }
            
        # Hitung estimasi waktu tersisa
        if self.training_start_time and self.current_epoch > 0:
            elapsed_time = time.time() - self.training_start_time
            time_per_epoch = elapsed_time / self.current_epoch
            total_epochs = self.config.get('training', {}).get('epochs', 30)
            remaining_epochs = total_epochs - self.current_epoch
            estimated_time = time_per_epoch * remaining_epochs
            
            # Format waktu
            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return {
                'status': 'training',
                'current_epoch': self.current_epoch,
                'total_epochs': total_epochs,
                'progress': (self.current_epoch / total_epochs) * 100,
                'best_val_loss': self.best_val_loss,
                'estimated_time_remaining': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
                'message': f"Training berjalan (Epoch {self.current_epoch+1}/{total_epochs})"
            }
        
        return {
            'status': 'training',
            'current_epoch': self.current_epoch,
            'message': f"Training berjalan (Epoch {self.current_epoch+1})"
        }
    
    def optimize_batch_size(self, model=None, target_memory_usage=0.7):
        """
        Optimasi batch size berdasarkan memori yang tersedia
        
        Args:
            model: Model yang akan dioptimasi (jika None, akan dibuat model baru)
            target_memory_usage: Target penggunaan memori GPU (0.0-1.0)
            
        Returns:
            Batch size optimal
        """
        model = model or self.model_manager.get_model()
        return self.memory_optimizer.optimize_batch_size(model, target_memory_usage)