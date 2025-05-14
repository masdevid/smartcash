"""
File: smartcash/model/services/training/callbacks_training_service.py
Deskripsi: Modul callback untuk layanan pelatihan
"""

import time
import torch
import os
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union

from smartcash.common.logger import get_logger


class TrainingCallbacks:
    """
    Kelas utilitas untuk membuat dan mengelola callback pada training loop.
    
    * new: Integrated callback system for extensible training
    """
    
    def __init__(self, logger = None):
        """
        Inisialisasi kumpulan callbacks.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("training_callbacks")
        self.callbacks = []
        
    def add_callback(self, callback: Callable) -> None:
        """
        Tambahkan callback ke daftar.
        
        Args:
            callback: Fungsi callback yang menerima metrics epoch sebagai parameter
        """
        self.callbacks.append(callback)
        self.logger.info(f"➕ Callback ditambahkan: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
        
    def execute(self, metrics: Dict[str, Any]) -> None:
        """
        Jalankan semua callbacks dengan metrics.
        
        Args:
            metrics: Dictionary metrics dari epoch saat ini
        """
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                self.logger.error(f"❌ Error saat menjalankan callback: {str(e)}")
    
    @staticmethod
    def create_checkpoint_callback(save_dir: str, model: torch.nn.Module, 
                                 prefix: str = "checkpoint", every_n_epochs: int = 10,
                                 save_best_only: bool = True, 
                                 monitor: str = "val_loss", mode: str = "min",
                                 logger = None) -> Callable:
        """
        Buat callback untuk menyimpan checkpoint model.
        
        Args:
            save_dir: Direktori untuk menyimpan checkpoint
            model: Model yang akan disimpan
            prefix: Prefix untuk nama file checkpoint
            every_n_epochs: Frekuensi penyimpanan checkpoint
            save_best_only: Hanya simpan checkpoint terbaik
            monitor: Metrik yang dimonitor untuk checkpoint terbaik
            mode: Mode evaluasi ('min' atau 'max')
            logger: Logger untuk mencatat aktivitas
            
        Returns:
            Fungsi callback
        """
        # Buat direktori jika belum ada
        os.makedirs(save_dir, exist_ok=True)
        
        # Simpan best metric dan inisialisasi state
        best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Set logger
        local_logger = logger or get_logger("checkpoint_callback")
        
        # Callback function
        def checkpoint_callback(metrics: Dict[str, Any]) -> None:
            nonlocal best_metric
            
            epoch = metrics.get('epoch', 0)
            
            # Cek apakah harus menyimpan pada epoch ini
            should_save_periodic = every_n_epochs > 0 and (epoch + 1) % every_n_epochs == 0
            
            # Cek apakah metrik saat ini lebih baik
            current_metric = metrics.get(monitor, float('inf') if mode == 'min' else float('-inf'))
            is_improved = (mode == 'min' and current_metric < best_metric) or \
                          (mode == 'max' and current_metric > best_metric)
            
            # Simpan checkpoint jika diperlukan
            if (should_save_periodic and not save_best_only) or (is_improved and save_best_only):
                checkpoint_path = os.path.join(save_dir, f"{prefix}_epoch{epoch + 1}.pt")
                
                # Simpan model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'best_metric': current_metric if is_improved else best_metric
                }, checkpoint_path)
                
                # Update best metric jika diperlukan
                if is_improved:
                    best_metric = current_metric
                    best_path = os.path.join(save_dir, f"{prefix}_best.pt")
                    # Copy file terbaik
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics,
                        'best_metric': best_metric
                    }, best_path)
                    
                    local_logger.success(f"💾 Model terbaik tersimpan: {best_path} [val_loss: {best_metric:.4f}]")
                
                local_logger.info(f"💾 Checkpoint tersimpan: {checkpoint_path}")
                
        return checkpoint_callback
    
    @staticmethod
    def create_progress_callback(log_every_n_steps: int = 10, logger = None) -> Callable:
        """
        Buat callback untuk menampilkan progress training.
        
        Args:
            log_every_n_steps: Frekuensi log progress
            logger: Logger untuk mencatat aktivitas
            
        Returns:
            Fungsi callback
        """
        # Set logger
        local_logger = logger or get_logger("progress_callback")
        
        # Simpan waktu mulai untuk perhitungan ETA
        start_time = time.time()
        
        # Callback function
        def progress_callback(metrics: Dict[str, Any]) -> None:
            epoch = metrics.get('epoch', 0)
            
            # Hanya log pada interval
            if epoch % log_every_n_steps != 0:
                return
                
            # Hitung ETA
            elapsed_time = time.time() - start_time
            steps_done = epoch + 1
            steps_total = metrics.get('total_epochs', 100)  # Default 100 jika tidak ada
            
            # Hindari division by zero
            if steps_done > 0:
                steps_remaining = steps_total - steps_done
                estimated_time_per_step = elapsed_time / steps_done
                eta_seconds = steps_remaining * estimated_time_per_step
                
                # Format ETA
                hours, remainder = divmod(eta_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                eta_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Log progress
                local_logger.info(
                    f"⏱️ Progress: {steps_done}/{steps_total} epochs "
                    f"({steps_done/steps_total*100:.1f}%) | "
                    f"ETA: {eta_str}"
                )
        
        return progress_callback
    
    @staticmethod
    def create_tensorboard_callback(log_dir: str, comment: str = "") -> Callable:
        """
        Buat callback untuk logging ke TensorBoard.
        
        Args:
            log_dir: Direktori untuk log TensorBoard
            comment: Komentar untuk run TensorBoard
            
        Returns:
            Fungsi callback
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return lambda x: None  # No-op jika TensorBoard tidak tersedia
            
        # Buat writer
        writer = SummaryWriter(log_dir=log_dir, comment=comment)
        
        # Callback function
        def tensorboard_callback(metrics: Dict[str, Any]) -> None:
            epoch = metrics.get('epoch', 0)
            
            # Log semua metrik numerik
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(key, value, epoch)
                    
        return tensorboard_callback
    
    @staticmethod
    def create_reduceLR_callback(scheduler: torch.optim.lr_scheduler._LRScheduler, 
                               monitor: str = 'val_loss', mode: str = 'min', 
                               patience: int = 3, factor: float = 0.1, 
                               min_lr: float = 1e-6, verbose: bool = True,
                               logger = None) -> Callable:
        """
        Buat callback untuk mengurangi learning rate pada plateau.
        
        Args:
            scheduler: Learning rate scheduler
            monitor: Metrik yang dimonitor
            mode: Mode evaluasi ('min' atau 'max')
            patience: Jumlah epoch tanpa peningkatan sebelum mengurangi LR
            factor: Faktor pengurangan learning rate
            min_lr: Learning rate minimum
            verbose: Tampilkan pesan log
            logger: Logger untuk mencatat aktivitas
            
        Returns:
            Fungsi callback
        """
        # Set logger
        local_logger = logger or get_logger("reduce_lr_callback")
        
        # State counter dan best metric
        counter = 0
        best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Callback function
        def reduceLR_callback(metrics: Dict[str, Any]) -> None:
            nonlocal counter, best_metric
            
            # Pastikan metrik yang dimonitor ada
            if monitor not in metrics:
                return
                
            current_metric = metrics[monitor]
            
            # Periksa apakah metrik membaik
            is_improved = (mode == 'min' and current_metric < best_metric) or \
                          (mode == 'max' and current_metric > best_metric)
                          
            if is_improved:
                # Reset counter
                counter = 0
                best_metric = current_metric
            else:
                # Tambah counter
                counter += 1
                
                # Kurangi LR jika patience tercapai
                if counter >= patience:
                    # Get current learning rates
                    current_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
                    
                    # Step scheduler jika ini adalah ReduceLROnPlateau
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(current_metric)
                    # Bila bukan ReduceLROnPlateau, kita harus mengurangi LR secara manual
                    else:
                        for i, param_group in enumerate(scheduler.optimizer.param_groups):
                            old_lr = param_group['lr']
                            new_lr = max(old_lr * factor, min_lr)
                            param_group['lr'] = new_lr
                            
                            if verbose:
                                local_logger.info(f"📉 Mengurangi learning rate: {old_lr:.6f} → {new_lr:.6f}")
                    
                    # Reset counter
                    counter = 0
        
        return reduceLR_callback