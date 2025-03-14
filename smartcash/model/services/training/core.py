"""
File: smartcash/model/services/training/core.py
Deskripsi: Implementasi inti untuk layanan training model SmartCash dengan pendekatan modular dan terintegrasi dengan EfficientNet-B4 sebagai backbone.
"""
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable

from smartcash.model.services.training.optimizer import OptimizerFactory
from smartcash.model.services.training.scheduler import SchedulerFactory
from smartcash.model.services.training.early_stopping import EarlyStoppingHandler
from smartcash.utils.experiment_tracker import ExperimentTracker
from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config


class TrainingService:
    """
    Layanan inti untuk proses training model SmartCash dengan dukungan untuk
    berbagai backbone termasuk EfficientNet-B4.
    
    * old: handlers.model.core.model_trainer.train()
    * migrated: Simplified service-based training
    """
    
    def __init__(
        self, 
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        logger: Optional = None,
        experiment_name: Optional[str] = None
    ):
        """
        Inisialisasi training service.
        
        Args:
            model: Model PyTorch yang akan dilatih
            config: Konfigurasi training
            device: Device untuk training (CPU/GPU)
            logger: Logger untuk mencatat progress
            experiment_name: Nama eksperimen (untuk tracking)
        """
        self.model = model
        self.config = config
        self.logger = logger or get_logger("training_service")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        
        # Konfigurasi training
        self.train_config = config.get('training', {})
        self.epochs = self.train_config.get('epochs', 100)
        self.save_period = self.train_config.get('save_period', 10)
        
        # Parameter training dinamis
        self.best_metric = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        self.experiment_tracker = None
        
        # Initialize components
        self._setup_components()
        
        # Pindahkan model ke device
        self.model.to(self.device)
        
        # Setup experiment tracker untuk monitoring
        if self.train_config.get('enable_tracking', True):
            self._setup_experiment_tracker()
        
        self.logger.info(f"🚀 Training service diinisialisasi: {self.experiment_name}")
        self.logger.info(f"🖥️ Device: {self.device}")
        
    def _setup_components(self):
        """Setup komponen training seperti optimizer, scheduler, dll."""
        # Setup optimizer
        optim_config = self.train_config.get('optimizer', {'type': 'adam', 'lr': 0.001})
        self.optimizer = OptimizerFactory.create(
            optimizer_type=optim_config.get('type', 'adam'),
            model_params=self.model.parameters(),
            **optim_config
        )
        
        # Setup scheduler
        sched_config = self.train_config.get('scheduler', {'type': 'cosine', 'T_max': self.epochs})
        # Pastikan T_max sesuai dengan jumlah epochs
        if 'T_max' not in sched_config and sched_config.get('type') in ['cosine', 'cosine_with_restarts']:
            sched_config['T_max'] = self.epochs
            
        self.scheduler = SchedulerFactory.create(
            scheduler_type=sched_config.get('type', 'cosine'),
            optimizer=self.optimizer,
            **sched_config
        )
        
        # Setup early stopping
        early_stopping_config = self.train_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            self.early_stopping = EarlyStoppingHandler(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                mode=early_stopping_config.get('mode', 'min'),
                logger=self.logger
            )
        else:
            self.early_stopping = None
        
        # Get layer configs
        self.layer_config = get_layer_config()
    
    def _setup_experiment_tracker(self):
        """Setup experiment tracker untuk monitoring."""
        output_dir = self.train_config.get('output_dir', 'runs/train/experiments')
        self.experiment_tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            output_dir=output_dir,
            logger=self.logger
        )
        
        # Start experiment dengan konfigurasi
        self.experiment_tracker.start_experiment(self.config)
        
    def train(self, train_loader, val_loader, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Latih model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk dataset training
            val_loader: DataLoader untuk dataset validasi
            callbacks: List fungsi callback untuk kustomisasi training loop
            
        Returns:
            Dictionary hasil training
        """
        total_start_time = time.time()
        self.logger.start(f"🚀 Memulai training untuk {self.epochs} epochs")
        
        # Pastikan model dalam mode training
        self.model.train()
        
        # Reset best metric dan counter
        self.best_metric = float('inf')
        results = {'epochs': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch(train_loader)
            
            # Validation epoch
            val_loss, metrics = self._validate_epoch(val_loader)
            
            # Learning rate saat ini
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update hasil
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr,
                **metrics
            }
            
            # Track metrics
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    additional_metrics=metrics
                )
            
            # Tambahkan ke hasil
            for key in epoch_result:
                if key not in results:
                    results[key] = []
                results[key].append(epoch_result[key])
                
            # Log progress
            self.logger.info(
                f"📊 Epoch {epoch+1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"lr={current_lr:.6f}"
            )
            
            # Jalankan callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch_result)
            
            # Check early stopping
            if self.early_stopping:
                stop = self.early_stopping({'val_loss': val_loss, **metrics})
                if stop:
                    self.logger.warning(f"⚠️ Early stopping triggered pada epoch {epoch+1}")
                    break
            
            # Save checkpoint jika periode save
            if (epoch + 1) % self.save_period == 0 or epoch == self.epochs - 1:
                checkpoint_name = f"{self.experiment_name}_epoch{epoch + 1}"
                self._save_checkpoint(checkpoint_name)
                
            # Update learning rate
            self.scheduler.step()
            
        # End experiment jika menggunakan tracker
        if self.experiment_tracker:
            self.experiment_tracker.end_experiment()
            
        # Hitung total waktu training
        total_duration = time.time() - total_start_time
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        
        self.logger.success(
            f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"   Best val_loss: {self.best_metric:.4f}"
        )
        
        return results
    
    def _train_epoch(self, train_loader) -> float:
        """
        Proses training untuk satu epoch.
        
        Args:
            train_loader: DataLoader untuk dataset training
            
        Returns:
            Rata-rata training loss
        """
        self.model.train()
        total_loss = 0
        
        # Loop untuk batch data
        for batch_idx, batch in enumerate(train_loader):
            # Pindahkan data ke device
            images = batch['images'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Reset gradien
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Hitung loss untuk semua layer yang aktif
            loss = self._compute_loss(outputs, targets)
            
            # Backward dan optimize
            loss.backward()
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            self.global_step += 1
            
        # Return rata-rata loss
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """
        Proses validasi untuk satu epoch.
        
        Args:
            val_loader: DataLoader untuk dataset validasi
            
        Returns:
            Tuple (val_loss, metrics) dengan val_loss sebagai float dan metrics sebagai dictionary
        """
        self.model.eval()
        total_loss = 0
        all_metrics = {}
        
        # Tidak hitung gradien pada saat validasi
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Pindahkan data ke device
                images = batch['images'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                # Forward pass
                outputs = self.model(images)
                
                # Hitung loss dan metrics
                loss = self._compute_loss(outputs, targets)
                metrics = self._compute_metrics(outputs, targets)
                
                # Update total loss dan metrics
                total_loss += loss.item()
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = 0
                    all_metrics[key] += value
        
        # Rata-rata loss dan metrics
        val_loss = total_loss / len(val_loader)
        for key in all_metrics:
            all_metrics[key] /= len(val_loader)
        
        # Update best metric
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            # Simpan model terbaik
            self._save_checkpoint(f"{self.experiment_name}_best")
        
        return val_loss, all_metrics
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        """
        Hitung loss untuk setiap layer dan gabungkan.
        
        Args:
            outputs: Output dari model
            targets: Target label
            
        Returns:
            Tensor loss yang digabungkan
        """
        # Implementasikan perhitungan loss
        # Ini hanya contoh, implementasi sebenarnya tergantung pada arsitektur model
        total_loss = 0
        
        # Jika output dan target sudah dalam bentuk dict dengan multi-layer
        if isinstance(outputs, dict) and isinstance(targets, dict):
            for layer_name in outputs:
                if layer_name in targets:
                    # Hitung loss per layer
                    pred = outputs[layer_name]
                    target = targets[layer_name]
                    
                    # Gunakan loss function yang sesuai
                    # Contoh: MSE untuk regression, BCE untuk classification
                    layer_loss = nn.MSELoss()(pred, target)
                    
                    # Tambahkan ke total loss
                    total_loss += layer_loss
        else:
            # Fallback jika bukan dict (single output)
            total_loss = nn.MSELoss()(outputs, targets)
        
        return total_loss
    
    def _compute_metrics(self, outputs, targets) -> Dict[str, float]:
        """
        Hitung metrics lain selain loss.
        
        Args:
            outputs: Output dari model
            targets: Target label
            
        Returns:
            Dictionary berisi metrics tambahan
        """
        # Ini contoh sederhana, implementasi sebenarnya tergantung pada kebutuhan
        metrics = {}
        
        # Hitung metrik seperti accuracy, precision, recall, dll
        # ...
        
        return metrics
    
    def _save_checkpoint(self, checkpoint_name: str) -> str:
        """
        Simpan checkpoint model.
        
        Args:
            checkpoint_name: Nama untuk checkpoint
            
        Returns:
            Path file checkpoint
        """
        # Buat direktori checkpoint jika belum ada
        checkpoint_dir = os.path.join(self.train_config.get('checkpoint_dir', 'runs/train/weights'))
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        
        # Siapkan checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'experiment': self.experiment_name
        }
        
        # Simpan checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint tersimpan: {checkpoint_path}")
        
        return checkpoint_path
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Lanjutkan training dari checkpoint.
        
        Args:
            checkpoint_path: Path ke file checkpoint
            
        Returns:
            Dictionary berisi data checkpoint
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
            return {}
        
        self.logger.info(f"📂 Memuat checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1  # Resume dari epoch berikutnya
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        self.logger.success(
            f"✅ Berhasil memuat checkpoint dari epoch {checkpoint.get('epoch', 0)}\n"
            f"   Best metric: {self.best_metric:.4f}"
        )
        
        return checkpoint