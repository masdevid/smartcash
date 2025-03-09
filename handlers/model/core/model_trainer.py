# File: smartcash/handlers/model/core/model_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk melatih model yang diringkas menggunakan EarlyStopping

import torch
import time
from typing import Dict, Optional, Any, List
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError, TrainingError

class ModelTrainer(ModelComponent):
    """
    Komponen untuk melatih model dengan pendekatan standar.
    Menggunakan TrainingPipeline dari utils untuk implementasi training.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        model_factory = None,
        optimizer_factory = None,
        checkpoint_adapter = None,
        metrics_adapter = None
    ):
        """
        Inisialisasi model trainer.
        
        Args:
            config: Konfigurasi training
            logger: Custom logger (opsional)
            model_factory: Factory untuk membuat model (opsional, lazy-loaded)
            optimizer_factory: Factory untuk membuat optimizer (opsional, lazy-loaded)
            checkpoint_adapter: Adapter untuk checkpoint management (opsional, lazy-loaded)
            metrics_adapter: Adapter untuk metrics calculation (opsional, lazy-loaded)
        """
        super().__init__(config, logger, "model_trainer")
        
        # Simpan factories dan adapters
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._checkpoint_adapter = checkpoint_adapter
        self._metrics_adapter = metrics_adapter
    
    def _initialize(self) -> None:
        """Inisialisasi internal komponen."""
        self.training_config = self.config.get('training', {})
        
        # Default parameter training
        self.default_epochs = self.training_config.get('epochs', 30)
        self.default_batch_size = self.training_config.get('batch_size', 16)
        self.default_early_stopping_patience = self.training_config.get('early_stopping_patience', 10)
        
        # Cek apakah di Google Colab
        self.in_colab = self._is_running_in_colab()
    
    def _is_running_in_colab(self) -> bool:
        """Deteksi apakah kode berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        if self._model_factory is None:
            from smartcash.handlers.model.core.model_factory import ModelFactory
            self._model_factory = ModelFactory(self.config, self.logger)
        return self._model_factory
    
    @property
    def optimizer_factory(self):
        """Lazy-loaded optimizer factory."""
        if self._optimizer_factory is None:
            from smartcash.handlers.model.core.optimizer_factory import OptimizerFactory
            self._optimizer_factory = OptimizerFactory(self.config, self.logger)
        return self._optimizer_factory
    
    @property
    def checkpoint_adapter(self):
        """Lazy-loaded checkpoint adapter."""
        if self._checkpoint_adapter is None:
            from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
            self._checkpoint_adapter = CheckpointAdapter(self.config, self.logger)
        return self._checkpoint_adapter
    
    @property
    def metrics_adapter(self):
        """Lazy-loaded metrics adapter."""
        if self._metrics_adapter is None:
            from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter
            self._metrics_adapter = MetricsAdapter(self.logger, self.config)
        return self._metrics_adapter
    
    def process(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses training model. Alias untuk train().
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            **kwargs: Parameter tambahan untuk training
            
        Returns:
            Dict hasil training
        """
        return self.train(train_loader, val_loader, **kwargs)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        epochs: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        save_path: Optional[str] = None,
        save_every: Optional[int] = None,
        early_stopping: Optional[bool] = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = 'val_loss',
        early_stopping_mode: str = 'min',
        early_stopping_min_delta: float = 0.0,
        mixed_precision: Optional[bool] = None,
        observers: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Latih model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model yang akan dilatih (opsional, buat baru jika None)
            epochs: Jumlah epoch training
            optimizer: Optimizer (opsional, buat baru jika None)
            scheduler: Learning rate scheduler (opsional, buat baru jika None)
            checkpoint_path: Path ke checkpoint untuk resume training (opsional)
            save_path: Path untuk menyimpan checkpoint (opsional)
            save_every: Simpan checkpoint setiap n epoch (opsional)
            early_stopping: Aktifkan early stopping (opsional)
            early_stopping_patience: Jumlah epoch tunggu untuk early stopping (opsional)
            early_stopping_metric: Metrik untuk early stopping (opsional)
            early_stopping_mode: Mode early stopping ('min' atau 'max') (opsional)
            early_stopping_min_delta: Delta minimum untuk dianggap improvement (opsional)
            mixed_precision: Aktifkan mixed precision training (opsional)
            observers: List observer untuk monitoring (opsional)
            **kwargs: Parameter tambahan untuk training pipeline
            
        Returns:
            Dict hasil training
        """
        start_time = time.time()
        
        try:
            # Buat model jika tidak diberikan
            if model is None:
                if checkpoint_path:
                    # Load model dari checkpoint
                    self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
                    model, _ = self.model_factory.load_model(checkpoint_path)
                else:
                    # Buat model baru
                    self.logger.info("🔄 Membuat model baru untuk training")
                    model = self.model_factory.create_model()
            
            # Tentukan device
            device = kwargs.get('device', None)
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Pindahkan model ke device
            model = model.to(device)
            
            # Buat optimizer jika tidak diberikan
            if optimizer is None:
                optimizer = self.optimizer_factory.create_optimizer(model)
            
            # Buat scheduler jika tidak diberikan
            if scheduler is None:
                steps_per_epoch = len(train_loader)
                scheduler = self.optimizer_factory.create_scheduler(
                    optimizer,
                    steps_per_epoch=steps_per_epoch
                )
            
            # Setup parameter training
            epochs = epochs or self.default_epochs
            early_stopping_patience = early_stopping_patience or self.default_early_stopping_patience
            save_every = save_every or self.training_config.get('save_period', 5)
            
            # Setup early stopping
            early_stopper = None
            if early_stopping:
                early_stopper = EarlyStopping(
                    monitor=early_stopping_metric,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode=early_stopping_mode,
                    logger=self.logger
                )
            
            # Setup mixed precision
            if mixed_precision is None:
                # Default ke True jika CUDA tersedia
                mixed_precision = torch.cuda.is_available() and self.config.get('model', {}).get('half_precision', True)
            
            # Log informasi training
            self.logger.info(
                f"🚀 Memulai training:\n"
                f"   • Epochs: {epochs}\n"
                f"   • Device: {device}\n"
                f"   • Early stopping: {early_stopping} "
                f"(metric: {early_stopping_metric}, patience: {early_stopping_patience})\n"
                f"   • Mixed precision: {mixed_precision}"
            )
            
            # Impor TrainingPipeline dari utils
            from smartcash.utils.training import TrainingPipeline
            
            # Buat pipeline dengan konfigurasi
            pipeline = TrainingPipeline(
                config=self.config,
                model_handler=model,
                logger=self.logger
            )
            
            # Register callback untuk observers
            if observers:
                for observer in observers:
                    pipeline.register_callback('epoch_end', observer.on_epoch_end)
                    pipeline.register_callback('training_start', observer.on_training_start)
                    pipeline.register_callback('training_end', observer.on_training_end)
            
            # Register early stopping callback jika diaktifkan
            if early_stopping and early_stopper:
                def early_stopping_callback(epoch, metrics, **callback_kwargs):
                    return early_stopper(metrics)
                
                pipeline.register_callback('epoch_end', early_stopping_callback)
            
            # Jalankan training
            results = pipeline.train(
                dataloaders={
                    'train': train_loader,
                    'val': val_loader
                },
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=epochs,
                resume_from_checkpoint=checkpoint_path,
                save_every=save_every,
                early_stopping=early_stopping,  # Gunakan flag untuk pipeline
                early_stopping_patience=early_stopping_patience,  # Masih dikirim untuk kompatibilitas
                mixed_precision=mixed_precision,
                device=device,
                **kwargs
            )
            
            # Hitung waktu eksekusi
            execution_time = time.time() - start_time
            hours, remainder = divmod(execution_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.logger.success(
                f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
                f"   • Epochs selesai: {results.get('epoch', 0)}/{epochs}\n"
                f"   • Best val loss: {results.get('best_val_loss', 'unknown')}\n"
                f"   • Best checkpoint: {results.get('best_checkpoint_path', 'unknown')}"
            )
            
            # Tambahkan informasi waktu eksekusi
            results['execution_time'] = execution_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error saat training model: {str(e)}")
            raise TrainingError(f"Gagal melakukan training: {str(e)}")
    
    def train_with_mixed_precision(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Latih model dengan mixed precision (FP16) untuk kecepatan lebih baik.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model yang akan dilatih (opsional)
            **kwargs: Parameter tambahan untuk train()
            
        Returns:
            Dict hasil training
        """
        # Aktifkan mixed precision
        kwargs['mixed_precision'] = True
        
        # Forward ke train() dengan mixed precision aktif
        return self.train(train_loader, val_loader, model, **kwargs)