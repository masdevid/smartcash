import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.model_handler import ModelHandler
from smartcash.handlers.data_handler import DataHandler
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.utils.model_checkpoint import ModelCheckpoint
from smartcash.utils.metrics import MetricsCalculator
from smartcash.exceptions.base import (
    TrainingError, DataError, ResourceError, ValidationError
)
from smartcash.interface.utils.safe_reporter import SafeProgressReporter

class TrainingPipeline:
    """Handler untuk pipeline training model SmartCash."""
    
    def __init__(self, config: Dict, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi training pipeline.
        
        Args:
            config: Konfigurasi training
            logger: Logger opsional
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Validasi dan setup path
            self._setup_paths()
            
            # Inisialisasi komponen training
            self._initialize_handlers()
            
            # Setup early stopping dan checkpoint
            self.early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=config['training'].get('early_stopping_patience', 10),
                logger=self.logger
            )
            
            self.checkpoint = ModelCheckpoint(
                save_dir=str(self.output_dir / 'weights'),
                logger=self.logger
            )
            
            self.metrics = MetricsCalculator()
            
        except Exception as e:
            error_msg = f"Gagal inisialisasi training pipeline: {str(e)}"
            self.logger.error(error_msg)
            raise TrainingError(error_msg)
    
    def _setup_paths(self):
        """Setup dan validasi path yang diperlukan."""
        try:
            # Buat direktori output
            self.output_dir = Path(self.config.get('output_dir', 'runs/train'))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set paths dalam config
            self.config.update({
                'output_dir': str(self.output_dir),
                'checkpoints_dir': str(self.output_dir / 'weights'),
                'tensorboard_dir': str(self.output_dir / 'tensorboard'),
                'visualization_dir': str(self.output_dir / 'visualizations')
            })
            
            # Buat subdirektori
            for path in [
                self.output_dir / 'weights',
                self.output_dir / 'tensorboard',
                self.output_dir / 'visualizations'
            ]:
                path.mkdir(exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyiapkan direktori: {str(e)}")
            raise

    def _initialize_handlers(self):
        """Inisialisasi model dan data handlers."""
        try:
            # Inisialisasi model handler
            # Mendapatkan jumlah kelas berdasarkan layer deteksi aktif
            detection_layers = self.config.get('layers', ['banknote'])
            self.model_handler = ModelHandler(
                config=self.config,
                config_path=self.config.get('config_path', 'configs/base_config.yaml'),
                num_classes=self._get_num_classes(detection_layers),
                logger=self.logger
            )
            
            # Inisialisasi data handler berdasarkan sumber data
            if self.config.get('data_source') == 'roboflow':
                from smartcash.handlers.roboflow_handler import RoboflowHandler
                self.data_handler = RoboflowHandler(
                    config=self.config,
                    logger=self.logger
                )
            else:
                self.data_handler = DataHandler(
                    config=self.config,
                    logger=self.logger
                )
                
        except Exception as e:
            raise TrainingError(f"Gagal inisialisasi handlers: {str(e)}")
    
    def _get_num_classes(self, layers: List[str]) -> int:
        """
        Menghitung total kelas berdasarkan layer deteksi yang aktif.
        
        Args:
            layers: List layer deteksi aktif
            
        Returns:
            Total jumlah kelas
        """
        # Definisikan jumlah kelas per layer
        layer_class_counts = {
            'banknote': 7,  # 001, 002, 005, 010, 020, 050, 100
            'nominal': 7,    # l2_001, l2_002, l2_005, l2_010, l2_020, l2_050, l2_100
            'security': 3    # l3_sign, l3_text, l3_thread
        }
        
        # Jumlahkan kelas dari layer aktif
        total_classes = 0
        for layer in layers:
            if layer in layer_class_counts:
                total_classes += layer_class_counts[layer]
        
        # Default ke 7 jika tidak ada layer valid
        return total_classes if total_classes > 0 else 7

    def train(self, display_manager=None) -> Dict:
        """
        Jalankan pipeline training dengan progress tracking aman.
        
        Args:
            display_manager: Optional display manager untuk tracking
        
        Returns:
            Dict berisi informasi training
        """
        # Gunakan SafeProgressReporter
        reporter = SafeProgressReporter(display_manager)
       
        try:
            # Validasi resource dan data
            self._check_resource_requirements()
            self._validate_data_paths()
            
            # Setup data loading
            batch_size = self.config['training']['batch_size']
            num_workers = self.config.get('model', {}).get('workers', 4)
            
            # Get train loader
            train_loader = self.data_handler.get_train_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # Get validation loader
            val_loader = self.data_handler.get_val_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # Inisialisasi model dan optimizer
            model = self.model_handler.get_model()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            # Setup variabel training
            best_metrics = {}
            n_epochs = self.config['training']['epochs']
            
            # Log konfigurasi training
            dialog = reporter.show_dialog(
                "Mulai Training", 
                f"Backbone: {self.config['backbone']} | Sumber Data: {self.config['data_source']} | Mode Deteksi: {self.config['detection_mode']} | "
                f"Total Epoch: {n_epochs} | Batch Size: {batch_size} | Learning Rate: {self.config['training']['learning_rate']}",
                {"y": "Ya", "n": "Tidak"}
            )
            if dialog == 'n':            
                return {}
            
            try:
                for epoch in range(n_epochs):
                    # Training phase
                    try:
                        reporter.show_progress(
                            message=f"Training Epoch {epoch+1}/{n_epochs}", 
                            current=epoch+1, 
                            total=n_epochs
                        )
                        
                        train_metrics = self._train_epoch(
                            model=model,
                            loader=train_loader,
                            optimizer=optimizer,
                            epoch=epoch,
                            total_epochs=n_epochs,
                            reporter=reporter
                        )
                    except Exception as train_err:
                        reporter.show_dialog(
                            title="Error Training", 
                            message=f"Gagal pada training epoch {epoch}: {str(train_err)}"
                        )
                        raise TrainingError(f"Error saat training: {str(train_err)}")
                    
                    # Validation phase    
                    try:
                        val_metrics = self._validate_epoch(
                            model=model,
                            loader=val_loader,
                            epoch=epoch,
                            total_epochs=n_epochs,
                            reporter=reporter
                        )
                    except Exception as val_err:
                        reporter.show_dialog(
                            title="Error Validasi", 
                            message=f"Gagal pada validasi epoch {epoch}: {str(val_err)}"
                        )
                        raise TrainingError(f"Error saat validasi: {str(val_err)}")
                    
                    # Early stopping check
                    if self.early_stopping(val_metrics):
                        reporter.show_dialog(
                            title="Early Stopping", 
                            message="Early stopping triggered"
                        )
                        break
                    
                    # Save checkpoint
                    is_best = val_metrics['val_loss'] < best_metrics.get('val_loss', float('inf'))
                    if is_best:
                        best_metrics = val_metrics
                        reporter.show_dialog(
                            title="Model Terbaik",
                            message=f"Model terbaik! Val Loss: {val_metrics['val_loss']:.4f}"
                        )
                    
                    try:
                        self.checkpoint.save(
                            model=model,
                            config=self.config,
                            epoch=epoch,
                            loss=val_metrics['val_loss'],
                            is_best=is_best
                        )
                    except Exception as checkpoint_err:
                        reporter.show_dialog(
                            title="Error Checkpoint", 
                            message=f"Gagal menyimpan checkpoint: {str(checkpoint_err)}"
                        )
                        raise TrainingError(f"Gagal menyimpan checkpoint: {str(checkpoint_err)}")
                
                # Akhir training
                reporter.show_dialog(
                    title="Training Selesai",
                    message=(
                        "Training berhasil diselesaikan!\n"
                        f"Best Loss: {best_metrics.get('val_loss', 'N/A'):.4f}"
                    )
                )
                
                return {
                    'train_dir': str(self.output_dir),
                    'best_metrics': best_metrics,
                    'config': self.config
                }
                
            except KeyboardInterrupt:
                reporter.show_dialog(
                    title="Training Dibatalkan", 
                    message="Training dihentikan oleh pengguna"
                )
                raise
                
        except Exception as e:
            reporter.show_dialog(
                title="Error Fatal", 
                message=f"Gagal menjalankan training: {str(e)}"
            )
            raise TrainingError(f"Gagal menjalankan training: {str(e)}")

    def _train_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
        reporter=None
    ) -> Dict:
        """Proses training satu epoch."""
        model.train()
        self.metrics.reset()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(loader):
            # Update progress
            if reporter:
                reporter.show_progress(
                    message=f"Training Epoch {epoch+1}/{total_epochs}", 
                    current=batch_idx+1, 
                    total=len(loader)
                )
            
            # Forward pass
            predictions = model(images)
            loss = model.compute_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss['total_loss'].backward()
            optimizer.step()
            
            # Update metrics
            self.metrics.update(predictions, targets)
            epoch_loss += loss['total_loss'].item()
        
        # Calculate epoch metrics
        metrics = self.metrics.compute()
        metrics['train_loss'] = epoch_loss / len(loader)
        
        return metrics

    def _validate_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        total_epochs: int,
        reporter=None
    ) -> Dict:
        """Proses validasi satu epoch."""
        model.eval()
        self.metrics.reset()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(loader):
                # Update progress
                if reporter:
                    reporter.show_progress(
                        message=f"Validasi Epoch {epoch+1}/{total_epochs}", 
                        current=batch_idx+1, 
                        total=len(loader)
                    )
                
                # Forward pass
                predictions = model(images)
                loss = model.compute_loss(predictions, targets)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                epoch_loss += loss['total_loss'].item()
        
        # Calculate epoch metrics
        metrics = self.metrics.compute()
        metrics['val_loss'] = epoch_loss / len(loader)
        
        return metrics
    
    def _check_resource_requirements(self):
        """
        Periksa ketersediaan sumber daya untuk training.
        
        Raises:
            ResourceError: Jika sumber daya tidak mencukupi
        """
        try:
            # Periksa ketersediaan GPU jika diperlukan
            if (self.config.get('device') == 'cuda' and 
                not torch.cuda.is_available()):
                raise ResourceError(
                    "GPU diperlukan tapi tidak tersedia. "
                    "Gunakan device='cpu' atau pastikan CUDA terinstal"
                )
                
            # Periksa kebutuhan memori
            batch_size = self.config['training']['batch_size']
            if batch_size > 128:
                self.logger.warning(
                    "Batch size besar (>128) dapat menyebabkan masalah memori"
                )
        except Exception as e:
            raise ResourceError(f"Gagal memeriksa resource: {str(e)}")

    def _validate_data_paths(self):
        """
        Validasi path dataset.
        
        Raises:
            DataError: Jika struktur atau isi dataset tidak valid
        """
        try:
            if self.config['data_source'] == 'local':
                for split in ['train', 'val', 'test']:
                    path = Path(self.config[f'{split}_data_path'])
                    if not path.exists():
                        raise DataError(
                            f"Path dataset {split} tidak ditemukan: {path}"
                        )
                    
                    # Periksa struktur folder
                    image_dir = path / 'images'
                    label_dir = path / 'labels'
                    
                    if not (image_dir.exists() and label_dir.exists()):
                        raise DataError(
                            f"Struktur folder {split} tidak valid. "
                            "Harus ada subfolder 'images' dan 'labels'"
                        )
                        
                    images = list(image_dir.glob('*.jpg'))
                    labels = list(label_dir.glob('*.txt'))
                    
                    if not images or not labels:
                        raise DataError(
                            f"Dataset {split} kosong. "
                            f"Images: {len(images)}, Labels: {len(labels)}"
                        )
                        
        except Exception as e:
            raise DataError(f"Gagal memvalidasi dataset: {str(e)}")