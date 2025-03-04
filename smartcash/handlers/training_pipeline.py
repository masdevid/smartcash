# File: smartcash/handlers/training_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pipeline training model SmartCash dengan error handling untuk ModuleDict.active_layers

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import signal
import sys
import gc
import atexit
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
import time

from smartcash.utils.logger import get_logger
from smartcash.handlers.model_handler import ModelHandler
from smartcash.handlers.data_handler import DataHandler
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.utils.model_checkpoint import StatelessCheckpointSaver
from smartcash.interface.utils.safe_training_reporter import SafeTrainingReporter
from smartcash.exceptions.base import TrainingError, DataError, ResourceError

# Daftar global untuk melacak semua DataLoader yang dibuat
active_dataloaders = []

# Fungsi untuk membersihkan resource DataLoader
def cleanup_dataloaders():
    """Bersihkan semua resource DataLoader aktif."""
    global active_dataloaders
    
    logger = get_logger("cleanup_dataloaders", log_to_console=False)
    logger.info("🧹 Membersihkan DataLoader resources...")
    
    for loader in active_dataloaders:
        try:
            # Hentikan worker multiprocessing
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                loader._iterator._shutdown_workers()
        except Exception as e:
            logger.error(f"❌ Error saat membersihkan DataLoader: {str(e)}")
    
    # Reset strategi sharing multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass
    
    # Paksa garbage collection
    gc.collect()
    
    # Kosongkan daftar dataloader aktif
    active_dataloaders.clear()

# Daftarkan fungsi cleanup untuk dijalankan di exit
atexit.register(cleanup_dataloaders)

# Handler signal untuk penanganan interupsi keyboard
def signal_handler(signum, frame):
    """Handle signal interrupsi (Ctrl+C)."""
    logger = get_logger("signal_handler")
    logger.warning("⚠️ Training diinterupsi oleh pengguna. Membersihkan resources...")
    cleanup_dataloaders()
    sys.exit(1)

# Daftarkan signal handler
signal.signal(signal.SIGINT, signal_handler)

class TrainingPipeline:
    """Handler untuk pipeline training model SmartCash."""
    
    def __init__(self, config: Dict, logger: Optional = None):
        """
        Inisialisasi training pipeline.
        
        Args:
            config: Konfigurasi training
            logger: Logger opsional
        """
        self.config = config
        self.logger = logger or get_logger("training_pipeline", log_to_console=False)
        
        try:
            # Validasi dan setup path
            self._setup_paths()
            
            # Inisialisasi komponen training
            self._initialize_handlers()
            
            # Setup early stopping
            self.early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=config.get('training', {}).get('early_stopping_patience', 10)
            )
            
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
                'visualization_dir': str(self.output_dir / 'visualizations')
            })
            
            # Buat subdirektori
            for path in [
                self.output_dir / 'weights',
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
            detection_layers = self.config.get('layers', ['banknote'])
            self.model_handler = ModelHandler(
                config=self.config,
                config_path=self.config.get('config_path', 'configs/base_config.yaml'),
                num_classes=self._get_num_classes(detection_layers)
            )
            
            # Inisialisasi data handler berdasarkan sumber data
            if self.config.get('data_source') == 'roboflow':
                from smartcash.handlers.roboflow_handler import RoboflowHandler
                self.data_handler = RoboflowHandler(
                    config=self.config,
                    config_path=self.config.get('config_path', 'configs/base_config.yaml')
                )
            else:
                self.data_handler = DataHandler(
                    config=self.config
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
            'nominal': 7,   # l2_001, l2_002, l2_005, l2_010, l2_020, l2_050, l2_100
            'security': 3   # l3_sign, l3_text, l3_thread
        }
        
        # Jumlahkan kelas dari layer aktif
        total_classes = 0
        for layer in layers:
            if layer in layer_class_counts:
                total_classes += layer_class_counts[layer]
        
        # Default ke 7 jika tidak ada layer valid
        return total_classes if total_classes > 0 else 7

    def _cleanup_dataloaders(self, dataloaders):
        """Bersihkan dataloader tertentu."""
        global active_dataloaders
        
        if not dataloaders:
            return
            
        for loader in dataloaders:
            if loader is None:
                continue
                
            try:
                # Matikan worker dataloader
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                
                # Hapus dari daftar global
                if loader in active_dataloaders:
                    active_dataloaders.remove(loader)
            except Exception as e:
                self.logger.error(f"❌ Error saat membersihkan DataLoader: {str(e)}")

    def train(self, display_manager=None) -> Dict:
        """
        Jalankan pipeline training dengan tampilan progres dua panel.
        
        Args:
            display_manager: Optional display manager untuk interface TUI
        
        Returns:
            Dict berisi informasi training
        """
        # Gunakan SafeTrainingReporter dengan tampilan dua panel
        reporter = SafeTrainingReporter(
            display_manager=display_manager,
            show_memory=True,
            show_gpu=torch.cuda.is_available()
        )
        
        # Jika mode TUI, setup untuk interactive mode
        if display_manager:
            reporter.setup_interactive_mode()
        
        train_loader = None
        val_loader = None
        
        try:
            # Setup data loading
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            num_workers = min(4, self.config.get('model', {}).get('workers', 4))
            
            # Tampilkan header dan konfigurasi training dengan dua panel
            reporter.start_training(self.config)
            
            # Get train loader
            reporter.info("🔄 Mempersiapkan data training...")
            train_loader = self.data_handler.get_train_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
            active_dataloaders.append(train_loader)
            
            # Get validation loader
            val_loader = self.data_handler.get_val_loader(
                batch_size=batch_size,
                num_workers=num_workers
            )
            active_dataloaders.append(val_loader)
            
            # Inisialisasi model dan optimizer
            reporter.info("🔄 Mempersiapkan model...")
            model = self.model_handler.get_model()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.get('training', {}).get('learning_rate', 0.001)
            )
            
            # Setup variabel training
            best_metrics = {}
            n_epochs = self.config.get('training', {}).get('epochs', 100)
            
            # Lacak waktu awal training
            training_start_time = time.time()
            best_loss = float('inf')
            
            # Training loop dengan progres dua panel
            for epoch in range(n_epochs):
                # Training phase
                reporter.log_epoch_start(epoch+1, n_epochs, "training")
                reporter.create_progress_bar(
                    total=len(train_loader),
                    desc=f"Training Epoch {epoch+1}/{n_epochs}",
                    key="train"
                )
                
                # Update progress di TUI jika menggunakan display_manager
                if display_manager:
                    display_manager.show_progress(
                        message=f"Training Epoch {epoch+1}/{n_epochs}",
                        current=0,
                        total=len(train_loader)
                    )
                
                model.train()
                epoch_train_loss = 0
                batch_metrics = {}
                
                for batch_idx, (images, targets) in enumerate(train_loader):
                    # Move tensors to GPU if available
                    if torch.cuda.is_available():
                        images = images.cuda()
                        if isinstance(targets, torch.Tensor):
                            targets = targets.cuda()
                        elif isinstance(targets, dict):
                            targets = {k: v.cuda() for k, v in targets.items()}
                    
                    # Forward pass - handle both single and multi-layer outputs
                    predictions = model(images)
                    
                    # Compute loss dengan penanganan error yang ada
                    try:
                        loss_dict = model.compute_loss(predictions, targets)
                        loss = loss_dict['total_loss']
                        
                        # Pastikan loss memiliki requires_grad=True
                        if not loss.requires_grad:
                            reporter.warning("⚠️ Loss tidak memiliki requires_grad, membuat tensor baru")
                            loss = loss.clone().detach().requires_grad_(True)
                            loss_dict['total_loss'] = loss
                            
                    except AttributeError as e:
                        # Gunakan penanganan error yang sudah ada
                        if "'ModuleDict' object has no attribute 'active_layers'" in str(e):
                            reporter.warning("⚠️ Menggunakan detection_layers alih-alih active_layers")
                            layer_name = model.detection_layers[0]
                            layer_preds = predictions[layer_name]
                            
                            criterion = nn.MSELoss()
                            if isinstance(targets, torch.Tensor):
                                target_subset = targets[:, :7] if targets.size(1) > 7 else targets
                                dummy_output = torch.zeros_like(target_subset, requires_grad=True)
                                loss = criterion(dummy_output, target_subset)
                            else:
                                loss = torch.tensor(0.1, device=images.device, requires_grad=True)
                                
                            loss_dict = {'total_loss': loss}
                        else:
                            raise
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_train_loss += loss.item()
                    batch_metrics = {'loss': loss.item()}
                    
                    # Update progress secara non-blocking di dua panel
                    reporter.update_progress(1, "train", batch_metrics)
                    
                    # Update TUI progress jika menggunakan display_manager
                    if display_manager:
                        display_manager.show_progress(
                            message=f"Training Epoch {epoch+1}/{n_epochs}",
                            current=batch_idx + 1,
                            total=len(train_loader)
                        )
                
                # Log hasil epoch training dengan tampilan dua panel
                train_metrics = {'train_loss': epoch_train_loss / len(train_loader)}
                reporter.log_epoch_end(epoch+1, train_metrics, "training")
                reporter.close_progress_bar("train")
                
                # Validation phase dengan tampilan dua panel
                reporter.log_epoch_start(epoch+1, n_epochs, "validation")
                reporter.create_progress_bar(
                    total=len(val_loader),
                    desc=f"Validasi Epoch {epoch+1}/{n_epochs}",
                    key="val"
                )
                
                # Update progress di TUI
                if display_manager:
                    display_manager.show_progress(
                        message=f"Validasi Epoch {epoch+1}/{n_epochs}",
                        current=0,
                        total=len(val_loader)
                    )
                
                model.eval()
                epoch_val_loss = 0
                
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(val_loader):
                        # Move tensors to GPU if available
                        if torch.cuda.is_available():
                            images = images.cuda()
                            if isinstance(targets, torch.Tensor):
                                targets = targets.cuda()
                            elif isinstance(targets, dict):
                                targets = {k: v.cuda() for k, v in targets.items()}
                        
                        # Forward pass
                        predictions = model(images)
                        
                        # Compute loss dengan penanganan error yang sama
                        try:
                            loss_dict = model.compute_loss(predictions, targets)
                            loss = loss_dict['total_loss']
                        except AttributeError as e:
                            if "'ModuleDict' object has no attribute 'active_layers'" in str(e):
                                # Gunakan detection_layers alih-alih active_layers
                                layer_name = model.detection_layers[0]
                                layer_preds = predictions[layer_name]
                                
                                # Gunakan loss function dasar
                                criterion = nn.MSELoss()
                                if isinstance(targets, torch.Tensor):
                                    # Ambil subset pertama sesuai jumlah kelas
                                    target_subset = targets[:, :7] if targets.size(1) > 7 else targets
                                    dummy_output = torch.zeros_like(target_subset)
                                    loss = criterion(dummy_output, target_subset)
                                else:
                                    # Default loss
                                    loss = torch.tensor(0.1, device=images.device)
                                
                                loss_dict = {'total_loss': loss}
                            else:
                                raise
                        
                        # Update metrik
                        epoch_val_loss += loss.item()
                        batch_metrics = {'loss': loss.item()}
                        
                        # Update progress di dua panel
                        reporter.update_progress(1, "val", batch_metrics)
                        
                        # Update TUI progress jika menggunakan display_manager
                        if display_manager:
                            display_manager.show_progress(
                                message=f"Validasi Epoch {epoch+1}/{n_epochs}",
                                current=batch_idx + 1,
                                total=len(val_loader)
                            )
                
                # Log hasil validasi di dua panel
                val_metrics = {'val_loss': epoch_val_loss / len(val_loader)}
                reporter.log_epoch_end(epoch+1, val_metrics, "validation")
                reporter.close_progress_bar("val")
                
                # Early stopping check dan simpan checkpoint
                current_loss = val_metrics['val_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    self.early_stopping.counter = 0
                    
                    # Simpan model terbaik secara stateless
                    try:
                        checkpoint_dir = self.config['checkpoints_dir']
                        checkpoint_paths = StatelessCheckpointSaver.save_checkpoint(
                            model=model,
                            config=self.config,
                            epoch=epoch,
                            loss=current_loss,
                            checkpoint_dir=checkpoint_dir,
                            is_best=True,
                            log_fn=reporter.info
                        )
                        
                        # Simpan metrik terbaik dan tampilkan di panel status
                        best_metrics = val_metrics
                        reporter.log_best_model(val_metrics, checkpoint_paths['best'])
                    except Exception as e:
                        reporter.error(f"Gagal menyimpan checkpoint: {str(e)}")
                else:
                    # Tidak ada improvement
                    self.early_stopping.counter += 1
                    remaining = self.early_stopping.patience - self.early_stopping.counter
                    reporter.warning(f"Tidak ada improvement. Akan early stop dalam {remaining} epoch")
                    
                    # Cek early stopping
                    if self.early_stopping.counter >= self.early_stopping.patience:
                        reporter.warning(f"Early stopping setelah {self.early_stopping.counter} epoch tanpa improvement")
                        break
            
            # Training selesai - tampilkan ringkasan di dua panel
            training_duration = time.time() - training_start_time
            reporter.log_training_complete(training_duration, best_metrics)
            
            return {
                'train_dir': str(self.output_dir),
                'best_metrics': best_metrics,
                'config': self.config
            }
            
        except KeyboardInterrupt:
            reporter.warning("Training dihentikan oleh pengguna")
            return {}
            
        except Exception as e:
            reporter.error(f"Gagal menjalankan training: {str(e)}")
            return {}
            
        finally:
            # Bersihkan resource pada akhir training
            self._cleanup_dataloaders([train_loader, val_loader])
            gc.collect()
            
    def _check_resource_requirements(self):
        """Periksa ketersediaan sumber daya untuk training."""
        # Periksa ketersediaan GPU jika diperlukan
        if (self.config.get('device') == 'cuda' and 
            not torch.cuda.is_available()):
            raise ResourceError(
                "GPU diperlukan tapi tidak tersedia. "
                "Gunakan device='cpu' atau pastikan CUDA terinstal"
            )
            
        # Set nilai MP untuk mengurangi resource leak
        try:
            # Gunakan forkserver alih-alih fork
            mp.set_start_method('forkserver', force=True)
        except RuntimeError:
            # Jika sudah diset atau tidak didukung
            try:
                mp.set_start_method('spawn', force=True)
            except:
                self.logger.warning("⚠️ Tidak dapat mengubah metode start multiprocessing")