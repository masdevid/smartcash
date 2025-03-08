# File: smartcash/handlers/model_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler terkonsolidasi untuk model training, evaluasi, dan inferensi 
# dengan dukungan multi-backbone dan multilayer detection

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any, Tuple
import yaml
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel
from smartcash.handlers.checkpoint_handler import CheckpointHandler
from smartcash.utils.visualization import plot_training_metrics, plot_confusion_matrix, plot_detections

class ModelHandler:
    """
    Handler terkonsolidasi untuk model training, evaluasi, dan manajemen 
    dengan dukungan multi-backbone dan deteksi multilayer
    """
    
    def __init__(
        self,
        config: Dict,
        config_path: str = "configs/base_config.yaml",
        num_classes: int = 7,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi model handler.
        
        Args:
            config: Konfigurasi model dan training
            config_path: Path ke file konfigurasi (opsional)
            num_classes: Jumlah kelas (jika None, akan diambil dari config)
            logger: Custom logger (opsional)
        """
        self.config = config  
        self.config_path = Path(config_path) if config_path else None
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = num_classes or config.get('model', {}).get('num_classes', 7)
        
        # Setup checkpoint handler
        checkpoints_dir = config.get('output_dir', 'runs/train') + '/weights'
        self.checkpoint_handler = CheckpointHandler(
            output_dir=checkpoints_dir,
            logger=self.logger
        )
        
        # Track hasil training
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': {}
        }
        
        # Konfigurasi layer deteksi
        self.layers = config.get('layers', ['banknote'])
        
        # Track results eksperimen
        self.results = {}
        
        self.logger.info(
            f"🔧 ModelHandler diinisialisasi:\n"
            f"   • Num classes: {self.num_classes}\n"
            f"   • Detection layers: {self.layers}\n"
            f"   • Checkpoints: {checkpoints_dir}"
        )
    
    def create_model(self, backbone_type: Optional[str] = None) -> Union[YOLOv5Model, BaselineModel]:
        """
        Membuat model dengan backbone dan konfigurasi yang ditentukan.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            
        Returns:
            Model yang siap digunakan
        """
        # Prioritaskan backbone yang diberikan, atau gunakan dari config
        backbone_type = backbone_type or self.config.get('model', {}).get('backbone', 'cspdarknet')
        
        # Parameter lain dari config
        pretrained = self.config.get('model', {}).get('pretrained', True)
        detection_layers = self.config.get('layers', ['banknote'])
        
        # Log informasi model
        self.logger.info(
            f"🔄 Membuat model dengan backbone: {backbone_type}\n"
            f"   • Pretrained: {pretrained}\n"
            f"   • Detection layers: {detection_layers}\n"
            f"   • Num classes: {self.num_classes}"
        )
        
        # Buat model sesuai tipe backbone
        try:
            if backbone_type in ['efficientnet', 'cspdarknet']:
                model = YOLOv5Model(
                    num_classes=self.num_classes,
                    backbone_type=backbone_type,
                    pretrained=pretrained,
                    detection_layers=detection_layers,
                    logger=self.logger
                )
            else:
                # Fallback untuk backbone lain
                model = BaselineModel(
                    num_classes=self.num_classes,
                    backbone=backbone_type,
                    pretrained=pretrained
                )
                
            self.logger.success(f"✅ Model berhasil dibuat dengan backbone {backbone_type}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat model: {str(e)}")
            raise e
    
    def get_model(self, backbone_type: Optional[str] = None) -> Union[YOLOv5Model, BaselineModel]:
        """
        Alias untuk create_model untuk kompatibilitas dengan kode lama
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            
        Returns:
            Model yang siap digunakan
        """
        return self.create_model(backbone_type)
            
    def get_optimizer(
        self, 
        model: torch.nn.Module, 
        lr: Optional[float] = None,
        clip_grad_norm: Optional[float] = None
    ) -> torch.optim.Optimizer:
        """
        Membuat optimizer untuk model.
        
        Args:
            model: Model yang akan dioptimasi
            lr: Learning rate (jika None, akan diambil dari config)
            clip_grad_norm: Nilai maksimum untuk gradient clipping (opsional)
            
        Returns:
            Optimizer yang sesuai dengan konfigurasi
        """
        # Ambil parameter dari config
        learning_rate = lr or self.config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.0005)
        
        self.logger.info(f"🔧 Membuat optimizer dengan learning rate {learning_rate}")
        
        # Pilih optimizer berdasarkan konfigurasi
        optimizer_type = self.config.get('training', {}).get('optimizer', 'adam').lower()
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('training', {}).get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"⚠️ Tipe optimizer '{optimizer_type}' tidak dikenal, menggunakan Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
        # Tambahkan gradient clipping jika diminta
        if clip_grad_norm is not None:
            for group in optimizer.param_groups:
                group.setdefault('max_grad_norm', clip_grad_norm)
            
            # Wrap step method dengan gradient clipping
            original_step = optimizer.step
            
            def step_with_clip(*args, **kwargs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                return original_step(*args, **kwargs)
                
            optimizer.step = step_with_clip
            self.logger.info(f"🔒 Gradient clipping diaktifkan dengan max_norm={clip_grad_norm}")
            
        return optimizer
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """
        Membuat learning rate scheduler berdasarkan konfigurasi.
        
        Args:
            optimizer: Optimizer yang akan dijadwalkan
            
        Returns:
            Learning rate scheduler
        """
        scheduler_type = self.config.get('training', {}).get('scheduler', 'plateau').lower()
        
        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('training', {}).get('lr_step_size', 10)
            gamma = self.config.get('training', {}).get('lr_gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'cosine':
            epochs = self.config.get('training', {}).get('epochs', 30)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )
        else:
            self.logger.warning(f"⚠️ Tipe scheduler '{scheduler_type}' tidak dikenal, menggunakan ReduceLROnPlateau")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def save_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int, 
        metrics: Dict,
        is_best: bool = False
    ) -> Dict[str, str]:
        """
        Simpan model checkpoint.
        
        Args:
            model: Model yang akan disimpan
            optimizer: Optimizer yang digunakan
            scheduler: Learning rate scheduler
            epoch: Epoch saat ini
            metrics: Metrik evaluasi
            is_best: Flag untuk menandai checkpoint terbaik
            
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        return self.checkpoint_handler.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
    
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mencari checkpoint terbaik)
            device: Device untuk menempatkan model
            
        Returns:
            Tuple (Model yang dimuat dari checkpoint, metadata checkpoint)
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cari checkpoint terbaik jika tidak ada path yang diberikan
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_handler.find_best_checkpoint()
            if checkpoint_path is None:
                self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan, membuat model baru")
                model = self.create_model()
                return model, {'epoch': 0, 'metrics': {}}
                
        try:
            # Muat checkpoint
            checkpoint = self.checkpoint_handler.load_checkpoint(checkpoint_path)
            checkpoint_config = checkpoint.get('config', {})
            
            # Dapatkan informasi backbone dari checkpoint
            backbone = checkpoint_config.get('model', {}).get('backbone', 
                    self.config.get('model', {}).get('backbone', 'efficientnet'))
            
            # Buat model baru dengan konfigurasi yang sama dengan checkpoint
            model = self.create_model(backbone_type=backbone)
            
            # Muat state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Pindahkan model ke device
            model = model.to(device)
            
            # Log informasi
            self.logger.success(
                f"✅ Model berhasil dimuat dari checkpoint:\n"
                f"   • Path: {checkpoint_path}\n"
                f"   • Epoch: {checkpoint.get('epoch', 'unknown')}\n"
                f"   • Loss: {checkpoint.get('metrics', {}).get('loss', 'unknown')}\n"
                f"   • Backbone: {backbone}"
            )
            
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise e
            
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
        Jalankan training loop untuk model.
        
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
        
        # Buat model jika belum ada
        if model is None:
            if resume_from:
                model, checkpoint = self.load_model(resume_from, device)
                start_epoch = checkpoint.get('epoch', 0) + 1
                self.logger.info(f"📋 Melanjutkan training dari epoch {start_epoch}")
            else:
                model = self.create_model()
                model = model.to(device)
                start_epoch = 0
        else:
            model = model.to(device)
            start_epoch = 0
        
        # Buat optimizer dan scheduler
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        
        # Track best metrics
        best_val_loss = float('inf')
        best_checkpoint_path = None
        
        # Early stopping
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)
        early_stopping_counter = 0
        
        self.logger.info(
            f"🚀 Memulai training:\n"
            f"   • Epochs: {epochs}\n"
            f"   • Device: {device}\n"
            f"   • Early stopping patience: {early_stopping_patience}"
        )
        
        # Training loop
        train_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = self._train_epoch(model, optimizer, train_loader, device)
            
            # Validation phase
            model.eval()
            val_loss, val_metrics = self._validate_epoch(model, val_loader, device)
            
            # Update learning rate scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Simpan learning rate saat ini
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
            
            # Log hasil epoch
            self.logger.info(
                f"📊 Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Simpan checkpoint
            is_best = val_loss < best_val_loss
            
            if is_best:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Simpan checkpoint
            checkpoint_info = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'loss': val_loss,
                    **val_metrics
                },
                is_best=is_best
            )
            
            if is_best and 'path' in checkpoint_info:
                best_checkpoint_path = checkpoint_info['path']
                self.logger.info(f"🌟 Model terbaik diperbarui: {best_checkpoint_path}")
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(
                    f"⏹️ Early stopping diaktifkan setelah {early_stopping_counter} "
                    f"epoch tanpa peningkatan"
                )
                break
                
            # Plot training progress untuk informasi
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self._plot_training_progress()
        
        # Hitung total waktu training
        total_time = time.time() - train_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.success(
            f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"   • Best val loss: {best_val_loss:.4f}\n"
            f"   • Best checkpoint: {best_checkpoint_path}"
        )
        
        return {
            'history': self.training_history,
            'best_val_loss': best_val_loss,
            'best_checkpoint_path': best_checkpoint_path,
            'total_epochs': epoch + 1,
            'total_time': total_time
        }
    
    def _train_epoch(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
        """
        Jalankan satu epoch training.
        
        Args:
            model: Model yang akan dilatih
            optimizer: Optimizer yang digunakan
            train_loader: DataLoader untuk training
            device: Device untuk komputasi
            
        Returns:
            Rata-rata loss untuk epoch ini
        """
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, data in enumerate(pbar):
            # Handle different data formats
            if isinstance(data, dict):
                # Multilayer dataset format
                images = data['images'].to(device)
                targets = data['targets']
                
                # Transfer targets to device
                for layer_name in targets:
                    targets[layer_name] = targets[layer_name].to(device)
            else:
                # Standard format
                images, targets = data
                images = images.to(device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                
            # Forward pass
            predictions = model(images)
            loss_dict = model.compute_loss(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)
    
    def _validate_epoch(
        self, 
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[float, Dict]:
        """
        Jalankan satu epoch validasi dengan pengukuran metrik.
        
        Args:
            model: Model yang akan divalidasi
            val_loader: DataLoader untuk validasi
            device: Device untuk komputasi
            
        Returns:
            Tuple (rata-rata validation loss, dict metrik)
        """
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        pbar = tqdm(val_loader, desc="Validasi", leave=False)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
                # Handle different data formats
                if isinstance(data, dict):
                    # Multilayer dataset format
                    images = data['images'].to(device)
                    targets = data['targets']
                    
                    # Transfer targets to device
                    for layer_name in targets:
                        targets[layer_name] = targets[layer_name].to(device)
                else:
                    # Standard format
                    images, targets = data
                    images = images.to(device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(device)
                    
                # Forward pass
                predictions = model(images)
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                if not isinstance(targets, dict):
                    # Flatten predictions and targets for standard format
                    try:
                        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                        true_classes = torch.argmax(targets, dim=1).cpu().numpy()
                        
                        all_predictions.extend(pred_classes)
                        all_targets.extend(true_classes)
                    except:
                        # Skip if can't convert for metrics
                        pass
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate validation metrics if available
        metrics = {}
        if all_targets and all_predictions:
            try:
                # Hitung precision, recall, F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='weighted'
                )
                
                # Hitung accuracy
                accuracy = accuracy_score(all_targets, all_predictions)
                
                metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menghitung metrik validasi: {str(e)}")
                
        return total_loss / len(val_loader), metrics
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model yang akan dievaluasi (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            device: Device untuk evaluasi
            
        Returns:
            Dict berisi metrik evaluasi
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Muat model jika belum ada
        if model is None:
            if checkpoint_path:
                model, _ = self.load_model(checkpoint_path, device)
            else:
                self.logger.error("❌ Tidak ada model yang diberikan, dan tidak ada checkpoint path")
                raise ValueError("Model atau checkpoint_path harus diberikan")
        else:
            model = model.to(device)
        
        model.eval()
        
        # Evaluasi model
        self.logger.info("🔍 Mengevaluasi model pada test dataset...")
        
        # Track total loss dan prediksi
        total_loss = 0
        all_targets = []
        all_predictions = []
        all_pred_scores = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluasi")):
                # Handle different data formats
                if isinstance(data, dict):
                    # Multilayer dataset format
                    images = data['images'].to(device)
                    targets = data['targets']
                    
                    # Transfer targets to device
                    for layer_name in targets:
                        targets[layer_name] = targets[layer_name].to(device)
                else:
                    # Standard format
                    images, targets = data
                    images = images.to(device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Compute loss
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                if not isinstance(targets, dict):
                    # Format predictions and targets for metrics
                    try:
                        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                        true_classes = torch.argmax(targets, dim=1).cpu().numpy()
                        
                        all_predictions.extend(pred_classes)
                        all_targets.extend(true_classes)
                        
                        # Save prediction scores for ROC curves
                        all_pred_scores.extend(predictions.cpu().numpy())
                    except:
                        # Skip if can't convert for metrics
                        pass
        
        # Calculate average loss and inference time
        avg_loss = total_loss / len(test_loader)
        avg_inference_time = np.mean(inference_times)
        
        # Calculate metrics
        metrics = {
            'loss': avg_loss,
            'inference_time': avg_inference_time
        }
        
        # Additional metrics if we have targets and predictions
        if all_targets and all_predictions:
            try:
                # Calculate precision, recall, F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='weighted'
                )
                
                # Calculate accuracy
                accuracy = accuracy_score(all_targets, all_predictions)
                
                # Calculate confusion matrix
                cm = confusion_matrix(all_targets, all_predictions)
                
                # Add metrics
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'confusion_matrix': cm
                })
                
                # Plot confusion matrix jika memungkinkan
                try:
                    class_names = [str(i) for i in range(model.num_classes)]
                    cm_fig = plot_confusion_matrix(cm, class_names)
                    # Simpan plot ke file
                    output_dir = Path(self.config.get('output_dir', 'runs/eval'))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cm_fig.savefig(output_dir / 'confusion_matrix.png')
                    plt.close(cm_fig)
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membuat plot confusion matrix: {str(e)}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menghitung metrik evaluasi: {str(e)}")
        
        # Log hasil evaluasi
        self.logger.success(
            f"✅ Evaluasi selesai:\n"
            f"   • Loss: {metrics.get('loss', 'N/A'):.4f}\n"
            f"   • Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n"
            f"   • Precision: {metrics.get('precision', 'N/A'):.4f}\n"
            f"   • Recall: {metrics.get('recall', 'N/A'):.4f}\n"
            f"   • F1 Score: {metrics.get('f1', 'N/A'):.4f}\n"
            f"   • Inference Time: {metrics.get('inference_time', 'N/A')*1000:.2f} ms/batch"
        )
        
        return metrics
    
    def predict(
        self,
        images: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        Lakukan prediksi pada satu atau beberapa gambar.
        
        Args:
            images: Tensor gambar input [B, C, H, W]
            model: Model yang akan digunakan (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            device: Device untuk prediksi
            conf_threshold: Confidence threshold untuk prediksi
            
        Returns:
            Dict berisi hasil prediksi
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Muat model jika belum ada
        if model is None:
            if checkpoint_path:
                model, _ = self.load_model(checkpoint_path, device)
            else:
                self.logger.error("❌ Tidak ada model yang diberikan, dan tidak ada checkpoint path")
                raise ValueError("Model atau checkpoint_path harus diberikan")
        else:
            model = model.to(device)
        
        # Pastikan model dalam mode evaluasi
        model.eval()
        
        # Pindahkan gambar ke device
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        # Tambahkan dimensi batch jika hanya 1 gambar
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        images = images.to(device)
        
        # Lakukan prediksi
        with torch.no_grad():
            # Measure inference time
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            
            # Format yang berbeda untuk YOLOv5Model dan BaselineModel
            if isinstance(model, YOLOv5Model):
                # Gunakan predict dengan threshold
                detections = model.predict(images, conf_threshold=conf_threshold)
            else:
                # Format output untuk model baseline
                detections = []
                for pred in predictions:
                    scores, indices = torch.max(torch.softmax(pred, dim=0), dim=0)
                    mask = scores > conf_threshold
                    
                    detection = {
                        'boxes': torch.tensor([]),  # Empty tensor for boxes
                        'scores': scores[mask],
                        'labels': indices[mask]
                    }
                    detections.append(detection)
        
        # Format hasil
        result = {
            'detections': detections,
            'inference_time': inference_time,
            'conf_threshold': conf_threshold
        }
        
        self.logger.info(
            f"🔍 Prediksi selesai dalam {inference_time*1000:.2f} ms\n"
            f"   • Batch size: {images.shape[0]}\n"
            f"   • Confidence threshold: {conf_threshold}"
        )
        
        return result
    
    def freeze_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Bekukan layer backbone untuk fine-tuning.
        
        Args:
            model: Model dengan backbone yang akan dibekukan
            
        Returns:
            Model dengan backbone yang dibekukan
        """
        if isinstance(model, YOLOv5Model):
            # Bekukan backbone YOLOv5
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            self.logger.info("🧊 Backbone telah dibekukan untuk fine-tuning")
            return model
        elif isinstance(model, BaselineModel):
            # Bekukan backbone BaselineModel
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            self.logger.info("🧊 Backbone telah dibekukan untuk fine-tuning")
            return model
        else:
            self.logger.warning("⚠️ Tipe model tidak dikenal, tidak dapat membekukan backbone")
            return model
    
    def unfreeze_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Lepaskan pembekuan layer backbone.
        
        Args:
            model: Model dengan backbone yang akan dilepas pembekuannya
            
        Returns:
            Model dengan backbone yang dilepas pembekuannya
        """
        if isinstance(model, YOLOv5Model):
            # Unfreeze backbone YOLOv5
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            self.logger.info("🔥 Backbone telah dilepas pembekuannya")
            return model
        elif isinstance(model, BaselineModel):
            # Unfreeze backbone BaselineModel
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            self.logger.info("🔥 Backbone telah dilepas pembekuannya")
            return model
        else:
            self.logger.warning("⚠️ Tipe model tidak dikenal, tidak dapat melepas pembekuan backbone")
            return model
    
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
            
            # Plot learning rate
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
            
    def run_experiment(
        self,
        scenario: Dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Jalankan eksperimen berdasarkan skenario.
        
        Args:
            scenario: Konfigurasi skenario eksperimen
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            
        Returns:
            Dict berisi hasil eksperimen
        """
        self.logger.info(f"🧪 Memulai eksperimen: {scenario['name']}")
        self.logger.info(f"📝 Deskripsi: {scenario['description']}")
        
        # Simpan konfigurasi awal
        original_config = self.config.copy()
        
        try:
            # Update konfigurasi sesuai skenario
            if 'backbone' in scenario:
                self.config['model']['backbone'] = scenario['backbone']
                
            # Buat model sesuai skenario
            model = self.create_model()
            
            # Training model dengan skenario
            training_results = self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model
            )
            
            # Evaluasi model pada test set
            eval_metrics = self.evaluate(
                test_loader=test_loader,
                checkpoint_path=training_results['best_checkpoint_path']
            )
            
            # Tambahkan metrik evaluasi ke hasil
            full_results = {
                **training_results,
                'metrics': eval_metrics,
                'scenario': scenario
            }
            
            # Simpan hasil
            self.results[scenario['name']] = full_results
            
            # Log hasil
            self.logger.success(
                f"✅ Eksperimen selesai: {scenario['name']}\n"
                f"📊 Hasil:\n"
                f"   • Best Val Loss: {training_results.get('best_val_loss', 'N/A'):.4f}\n"
                f"   • Test Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}\n"
                f"   • Precision: {eval_metrics.get('precision', 'N/A'):.4f}\n"
                f"   • Recall: {eval_metrics.get('recall', 'N/A'):.4f}\n"
                f"   • F1 Score: {eval_metrics.get('f1', 'N/A'):.4f}\n"
                f"   • Inference Time: {eval_metrics.get('inference_time', 'N/A')*1000:.2f} ms/batch"
            )
            
            return full_results
            
        except Exception as e:
            self.logger.error(f"❌ Eksperimen gagal: {str(e)}")
            raise e
            
        finally:
            # Restore konfigurasi asli
            self.config = original_config
            
    def compare_models(
        self,
        models: List[Dict],
        test_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Bandingkan beberapa model checkpoint pada dataset yang sama.
        
        Args:
            models: List dict yang berisi informasi model {'name': 'Model 1', 'path': 'path/to/ckpt.pth'}
            test_loader: DataLoader untuk testing
            device: Device untuk evaluasi
            
        Returns:
            Dict berisi hasil perbandingan
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = {}
        
        for model_info in models:
            name = model_info['name']
            path = model_info['path']
            
            self.logger.info(f"🔍 Mengevaluasi model: {name}")
            
            try:
                # Evaluasi model
                metrics = self.evaluate(
                    test_loader=test_loader,
                    checkpoint_path=path,
                    device=device
                )
                
                # Simpan hasil
                results[name] = metrics
                
            except Exception as e:
                self.logger.error(f"❌ Gagal mengevaluasi model {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Bandingkan hasil
        self.logger.info("📊 Hasil perbandingan model:")
        comparison_table = "| Model | Accuracy | Precision | Recall | F1 | Inference Time (ms) |\n"
        comparison_table += "| --- | --- | --- | --- | --- | --- |\n"
        
        for name, metrics in results.items():
            if 'error' not in metrics:
                comparison_table += f"| {name} | {metrics.get('accuracy', 'N/A'):.4f} | "
                comparison_table += f"{metrics.get('precision', 'N/A'):.4f} | "
                comparison_table += f"{metrics.get('recall', 'N/A'):.4f} | "
                comparison_table += f"{metrics.get('f1', 'N/A'):.4f} | "
                comparison_table += f"{metrics.get('inference_time', 'N/A')*1000:.2f} |\n"
            else:
                comparison_table += f"| {name} | Error: {metrics['error']} |\n"
        
        self.logger.info(f"\n{comparison_table}")
        
        return results