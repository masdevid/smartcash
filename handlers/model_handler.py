# File: handlers/model_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler terkonsolidasi untuk model training, evaluasi, dan manajemen dengan dukungan multi-backbone

import os
from typing import Dict, Optional, List, Union, Any
import yaml
import time
import torch
from pathlib import Path
from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel
from smartcash.handlers.checkpoint_handler import CheckpointHandler

class ModelHandler:
    """Handler terkonsolidasi untuk model training, evaluasi, dan manajemen dengan dukungan multi-backbone"""
    
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
        self.checkpoint_handler = CheckpointHandler(
            checkpoints_dir=config.get('output_dir', 'runs/train') + '/weights',
            logger=self.logger
        )
        
        # Track results eksperimen
        self.results = {}
    
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
                return YOLOv5Model(
                    num_classes=self.num_classes,
                    backbone_type=backbone_type,
                    pretrained=pretrained,
                    detection_layers=detection_layers,
                    logger=self.logger
                )
            else:
                # Fallback untuk backbone lain
                return BaselineModel(
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
        epoch: int, 
        loss: float, 
        is_best: bool = False
    ) -> Dict[str, str]:
        """
        Simpan model checkpoint.
        
        Args:
            model: Model yang akan disimpan
            epoch: Epoch saat ini
            loss: Loss terakhir
            is_best: Flag untuk menandai checkpoint terbaik
            
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        return self.checkpoint_handler.save_checkpoint(
            model=model,
            config=self.config,
            epoch=epoch,
            metrics={'loss': loss},
            is_best=is_best
        )
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mencari checkpoint terbaik)
            
        Returns:
            Model yang dimuat dari checkpoint
        """
        # Cari checkpoint terbaik jika tidak ada path yang diberikan
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_handler.find_best_checkpoint()
            if checkpoint_path is None:
                self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan, membuat model baru")
                return self.create_model()
                
        try:
            # Muat checkpoint
            checkpoint = self.checkpoint_handler.load_checkpoint(checkpoint_path)
            checkpoint_config = checkpoint.get('config', {})
            
            # Dapatkan informasi backbone dari checkpoint
            backbone = checkpoint_config.get('model', {}).get('backbone', self.config.get('model', {}).get('backbone', 'efficientnet'))
            
            # Buat model baru dengan konfigurasi yang sama dengan checkpoint
            model = self.create_model(backbone_type=backbone)
            
            # Muat state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Log informasi
            self.logger.success(
                f"✅ Model berhasil dimuat dari checkpoint:\n"
                f"   • Path: {checkpoint_path}\n"
                f"   • Epoch: {checkpoint.get('epoch', 'unknown')}\n"
                f"   • Loss: {checkpoint.get('metrics', {}).get('loss', 'unknown')}\n"
                f"   • Backbone: {backbone}"
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise e
            
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
            
            # Pindahkan ke GPU jika tersedia
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Setup optimizer dan scheduler
            optimizer = self.get_optimizer(model)
            scheduler = self.get_scheduler(optimizer)
            
            # Training loop
            start_time = time.time()
            epochs = self.config.get('training', {}).get('epochs', 30)
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = self._train_epoch(model, optimizer, train_loader, device)
                
                # Validation
                model.eval()
                val_loss = self._validate_epoch(model, val_loader, device)
                
                # Update scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                    
                # Log progress
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
            # Evaluasi final
            model.eval()
            metrics = self._evaluate_model(model, test_loader, device)
            
            # Tambahkan waktu eksperimen
            total_time = time.time() - start_time
            metrics['training_time'] = total_time
            
            # Simpan hasil
            self.results[scenario['name']] = {
                'metrics': metrics,
                'config': self.config.copy()
            }
            
            # Log hasil
            self.logger.success(
                f"✅ Eksperimen selesai: {scenario['name']}\n"
                f"📊 Hasil:\n"
                f"   • Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                f"   • mAP: {metrics.get('mAP', 0):.4f}\n"
                f"   • Inference time: {metrics.get('inference_time', 0)*1000:.2f}ms\n"
                f"   • Training time: {total_time:.2f}s"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Eksperimen gagal: {str(e)}")
            raise e
            
        finally:
            # Restore konfigurasi asli
            self.config = original_config
            
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
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move to device
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
            
        return total_loss / len(train_loader)
    
    def _validate_epoch(
        self, 
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
        """
        Jalankan satu epoch validasi.
        
        Args:
            model: Model yang akan divalidasi
            val_loader: DataLoader untuk validasi
            device: Device untuk komputasi
            
        Returns:
            Rata-rata validation loss
        """
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Move to device
                images = images.to(device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                    
                # Forward pass
                predictions = model(images)
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def _evaluate_model(
        self, 
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict:
        """
        Evaluasi model pada test set.
        
        Args:
            model: Model yang akan dievaluasi
            test_loader: DataLoader untuk testing
            device: Device untuk komputasi
            
        Returns:
            Dict berisi metrik evaluasi
        """
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'mAP': 0,
            'inference_time': 0
        }
        
        # Implementasi evaluasi model disini
        # ...
        
        return metrics