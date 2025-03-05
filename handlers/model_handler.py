# File: handlers/model_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk model training dan evaluasi dengan perbaikan untuk ModuleDict.active_layers

import os
from typing import Dict, Optional, List, Union
import yaml
import time
import torch
from pathlib import Path
from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel
from smartcash.utils.model_checkpoint import StatelessCheckpointSaver

class ModelHandler:
    """Handler untuk training dan evaluasi model"""
    
    def __init__(
        self,
        config: Dict,
        config_path: str,
        num_classes: int,
        logger: Optional[SmartCashLogger] = None
    ):
        self.config_path = Path(config_path)  # Store config path
        self.logger = logger or SmartCashLogger(__name__)
        
        # Load configuration
        self.config = config or self._load_config(config_path)
        
        self.num_classes = num_classes
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate critical configurations
            if not config:
                raise ValueError("Konfigurasi kosong atau tidak valid")
            
            return config
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            raise
    
    def get_model(self) -> Union[YOLOv5Model, BaselineModel]:
        """
        Inisialisasi model dengan dukungan backbone fleksibel
        
        Returns:
            Model yang siap untuk training
        """
        try:
            # Prioritaskan backbone dari konfigurasi
            backbone_type = (
                self.config.get('backbone') or 'cspdarknet'
            )
            
            # Pastikan detection_layers diambil dari config
            detection_layers = self.config.get('layers', ['banknote'])
            
            # Parameter tambahan
            pretrained = self.config.get('model', {}).get('pretrained', True)
            
            # Log konfirmasi backbone yang dipilih
            self.logger.info(
                f"🚀 Mempersiapkan model dengan:\n"
                f"   • Backbone: {backbone_type}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Jumlah Layer: {len(detection_layers)}\n"
                f"   • Jumlah Kelas: {self.num_classes}"
            )
            
            # Inisialisasi model dengan backbone yang dipilih
            if backbone_type == 'efficientnet':
                model = YOLOv5Model(
                    num_classes=self.num_classes,
                    backbone_type='efficientnet',
                    pretrained=pretrained,
                    detection_layers=detection_layers,  # Gunakan detection_layers, bukan layers
                    logger=self.logger
                )
            elif backbone_type == 'cspdarknet':
                model = YOLOv5Model(
                    num_classes=self.num_classes,
                    backbone_type='cspdarknet',
                    pretrained=pretrained,
                    detection_layers=detection_layers,  # Gunakan detection_layers, bukan layers
                    logger=self.logger
                )
            else:
                # Fallback untuk backbone khusus/eksperimental
                model = BaselineModel(
                    num_classes=self.num_classes,
                    backbone=backbone_type,
                    pretrained=pretrained
                )
            
            # Pindahkan ke GPU jika tersedia
            if torch.cuda.is_available():
                model = model.cuda()
                self.logger.info("💻 Model dialihkan ke GPU")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mempersiapkan model: {str(e)}")
            raise e
            
    def run_experiment(
        self,
        scenario: Dict,
        train_path: str,
        val_path: str,
        test_path: str
    ) -> Dict:
        """Jalankan eksperimen untuk satu skenario"""
        self.logger.start(
            f"Memulai eksperimen: {scenario['name']}\n"
            f"Deskripsi: {scenario['description']}"
        )
        
        try:
            # Initialize model sesuai skenario
            model = self.get_model()
            
            # Build & train model
            start_time = time.time()
            
            model.build()
            train_results = model.train(
                train_path=train_path,
                val_path=val_path,
                epochs=self.config['training']['epochs']
            )
            
            # Evaluasi
            eval_results = model.evaluate(
                test_path=test_path,
                save_visualizations=self.config['evaluation']['save_visualizations']
            )
            
            # Hitung inference time
            inference_time = (time.time() - start_time) / len(os.listdir(test_path))
            eval_results['inference_time'] = inference_time
            
            # Simpan hasil
            self.results[scenario['name']] = {
                'training': train_results,
                'evaluation': eval_results
            }
            
            self.logger.success(
                f"Eksperimen {scenario['name']} selesai!\n"
                f"Inference time: {inference_time:.4f} s/img"
            )
            
            return eval_results
            
        except Exception as e:
            self.logger.error(
                f"Eksperimen {scenario['name']} gagal: {str(e)}"
            )
            raise
    
    def run_all_experiments(
        self,
        train_path: str,
        val_path: str,
        test_path: str
    ) -> Dict:
        """Jalankan semua skenario eksperimen"""
        all_results = {}
        
        for scenario in self.config['experiment_scenarios']:
            results = self.run_experiment(
                scenario=scenario,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path
            )
            all_results[scenario['name']] = results
            
        return all_results
        
    def get_optimizer(self, model, lr=None):
        """Dapatkan optimizer untuk model"""
        learning_rate = lr or self.config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.0005)
        
        self.logger.info(f"🔧 Membuat optimizer dengan learning rate {learning_rate}")
        
        # Pilih optimizer berdasarkan konfigurasi
        optimizer_type = self.config.get('training', {}).get('optimizer', 'adam').lower()
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('training', {}).get('momentum', 0.9)
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"⚠️ Tipe optimizer '{optimizer_type}' tidak dikenal, menggunakan Adam")
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
    
    def get_scheduler(self, optimizer):
        """Dapatkan learning rate scheduler berdasarkan konfigurasi"""
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
    
    def save_checkpoint(self, model, epoch, loss, is_best=False):
        """Simpan checkpoint model"""
        checkpoint_dir = Path(self.config.get('output_dir', 'runs/train')) / 'weights'
        
        try:
            # Pastikan direktori ada
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Simpan model menggunakan StatelessCheckpointSaver
            return StatelessCheckpointSaver.save_checkpoint(
                model=model,
                config=self.config,
                epoch=epoch,
                loss=loss,
                checkpoint_dir=str(checkpoint_dir),
                is_best=is_best,
                log_fn=self.logger.info
            )
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            self.logger.error(f"📋 Detail: {str(e)}")
            return None
    
    def load_model(self, checkpoint_path=None):
        """Muat model dari checkpoint dengan dukungan fleksibilitas layer"""
        # Ambil direktori checkpoint
        checkpoint_dir = Path(self.config.get('output_dir', 'runs/train')) / 'weights'
        
        # Jika tidak ada path yang diberikan, cari checkpoint terbaik
        if checkpoint_path is None:
            best_checkpoints = list(checkpoint_dir.glob("*_best.pth"))
            if best_checkpoints:
                checkpoint_path = str(max(best_checkpoints, key=os.path.getmtime))
                self.logger.info(f"📂 Menggunakan checkpoint terbaik: {checkpoint_path}")
            else:
                latest_checkpoints = list(checkpoint_dir.glob("*_latest.pth"))
                if latest_checkpoints:
                    checkpoint_path = str(max(latest_checkpoints, key=os.path.getmtime))
                    self.logger.info(f"📂 Menggunakan checkpoint terakhir: {checkpoint_path}")
                else:
                    self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan")
                    return self.get_model()
                  
        try:
            # Muat checkpoint terlebih dahulu untuk mendapatkan informasi konfigurasi
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_config = checkpoint.get('config', {})
            
            # Dapatkan informasi backbone dan layer dari checkpoint
            backbone_type = checkpoint_config.get('model', {}).get('backbone', self.config.get('model', {}).get('backbone', 'efficientnet'))
            checkpoint_layers = checkpoint_config.get('layers', self.config.get('layers', ['banknote']))
            
            # Perbarui detection_layers pada instance jika diperlukan
            current_layers = self.config.get('layers', ['banknote'])
            if set(checkpoint_layers) != set(current_layers):
                self.logger.warning(f"⚠️ Layer pada checkpoint ({checkpoint_layers}) berbeda dengan konfigurasi saat ini ({current_layers})")
                self.logger.info(f"ℹ️ Menggunakan layer dari checkpoint: {checkpoint_layers}")
                
                # Update config dengan layer dari checkpoint
                self.config['layers'] = checkpoint_layers
            
            # Buat model baru dengan konfigurasi yang sama dengan checkpoint
            model = self.get_model()
            
            # Muat state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.success(f"✅ Model berhasil dimuat dari {checkpoint_path}")
            self.logger.info(f"📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
            self.logger.info(f"📉 Loss: {checkpoint.get('loss', 'unknown')}")
            self.logger.info(f"🔍 Backbone: {backbone_type}")
            self.logger.info(f"📋 Layers: {checkpoint_layers}")
            
            return model
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            self.logger.error(f"📋 Detail: {str(e)}")
            
            # Fallback ke model baru
            self.logger.warning("⚠️ Kembali ke model baru dengan konfigurasi default")
            return self.get_model()