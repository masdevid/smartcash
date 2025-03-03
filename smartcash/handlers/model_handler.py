# File: handlers/model_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk model training dan evaluasi

import os
from typing import Dict, Optional, List, Union
from roboflow.core.version import process
import yaml
import time
import torch
from pathlib import Path
from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel

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
            # Parameter tambahan
            pretrained = self.config.get('model', {}).get('pretrained', True)
            layers = self.config.get('layers', ['banknote'])
            num_classes = len(self.config.get('dataset', {}).get('classes', [7]))
            
            # Log detail inisialisasi
            self.logger.info(
                f"🚀 Mempersiapkan model dengan:\n"
                f"   • Backbone: {backbone_type}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Jumlah Layer: {len(layers)}\n"
                f"   • Jumlah Kelas: {num_classes}"
            )
            
            # Inisialisasi model dengan backbone yang dipilih
            if backbone_type == 'efficientnet':
                model = YOLOv5Model(
                    num_classes=num_classes,
                    backbone_type='efficientnet',
                    pretrained=pretrained,
                    layers=layers,  # Menggunakan layers, bukan detection_layers
                    logger=self.logger
                )
            elif backbone_type == 'cspdarknet':
                model = YOLOv5Model(
                    num_classes=num_classes,
                    backbone_type='cspdarknet',
                    pretrained=pretrained,
                    layers=layers,  # Menggunakan layers, bukan detection_layers
                    logger=self.logger
                )
            else:
                # Fallback untuk backbone khusus/eksperimental
                model = BaselineModel(
                    num_classes=num_classes,
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
            raise
            
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