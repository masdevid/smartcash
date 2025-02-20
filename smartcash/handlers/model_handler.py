# File: handlers/model_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk model training dan evaluasi

import os
from typing import Dict, Optional, List
import yaml
import time
from smartcash.utils.logger import SmartCashLogger
from smartcash.models.baseline import BaselineModel

class ModelHandler:
    """Handler untuk training dan evaluasi model"""
    
    def __init__(
        self,
        config_path: str,
        num_classes: int,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.num_classes = num_classes
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
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
            if scenario['backbone'] == 'csp':
                model = BaselineModel(
                    config_path=self.config,
                    num_classes=self.num_classes,
                    logger=self.logger
                )
            else:
                # TODO: Implementasi untuk EfficientNet backbone
                raise NotImplementedError(
                    "EfficientNet backbone belum diimplementasi"
                )
            
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
            raise e
    
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