# File: handlers/evaluation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk proses evaluasi dan perbandingan model deteksi nilai mata uang

import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm
from datetime import datetime

from utils.logger import SmartCashLogger
from utils.metrics import MetricsCalculator
from utils.visualization import ResultVisualizer
from models.baseline import BaselineModel
from models.yolov5_efficient import YOLOv5Efficient
from .evaluation_data_loader import create_evaluation_dataloader

class EvaluationHandler:
    """Handler untuk evaluasi dan perbandingan model deteksi nilai mata uang"""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = "results",
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.metrics_calc = MetricsCalculator()
        self.visualizer = ResultVisualizer(output_dir)
        
        # Buat direktori output
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> None:
        """Load konfigurasi eksperimen"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _prepare_scenario(
        self,
        scenario_config: Dict,
        dataset_path: str
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        """Siapkan model dan dataset untuk skenario tertentu"""
        self.logger.info(f"🔄 Mempersiapkan skenario: {scenario_config['name']}")
        
        # Setup model sesuai backbone
        if scenario_config['backbone'] == 'csp':
            model = BaselineModel(
                config_path=self.config_path,  # tambahkan config_path ke constructor
                num_classes=len(self.config['dataset']['classes']),
                logger=self.logger
            )
        else:
            model = YOLOv5Efficient(
                num_classes=len(self.config['dataset']['classes']),
                logger=self.logger
            )
            
        model.build()
        
        # Setup dataset dengan kondisi sesuai skenario
        dataloader = create_evaluation_dataloader(
            data_path=dataset_path,
            batch_size=self.config['model']['batch_size'],
            num_workers=self.config['model']['workers'],
            conditions=scenario_config['conditions']
        )
        
        return model, dataloader
        
    def evaluate_scenario(
        self,
        scenario_config: Dict,
        dataset_path: str
    ) -> Dict:
        """Evaluasi model untuk skenario tertentu"""
        model, dataloader = self._prepare_scenario(scenario_config, dataset_path)
        
        # Setup progress bar
        n_batches = len(dataloader)
        progress_bar = tqdm(
            total=n_batches,
            desc=f"Evaluasi {scenario_config['name']}",
            unit='batch'
        )
        
        results = {
            'scenario': scenario_config['name'],
            'metrics': {},
            'predictions': []
        }
        
        try:
            # Evaluasi per batch
            for batch_idx, (images, targets) in enumerate(dataloader):
                # Pindahkan ke device yang sesuai
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    predictions = model(images)
                
                # Update metrics
                batch_metrics = self.metrics_calc.update(
                    predictions,
                    targets
                )
                
                # Simpan untuk visualisasi
                if self.config['evaluation']['save_visualizations']:
                    self.visualizer.save_batch_predictions(
                        images,
                        predictions,
                        targets,
                        scenario_config['name'],
                        batch_idx
                    )
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{batch_metrics['loss']:.4f}",
                    mAP=f"{batch_metrics['mAP']:.4f}"
                )
                
            # Calculate final metrics
            results['metrics'] = self.metrics_calc.compute()
            
            # Log hasil
            self._log_scenario_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            raise e
        finally:
            progress_bar.close()
            
    def run_all_scenarios(self) -> Dict:
        """Jalankan semua skenario evaluasi"""
        self.logger.start("🚀 Memulai evaluasi semua skenario")
        
        all_results = {}
        
        # Setup progress tracking untuk semua skenario
        scenarios = self.config['experiment_scenarios']
        with tqdm(total=len(scenarios), desc="Progress Skenario") as pbar:
            for scenario in scenarios:
                results = self.evaluate_scenario(
                    scenario,
                    self.config['dataset']['test_path']
                )
                all_results[scenario['name']] = results
                pbar.update(1)
                
        # Generate comparison visualizations
        if self.config['evaluation']['save_visualizations']:
            self.visualizer.create_comparison_plots(all_results)
            
        # Save results
        self._save_results(all_results)
        
        return all_results
        
    def _log_scenario_results(self, results: Dict) -> None:
        """Log hasil evaluasi skenario"""
        metrics = results['metrics']
        self.logger.metric(
            f"\n📊 Hasil {results['scenario']}:\n"
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1-Score: {metrics['f1']:.4f}\n"
            f"mAP: {metrics['mAP']:.4f}\n"
            f"Inference Time: {metrics['inference_time']:.4f}ms"
        )
        
    def _save_results(self, results: Dict) -> None:
        """Simpan hasil evaluasi"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.success(f"✨ Hasil evaluasi berhasil disimpan ke {save_path}")

    @property
    def device(self) -> torch.device:
        """Get device untuk komputasi"""
        if not hasattr(self, '_device'):
            self._device = torch.device(
                'cuda' if torch.cuda.is_available() 
                else 'cpu'
            )
        return self._device