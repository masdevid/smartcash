# File: smartcash/handlers/model/model_experiments.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk menjalankan eksperimen model dengan konfigurasi berbeda

import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.model.model_factory import ModelFactory
from smartcash.handlers.model.model_trainer import ModelTrainer
from smartcash.handlers.model.model_evaluator import ModelEvaluator

class ModelExperiments:
    """
    Handler untuk menjalankan dan membandingkan berbagai eksperimen model
    dengan konfigurasi yang berbeda.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi model experiments.
        
        Args:
            config: Konfigurasi dasar
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup factories dan handlers
        self.model_factory = ModelFactory(config, logger)
        self.trainer = ModelTrainer(config, logger)
        self.evaluator = ModelEvaluator(config, logger)
        
        # Hasil eksperimen
        self.results = {}
        
        # Output direktori
        self.output_dir = Path(config.get('output_dir', 'runs/experiments'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🧪 ModelExperiments diinisialisasi")
    
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
            experiment_config = original_config.copy()
            if 'backbone' in scenario:
                experiment_config['model'] = experiment_config.get('model', {})
                experiment_config['model']['backbone'] = scenario['backbone']
                
            if 'learning_rate' in scenario:
                experiment_config['training'] = experiment_config.get('training', {})
                experiment_config['training']['learning_rate'] = scenario['learning_rate']
                
            if 'batch_size' in scenario:
                experiment_config['training'] = experiment_config.get('training', {})
                experiment_config['training']['batch_size'] = scenario['batch_size']
            
            # Buat trainer dan evaluator khusus untuk eksperimen ini
            experiment_trainer = ModelTrainer(experiment_config, self.logger)
            experiment_evaluator = ModelEvaluator(experiment_config, self.logger)
                
            # Buat model sesuai skenario
            model = self.model_factory.create_model(
                backbone_type=scenario.get('backbone')
            )
            
            # Training model dengan skenario
            training_results = experiment_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model
            )
            
            # Evaluasi model pada test set
            eval_metrics = experiment_evaluator.evaluate(
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
    
    def run_multiple_experiments(
        self,
        scenarios: List[Dict],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Dict]:
        """
        Jalankan beberapa eksperimen dan bandingkan hasilnya.
        
        Args:
            scenarios: List skenario eksperimen
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            
        Returns:
            Dict berisi hasil semua eksperimen
        """
        all_results = {}
        
        for i, scenario in enumerate(scenarios):
            self.logger.info(f"🧪 Eksperimen {i+1}/{len(scenarios)}: {scenario['name']}")
            
            try:
                result = self.run_experiment(
                    scenario=scenario,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader
                )
                
                all_results[scenario['name']] = result
                
            except Exception as e:
                self.logger.error(f"❌ Eksperimen {scenario['name']} gagal: {str(e)}")
                all_results[scenario['name']] = {'error': str(e)}
        
        # Buat perbandingan
        self._compare_experiments(all_results)
        
        return all_results
    
    def _compare_experiments(self, results: Dict[str, Dict]) -> None:
        """
        Bandingkan hasil beberapa eksperimen dan tampilkan dalam bentuk tabel.
        
        Args:
            results: Dict berisi hasil eksperimen
        """
        try:
            # Ekstrak metrik untuk perbandingan
            comparison_data = []
            
            for name, result in results.items():
                if 'error' in result:
                    # Skip eksperimen yang gagal
                    continue
                
                metrics = result.get('metrics', {})
                
                experiment_data = {
                    'Eksperimen': name,
                    'Backbone': result.get('scenario', {}).get('backbone', 'N/A'),
                    'Akurasi': metrics.get('accuracy', 0) * 100,
                    'Presisi': metrics.get('precision', 0) * 100,
                    'Recall': metrics.get('recall', 0) * 100,
                    'F1 Score': metrics.get('f1', 0) * 100,
                    'mAP': metrics.get('mAP', 0) * 100,
                    'Inference Time (ms)': metrics.get('inference_time', 0) * 1000,
                    'Best Val Loss': result.get('best_val_loss', 0)
                }
                
                comparison_data.append(experiment_data)
            
            # Buat dataframe
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                
                # Simpan ke CSV
                csv_path = self.output_dir / "experiments_comparison.csv"
                df.to_csv(csv_path, index=False)
                
                # Log tabel perbandingan
                self.logger.info(f"📊 Perbandingan Eksperimen:")
                
                # Format tabel untuk logger
                table_str = "\n"
                headers = ["Eksperimen", "Backbone", "Akurasi", "F1", "Inf. Time"]
                
                # Header
                header_row = "| " + " | ".join(headers) + " |"
                separator = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
                table_str += header_row + "\n" + separator + "\n"
                
                # Rows
                for data in comparison_data:
                    row = f"| {data['Eksperimen']} | {data['Backbone']} | "
                    row += f"{data['Akurasi']:.2f}% | {data['F1 Score']:.2f}% | "
                    row += f"{data['Inference Time (ms)']:.2f}ms |"
                    table_str += row + "\n"
                
                self.logger.info(table_str)
                self.logger.info(f"💾 Hasil lengkap tersimpan di {csv_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Gagal membandingkan eksperimen: {str(e)}")
    
    def compare_backbones(
        self,
        backbones: List[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Dict]:
        """
        Bandingkan performa berbagai backbone.
        
        Args:
            backbones: List backbone yang akan dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            
        Returns:
            Dict berisi hasil eksperimen untuk setiap backbone
        """
        # Buat skenario untuk setiap backbone
        scenarios = [
            {
                'name': f"Backbone-{backbone}",
                'description': f"Evaluasi model dengan backbone {backbone}",
                'backbone': backbone
            }
            for backbone in backbones
        ]
        
        # Jalankan eksperimen
        return self.run_multiple_experiments(
            scenarios=scenarios,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
    
    def hyperparameter_tuning(
        self,
        param_grid: Dict[str, List],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        max_experiments: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Lakukan grid search untuk hyperparameter tuning.
        
        Args:
            param_grid: Dict parameter dan nilai yang akan dievaluasi
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            max_experiments: Maksimum jumlah eksperimen (opsional)
            
        Returns:
            Dict berisi hasil eksperimen untuk setiap kombinasi parameter
        """
        from itertools import product
        
        # Generate kombinasi parameter
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        # Batasi jumlah eksperimen jika diperlukan
        if max_experiments is not None and len(combinations) > max_experiments:
            import random
            random.shuffle(combinations)
            combinations = combinations[:max_experiments]
            
        # Buat skenario untuk setiap kombinasi parameter
        scenarios = []
        for combo in combinations:
            param_combo = {name: value for name, value in zip(param_names, combo)}
            
            # Generate nama eksperimen
            scenario_name = "_".join([f"{k}-{v}" for k, v in param_combo.items()])
            
            # Buat skenario
            scenario = {
                'name': f"HPTuning-{scenario_name}",
                'description': f"Tuning dengan parameter: {param_combo}",
                **param_combo  # Tambahkan parameter sebagai atribut skenario
            }
            
            scenarios.append(scenario)
        
        # Jalankan semua skenario
        self.logger.info(f"🧪 Menjalankan {len(scenarios)} eksperimen hyperparameter tuning")
        
        return self.run_multiple_experiments(
            scenarios=scenarios,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )