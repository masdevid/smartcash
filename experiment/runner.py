# File: src/experiment/runner.py
# Author: Alfrida Sabar
# Deskripsi: Modul eksperimen untuk SmartCash Detector dengan integrasi komprehensif

import torch
from pathlib import Path
import yaml
from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.detector import SmartCashDetector
from models.efficientnet_backbone import EfficientNetBackbone
from data.dataloader import create_dataloader
from metrics.test_scenarios import TestScenarioRunner, ScenarioConfig
from utils.logging import ColoredLogger

@dataclass
class ExperimentConfig:
    name: str
    model_config: Dict
    data_config: Dict
    scenarios: List[Dict]
    metrics: List[str]

class ExperimentRunner:
    def __init__(self, config_path: str = 'config/experiments.yaml'):
        self.logger = ColoredLogger('Experiment')
        self.config_path = Path(config_path)
        self.results_dir = Path('experiment_results')
        self.results_dir.mkdir(exist_ok=True)
        self._load_config()
        
    def _load_config(self):
        """Load konfigurasi eksperimen"""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def run_experiment(self, experiment_name: str):
        """Jalankan eksperimen spesifik"""
        if experiment_name not in self.config['experiments']:
            raise ValueError(f"❌ Eksperimen {experiment_name} tidak ditemukan!")
            
        exp_config = ExperimentConfig(**self.config['experiments'][experiment_name])
        self.logger.info(f"🚀 Memulai eksperimen: {exp_config.name}")
        
        # Setup models
        baseline = self._setup_baseline_model(exp_config.model_config)
        proposed = self._setup_proposed_model(exp_config.model_config)
        
        # Setup data
        test_loader = create_dataloader(
            Path(exp_config.data_config['test_path']),
            batch_size=exp_config.data_config.get('batch_size', 16),
            augment=False
        )
        
        # Setup scenario runner
        runner = TestScenarioRunner(proposed, test_loader)
        
        # Run scenarios
        results = {}
        for scenario in tqdm(exp_config.scenarios, desc="Menjalankan skenario"):
            scenario_config = ScenarioConfig(**scenario)
            
            # Run baseline
            baseline_results = runner.run_scenario(
                scenario_config, model=baseline
            )
            
            # Run proposed
            proposed_results = runner.run_scenario(
                scenario_config, model=proposed
            )
            
            # Compare results
            comparison = runner.compare_models(
                baseline_results, 
                proposed_results,
                exp_config.metrics
            )
            
            results[scenario_config.name] = {
                'baseline': baseline_results,
                'proposed': proposed_results,
                'comparison': comparison
            }
            
            self._save_results(results, experiment_name)
        
        return results
    
    def _setup_baseline_model(self, config: Dict):
        """Setup model baseline YOLOv5"""
        self.logger.info("📦 Menyiapkan model baseline...")
        return SmartCashDetector(
            backbone='csp_darknet',
            **config
        )
        
    def _setup_proposed_model(self, config: Dict):
        """Setup model dengan EfficientNet backbone"""
        self.logger.info("📦 Menyiapkan model yang diusulkan...")
        backbone = EfficientNetBackbone(phi=config.get('backbone_phi', 4))
        return SmartCashDetector(
            backbone=backbone,
            **config
        )
        
    def _save_results(self, results: Dict, experiment_name: str):
        """Simpan hasil eksperimen"""
        save_path = self.results_dir / f"{experiment_name}_results.yml"
        with open(save_path, 'w') as f:
            yaml.dump(results, f)
        self.logger.info(f"💾 Hasil tersimpan di: {save_path}")
        
    def analyze_results(self, experiment_name: str):
        """Analisis hasil eksperimen"""
        results_path = self.results_dir / f"{experiment_name}_results.yml"
        if not results_path.exists():
            raise FileNotFoundError("❌ Hasil eksperimen tidak ditemukan!")
            
        with open(results_path) as f:
            results = yaml.safe_load(f)
            
        analysis = {}
        for scenario, data in results.items():
            improvements = {
                metric: ((data['proposed'][metric] - data['baseline'][metric]) 
                        / data['baseline'][metric] * 100)
                for metric in data['baseline'].keys()
                if isinstance(data['baseline'][metric], (int, float))
            }
            
            analysis[scenario] = {
                'improvements': improvements,
                'statistical_significance': data['comparison'].get('significance')
            }
            
        return analysis

if __name__ == '__main__':
    runner = ExperimentRunner()
    results = runner.run_experiment('baseline_vs_proposed')
    analysis = runner.analyze_results('baseline_vs_proposed')
    print("✨ Eksperimen selesai!")