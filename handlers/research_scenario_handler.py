"""Research scenario evaluation handler."""

import os
import time
import pandas as pd
from typing import Dict, Optional
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluator import Evaluator

class ResearchScenarioHandler(Evaluator):
    """Handler for research scenario evaluation."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__(config=config, logger=logger)
        self.results_df = pd.DataFrame(
            columns=['Skenario', 'Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Waktu Inferensi']
        )
    
    def evaluate_scenario(self, scenario_name: str, model_path: str, test_data_path: str) -> Dict:
        """
        Evaluate a specific research scenario.
        
        Args:
            scenario_name: Name of the scenario being tested
            model_path: Path to the model checkpoint
            test_data_path: Path to the test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"\n📊 Mengevaluasi {scenario_name}")
        
        # Run evaluation using base class method
        metrics = self.evaluate_model(
            model_path=model_path,
            dataset_path=test_data_path
        )
        
        # Add to results dataframe
        self.results_df.loc[len(self.results_df)] = [
            scenario_name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['mAP'],
            metrics['inference_time']
        ]
        
        return metrics
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all research scenarios and return results.
        
        Returns:
            DataFrame containing results for all scenarios
        """
        scenarios = {
            'Skenario-1': {
                'desc': 'YOLOv5 Default (CSPDarknet) - Posisi Bervariasi',
                'model': 'cspdarknet_position_varied.pth',
                'data': 'test_position_varied'
            },
            'Skenario-2': {
                'desc': 'YOLOv5 Default (CSPDarknet) - Pencahayaan Bervariasi',
                'model': 'cspdarknet_lighting_varied.pth',
                'data': 'test_lighting_varied'
            },
            'Skenario-3': {
                'desc': 'YOLOv5 EfficientNet-B4 - Posisi Bervariasi',
                'model': 'efficientnet_position_varied.pth',
                'data': 'test_position_varied'
            },
            'Skenario-4': {
                'desc': 'YOLOv5 EfficientNet-B4 - Pencahayaan Bervariasi',
                'model': 'efficientnet_lighting_varied.pth',
                'data': 'test_lighting_varied'
            }
        }
        
        # Run each scenario 3 times
        for scenario_name, scenario_config in scenarios.items():
            self.logger.info(f"\n🔬 Menjalankan {scenario_name}: {scenario_config['desc']}")
            
            # Use evaluate_multiple_runs from base class
            avg_metrics = self.evaluate_multiple_runs(
                model_path=os.path.join(self.config['checkpoints_dir'], scenario_config['model']),
                dataset_path=os.path.join(self.config['data_dir'], scenario_config['data']),
                num_runs=3
            )
            
            # Add average results to dataframe
            self.results_df.loc[len(self.results_df)] = [
                f"{scenario_name} (Rata-rata)",
                avg_metrics['accuracy'],
                avg_metrics['precision'],
                avg_metrics['recall'],
                avg_metrics['f1'],
                avg_metrics['mAP'],
                avg_metrics['inference_time']
            ]
        
        # Save results to CSV
        results_path = os.path.join(self.config['output_dir'], 'research_results.csv')
        self.results_df.to_csv(results_path, index=False)
        self.logger.info(f"\n💾 Hasil penelitian disimpan ke: {results_path}")
        
        return self.results_df