"""Research scenario evaluation handler."""

import os
import time
import pandas as pd
from typing import Dict, Optional
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.base_evaluation_handler import BaseEvaluationHandler

class ResearchScenarioHandler(BaseEvaluationHandler):
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
        
        # Load model and create data loader
        model = self._load_model(model_path)
        test_loader = self.data_handler.get_test_loader(
            dataset_path=test_data_path,
            batch_size=32
        )
        
        # Initialize metrics
        all_predictions = []
        all_targets = []
        inference_times = []
        
        # Evaluation loop
        model.eval()
        eval_pbar = tqdm(test_loader, desc=f"Memproses {scenario_name}")
        
        with torch.no_grad():
            for images, targets in eval_pbar:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                predictions = torch.sigmoid(predictions)
                inference_time = (time.time() - start_time) / images.size(0)  # per image
                
                # Store predictions, targets and inference time
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                inference_times.append(inference_time)
                
                # Update progress bar
                eval_pbar.set_postfix({
                    'batch_size': images.size(0),
                    'inf_time': f'{inference_time*1000:.1f}ms'
                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            np.concatenate(all_predictions),
            np.concatenate(all_targets)
        )
        metrics['Waktu Inferensi'] = np.mean(inference_times)
        
        # Add to results dataframe
        self.results_df.loc[len(self.results_df)] = [
            scenario_name,
            metrics['Akurasi'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-Score'],
            metrics['mAP'],
            metrics['Waktu Inferensi']
        ]
        
        # Log results
        self._log_scenario_metrics(scenario_name, metrics)
        
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
            
            scenario_results = []
            for run in range(3):
                self.logger.info(f"\nPercobaan ke-{run+1}")
                results = self.evaluate_scenario(
                    scenario_name=f"{scenario_name} (Run {run+1})",
                    model_path=os.path.join(self.config['checkpoints_dir'], scenario_config['model']),
                    test_data_path=os.path.join(self.config['data_dir'], scenario_config['data'])
                )
                scenario_results.append(results)
            
            # Calculate average results
            avg_results = {
                metric: np.mean([run[metric] for run in scenario_results])
                for metric in scenario_results[0].keys()
                if metric != 'confusion_matrix'
            }
            
            # Add average results to dataframe
            self.results_df.loc[len(self.results_df)] = [
                f"{scenario_name} (Rata-rata)",
                avg_results['Akurasi'],
                avg_results['Precision'],
                avg_results['Recall'],
                avg_results['F1-Score'],
                avg_results['mAP'],
                avg_results['Waktu Inferensi']
            ]
        
        # Save results to CSV
        results_path = os.path.join(self.config['output_dir'], 'research_results.csv')
        self.results_df.to_csv(results_path, index=False)
        self.logger.info(f"\n💾 Hasil penelitian disimpan ke: {results_path}")
        
        return self.results_df
    
    def _log_scenario_metrics(self, scenario_name: str, metrics: Dict) -> None:
        """Log scenario evaluation metrics."""
        self.logger.info(f"\nHasil untuk {scenario_name}:")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                if value is not None:
                    self.logger.info(f"\nConfusion Matrix:\n{value}")
            elif metric == 'Waktu Inferensi':
                self.logger.info(f"{metric}: {value*1000:.1f}ms")
            else:
                self.logger.info(f"{metric}: {value:.4f}")
