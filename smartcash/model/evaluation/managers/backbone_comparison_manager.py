"""
Backbone comparison manager for multi-backbone evaluation scenarios.
Handles comparison logic between different model backbones.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger
from smartcash.model.evaluation.checkpoint_selector import create_checkpoint_selector
from smartcash.model.evaluation.evaluation_service import create_evaluation_service
from smartcash.model.evaluation.processors.scenario_data_source_selector import create_scenario_data_source_selector


class BackboneComparisonManager:
    """Manages backbone comparison evaluations with graceful error handling"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger('backbone_comparison')
        self.config = config or {}
        
        # Initialize components
        self.checkpoint_selector = create_checkpoint_selector(config)
        self.scenario_selector = create_scenario_data_source_selector(config)
        
        # Default backbones for comparison
        self.default_backbones = ['cspdarknet', 'efficientnet_b4']
        self.default_scenarios = ['position_variation', 'lighting_variation']
    
    def run_backbone_comparison(self, backbones: List[str] = None, scenarios: List[str] = None, 
                               verbose: bool = False) -> Dict[str, Any]:
        """Run comprehensive backbone comparison evaluation"""
        
        # Use defaults if not specified
        if backbones is None:
            backbones = self.default_backbones.copy()
            self.logger.info(f"🚀 Running backbone comparison: {', '.join(backbones)}")
        else:
            self.logger.info(f"🚀 Running evaluation for specified backbones: {', '.join(backbones)}")
        
        if scenarios is None:
            scenarios = self.default_scenarios.copy()
        
        all_results = {}
        
        for backbone in backbones:
            self.logger.info(f"📋 Evaluating backbone: {backbone}")
            
            try:
                backbone_result = self._evaluate_single_backbone(backbone, scenarios, verbose)
                all_results[backbone] = backbone_result
                
            except Exception as e:
                self.logger.error(f"❌ Error evaluating {backbone}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                
                # Create empty results for failed backbone
                backbone_result = self._create_empty_backbone_results(backbone, scenarios, error=str(e))
                all_results[backbone] = backbone_result
        
        # Generate summary
        summary = self._generate_comparison_summary(all_results, scenarios)
        
        return {
            'status': 'success',
            'evaluation_type': 'backbone_comparison' if len(backbones) > 1 else 'single_backbone',
            'backbones_evaluated': backbones,
            'scenarios_evaluated': scenarios,
            'results': all_results,
            'summary': summary
        }
    
    def _evaluate_single_backbone(self, backbone: str, scenarios: List[str], verbose: bool) -> Dict[str, Any]:
        """Evaluate a single backbone across all scenarios"""
        
        # Find best checkpoint for backbone
        try:
            checkpoint_path = self._find_best_checkpoint_for_backbone(backbone)
            if not checkpoint_path:
                self.logger.warning(f"⚠️ No checkpoints found for backbone: {backbone}")
                return self._create_empty_backbone_results(backbone, scenarios)
            
            self.logger.info(f"📦 Using checkpoint: {Path(checkpoint_path).name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error finding checkpoint for {backbone}: {e}")
            return self._create_empty_backbone_results(backbone, scenarios, error=str(e))
        
        # Load model using training infrastructure (once for all scenarios)
        try:
            model_api = self._load_model_with_training_infrastructure(checkpoint_path)
            if not model_api:
                self.logger.warning(f"⚠️ Failed to load model for {backbone}")
                return self._create_empty_backbone_results(backbone, scenarios)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading model for {backbone}: {e}")
            return self._create_empty_backbone_results(backbone, scenarios, error=str(e))
        
        # Create evaluation service once per backbone (model reuse optimization)
        service = create_evaluation_service(model_api=model_api, config=self.config)
        
        # Run multiple scenarios with SAME model instance (no re-instantiation/warmup)
        self.logger.info(f"🚀 Running {len(scenarios)} scenarios with {backbone} (model reuse enabled)")
        backbone_results = {}
        
        for i, scenario in enumerate(scenarios, 1):
            self.logger.info(f"🎯 Scenario {i}/{len(scenarios)}: {scenario} with {backbone} (reusing loaded model)...")
            
            try:
                # Use optimized scenario evaluation that reuses model and warmup
                result = self._run_scenario_optimized(service, scenario, checkpoint_path, backbone)
                backbone_results[scenario] = result
                
                if result['status'] == 'success':
                    self._log_scenario_results(scenario, result['metrics'], brief=True)
                else:
                    self.logger.warning(f"⚠️ {scenario} failed: {result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error running {scenario} with {backbone}: {e}")
                backbone_results[scenario] = {
                    'status': 'error',
                    'error': str(e),
                    'metrics': self._get_empty_metrics()
                }
        
        # Add backbone metadata
        backbone_results['backbone'] = backbone
        backbone_results['checkpoint'] = checkpoint_path
        backbone_results['status'] = 'success'
        
        return backbone_results
    
    def _run_scenario_optimized(self, service, scenario_name: str, checkpoint_path: str, backbone: str) -> Dict[str, Any]:
        """🚀 Run scenario with model reuse optimization (no re-instantiation/warmup)"""
        
        # Get checkpoint info without reloading model (optimization)
        checkpoint_info = self.checkpoint_selector.get_checkpoint_info(checkpoint_path)
        if not checkpoint_info:
            return {'status': 'error', 'error': 'Failed to get checkpoint info'}
        
        # Mark model as already loaded to skip reloading
        checkpoint_info['model_loaded'] = True
        checkpoint_info['backbone'] = backbone
        checkpoint_info['optimized_run'] = True  # Flag for optimization
        
        # Use the evaluation service's evaluator directly (bypasses checkpoint loading)
        result = service._evaluate_scenario_with_evaluator(scenario_name, checkpoint_info, 0, 1)
        
        # Generate charts for this scenario result
        chart_files = service._generate_single_scenario_charts(scenario_name, checkpoint_info, result)
        
        return {
            'status': 'success',
            'scenario_name': scenario_name,
            'checkpoint_info': checkpoint_info,
            'metrics': result['metrics'],
            'additional_data': result.get('additional_data', {}),
            'chart_files': chart_files,
            'optimization_used': 'model_reuse'  # Track that optimization was used
        }
    
    def _find_best_checkpoint_for_backbone(self, backbone: str) -> Optional[str]:
        """Find the best checkpoint for a given backbone with priority logic"""
        
        # Get all available checkpoints
        available_checkpoints = self.checkpoint_selector.list_available_checkpoints()
        
        if not available_checkpoints:
            return None
        
        # Filter by backbone
        backbone_checkpoints = [cp for cp in available_checkpoints 
                               if cp['backbone'].lower() == backbone.lower()]
        
        if not backbone_checkpoints:
            return None
        
        # Prioritize checkpoints with _phase2 suffix and latest date
        def checkpoint_priority(cp):
            filename = cp['filename']
            
            # Priority scores (higher is better)
            phase2_score = 100 if '_phase2' in filename else 0
            
            # Extract date from filename for sorting (format: YYYYMMDD)
            import re
            date_match = re.search(r'_(\\d{8})', filename)
            date_score = int(date_match.group(1)) if date_match else 0
            
            # Use mAP as tiebreaker
            map_score = cp.get('metrics', {}).get('val_map', 0) * 10
            
            return phase2_score + date_score + map_score
        
        # Sort by priority (highest first)
        backbone_checkpoints.sort(key=checkpoint_priority, reverse=True)
        
        best_checkpoint = backbone_checkpoints[0]
        self.logger.info(f"📋 Selected: {best_checkpoint['filename']} (phase2: {'_phase2' in best_checkpoint['filename']})")
        
        return best_checkpoint['path']
    
    def _load_model_with_training_infrastructure(self, checkpoint_path: str) -> Any:
        """Load model using existing model API infrastructure"""
        
        self.logger.info(f"📦 Loading model: {Path(checkpoint_path).name}")
        
        try:
            # Import here to avoid circular imports
            from smartcash.model.api.core import create_api
            import torch
            
            # Load checkpoint to extract model configuration
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Use configuration from checkpoint if available
            model_config = checkpoint_data.get('config', {})
            if not model_config:
                # Fallback config based on checkpoint analysis
                state_dict = checkpoint_data.get('model_state_dict', {})
                detection_head_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
                
                if detection_head_keys:
                    output_channels = state_dict[detection_head_keys[0]].shape[0]
                    num_classes = (output_channels // 3) - 5  # YOLOv5 formula: (classes + 5) * 3 anchors
                    
                    model_config = {
                        'backbone': 'cspdarknet',
                        'num_classes': num_classes,
                        'layer_mode': 'multi',
                        'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                        'pretrained': False,
                        'img_size': 640
                    }
                    self.logger.info(f"🔍 Inferred model config: {num_classes} classes, {output_channels} outputs")
            
            # Create model API with checkpoint configuration
            config = {'model': model_config}
            api = create_api(config=config, use_yolov5_integration=True)
            
            # Build model with correct configuration first
            build_result = api.build_model(model_config=model_config)
            if not build_result.get('success', False):
                self.logger.error(f"❌ Failed to build model: {build_result.get('error')}")
                return None
            
            # Load checkpoint
            result = api.load_checkpoint(checkpoint_path)
            
            if result.get('success', False):
                self.logger.info(f"✅ Model loaded: {result.get('message', 'Success')}")
                return api
            else:
                self.logger.error(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return None
    
    def _create_empty_backbone_results(self, backbone: str, scenarios: List[str], error: str = None) -> Dict[str, Any]:
        """Create empty results structure for missing backbone"""
        
        empty_metrics = self._get_empty_metrics()
        
        backbone_results = {
            'backbone': backbone,
            'checkpoint': None,
            'status': 'missing' if not error else 'error',
        }
        
        if error:
            backbone_results['error'] = error
        
        # Create empty results for all scenarios
        for scenario in scenarios:
            backbone_results[scenario] = {
                'status': 'missing' if not error else 'error',
                'metrics': empty_metrics.copy(),
                'error': f'No checkpoint found for {backbone}' if not error else error
            }
        
        return backbone_results
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Get empty metrics structure"""
        return {
            'map50': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'inference_time_avg': 0.0
        }
    
    def _log_scenario_results(self, scenario: str, metrics: Dict[str, Any], brief: bool = False) -> None:
        """Log evaluation results"""
        if brief:
            map_score = metrics.get('map50', 0)
            accuracy = metrics.get('accuracy', 0)
            self.logger.info(f"   Results: mAP={map_score:.3f}, Accuracy={accuracy:.3f}")
        else:
            self.logger.info(f"📊 Results for {scenario}:")
            self.logger.info(f"   mAP@50: {metrics.get('map50', 0):.3f}")
            self.logger.info(f"   Precision: {metrics.get('precision', 0):.3f}")
            self.logger.info(f"   Recall: {metrics.get('recall', 0):.3f}")
            self.logger.info(f"   F1: {metrics.get('f1', 0):.3f}")
            self.logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
            if 'inference_time_avg' in metrics:
                self.logger.info(f"   Inference: {metrics['inference_time_avg']:.3f}s")
    
    def _generate_comparison_summary(self, all_results: Dict[str, Any], scenarios: List[str]) -> Dict[str, Any]:
        """Generate comprehensive comparison summary"""
        
        summary = {
            'total_backbones': len(all_results),
            'total_scenarios': len(scenarios),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'best_performers': {},
            'backbone_status': {}
        }
        
        # Count successful vs failed evaluations
        for backbone, result in all_results.items():
            if result.get('status') == 'success':
                summary['successful_evaluations'] += 1
                summary['backbone_status'][backbone] = 'success'
            else:
                summary['failed_evaluations'] += 1
                summary['backbone_status'][backbone] = 'failed'
        
        # Find best performers for each metric
        metrics_to_compare = ['map50', 'accuracy', 'precision', 'recall', 'f1', 'inference_time_avg']
        
        for metric in metrics_to_compare:
            best_backbone = None
            best_value = -1 if metric != 'inference_time_avg' else float('inf')
            
            for backbone, backbone_result in all_results.items():
                if backbone_result.get('status') != 'success':
                    continue
                
                total_value = 0
                valid_scenarios = 0
                
                for scenario in scenarios:
                    scenario_result = backbone_result.get(scenario, {})
                    if scenario_result.get('status') == 'success':
                        metrics = scenario_result.get('metrics', {})
                        value = metrics.get(metric, 0.0)
                        total_value += value
                        valid_scenarios += 1
                
                if valid_scenarios > 0:
                    avg_value = total_value / valid_scenarios
                    if metric == 'inference_time_avg':
                        if avg_value < best_value:
                            best_value = avg_value
                            best_backbone = backbone
                    else:
                        if avg_value > best_value:
                            best_value = avg_value
                            best_backbone = backbone
            
            if best_backbone:
                summary['best_performers'][metric] = {
                    'backbone': best_backbone,
                    'value': best_value
                }
        
        return summary
    
    def print_comparison_summary(self, comparison_result: Dict[str, Any]) -> None:
        """Print comprehensive comparison summary to console"""
        
        all_results = comparison_result.get('results', {})
        scenarios = comparison_result.get('scenarios_evaluated', [])
        summary = comparison_result.get('summary', {})
        
        self.logger.info(f"\\n📊 BACKBONE COMPARISON SUMMARY")
        self.logger.info("=" * 60)
        
        # Metrics to compare
        metrics_names = ['mAP', 'Accuracy', 'Precision', 'Recall', 'F1', 'Inference Time']
        metrics_keys = ['map50', 'accuracy', 'precision', 'recall', 'f1', 'inference_time_avg']
        
        for scenario in scenarios:
            self.logger.info(f"\\n🎯 {scenario.replace('_', ' ').title()}:")
            self.logger.info("-" * 40)
            
            # Header
            backbone_names = list(all_results.keys())
            header = f"{'Metric':<15}"
            for backbone in backbone_names:
                header += f"{backbone:<15}"
            self.logger.info(header)
            self.logger.info("-" * len(header))
            
            # Metrics rows
            for metric_name, metric_key in zip(metrics_names, metrics_keys):
                row = f"{metric_name:<15}"
                
                for backbone in backbone_names:
                    scenario_result = all_results[backbone].get(scenario, {})
                    if scenario_result.get('status') == 'success':
                        metrics = scenario_result.get('metrics', {})
                        value = metrics.get(metric_key, 0.0)
                        if metric_key == 'inference_time_avg':
                            row += f"{value:.3f}s".ljust(15)
                        else:
                            row += f"{value:.3f}".ljust(15)
                    else:
                        row += "N/A".ljust(15)
                
                self.logger.info(row)
        
        # Best performers
        best_performers = summary.get('best_performers', {})
        if best_performers:
            self.logger.info(f"\\n🏆 Best Performers:")
            self.logger.info("-" * 30)
            
            for metric, info in best_performers.items():
                backbone = info['backbone']
                value = info['value']
                if metric == 'inference_time_avg':
                    self.logger.info(f"   {metric.replace('_', ' ').title()}: {backbone} ({value:.3f}s)")
                else:
                    self.logger.info(f"   {metric.replace('_', ' ').title()}: {backbone} ({value:.3f})")


# Factory functions
def create_backbone_comparison_manager(config: Dict[str, Any] = None) -> BackboneComparisonManager:
    """Factory function to create backbone comparison manager"""
    return BackboneComparisonManager(config)


def run_backbone_comparison_evaluation(backbones: List[str] = None, scenarios: List[str] = None, 
                                     config: Dict[str, Any] = None, verbose: bool = False) -> Dict[str, Any]:
    """One-liner function to run backbone comparison evaluation"""
    manager = create_backbone_comparison_manager(config)
    return manager.run_backbone_comparison(backbones, scenarios, verbose)