"""
Refactored evaluation service using SRP-compliant modules.
Main orchestrator for evaluation scenarios - focused and lightweight.
"""

import signal
import sys
import atexit
from typing import Dict, Any, List, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.evaluation.scenario_manager import ScenarioManager
from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
from smartcash.model.evaluation.utils.inference_timer import InferenceTimer
from smartcash.model.evaluation.utils.results_aggregator import ResultsAggregator
from smartcash.model.evaluation.visualization.evaluation_chart_generator import EvaluationChartGenerator
from smartcash.model.evaluation.evaluators.scenario_evaluator import create_scenario_evaluator
from smartcash.model.evaluation.utils.model_config_extractor import create_model_config_extractor


class MockProgressBridge:
    """Mock progress bridge for testing and fallback scenarios."""
    
    def start_evaluation(self, scenarios, checkpoints, title):
        pass
    
    def update_scenario(self, idx, name, message):
        pass
    
    def update_checkpoint(self, idx, name, message):
        pass
    
    def update_metrics(self, progress, message):
        pass
    
    def update_substep(self, message):
        pass
    
    def complete_evaluation(self, message):
        pass
    
    def evaluation_error(self, message):
        pass


class EvaluationService:
    """Main evaluation service orchestrator - SRP refactored version"""
    
    def __init__(self, model_api=None, config: Dict[str, Any] = None):
        self.logger = get_logger('evaluation_service')
        self.config = self._normalize_config(config)
        self.model_api = model_api
        self._shutdown_requested = False
        
        # Initialize core components
        self._initialize_components()
        
        # Progress tracking - initialize with fallback
        self.progress_bridge = None
        self.ui_components = {}
        self._initialize_progress_bridge()
        
        # Setup graceful shutdown
        self._setup_graceful_shutdown()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate configuration structure"""
        if not isinstance(config, dict):
            self.logger.warning(f"Config is not a dictionary (got {type(config).__name__}), converting to dict")
            config = {"evaluation": {}} 
        
        # Ensure the config has the required structure
        if "evaluation" not in config:
            self.logger.warning("Config missing 'evaluation' key, adding it")
            if isinstance(config, dict) and config:
                config = {"evaluation": config}
            else:
                config = {"evaluation": {}}
        
        return config
    
    def _initialize_components(self):
        """Initialize evaluation components"""
        try:
            self.scenario_manager = ScenarioManager(self.config)
            self.checkpoint_selector = CheckpointSelector(config=self.config)
            self.inference_timer = InferenceTimer(self.config)
            self.results_aggregator = ResultsAggregator(self.config)
            self.model_config_extractor = create_model_config_extractor()
            
            # Initialize chart generator for visualization
            chart_config = self.config.get('evaluation', {}).get('export', {})
            if chart_config.get('include_visualizations', True):
                chart_output_dir = self.config.get('evaluation', {}).get('data', {}).get('charts_dir', 'data/evaluation/charts')
                self.chart_generator = EvaluationChartGenerator(config=self.config, output_dir=chart_output_dir)
                self.logger.info("📊 Chart generator initialized for evaluation visualizations")
            else:
                self.chart_generator = None
                self.logger.info("📊 Chart generation disabled in configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation service components: {e}", exc_info=True)
            raise
    
    def _initialize_progress_bridge(self):
        """Initialize progress bridge with fallback for tests."""
        try:
            from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge
            self.progress_bridge = EvaluationProgressBridge({}, None)
        except Exception as e:
            self.logger.debug(f"Failed to initialize progress bridge: {e}")
            self.progress_bridge = MockProgressBridge()
    
    def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self._cleanup()
            sys.exit(0)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            self.logger.info("🧹 Performing cleanup...")
            
            # Cleanup model resources
            if hasattr(self, 'model_api') and self.model_api:
                if hasattr(self.model_api, 'model') and self.model_api.model:
                    # Move model to CPU to free GPU memory
                    try:
                        self.model_api.model.cpu()
                        self.logger.debug("Model moved to CPU")
                    except:
                        pass
            
            # Cleanup chart generator
            if hasattr(self, 'chart_generator') and self.chart_generator:
                try:
                    # Ensure any pending chart operations are completed
                    pass
                except:
                    pass
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug("CUDA cache cleared")
            except:
                pass
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _check_shutdown(self):
        """Check if shutdown was requested"""
        if self._shutdown_requested:
            self.logger.info("🛑 Shutdown requested, stopping evaluation...")
            raise KeyboardInterrupt("Evaluation stopped due to shutdown request")
    
    def run_evaluation(self, scenarios: List[str] = None, checkpoints: List[str] = None,
                      progress_callback=None, metrics_callback=None, 
                      ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
        """🚀 Run comprehensive evaluation pipeline"""
        
        # Setup progress tracking
        self.ui_components = ui_components or {}
        try:
            from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge
            self.progress_bridge = EvaluationProgressBridge(self.ui_components, progress_callback)
        except:
            pass  # Keep mock bridge
        
        # Default scenarios
        if scenarios is None:
            scenarios = ['position_variation', 'lighting_variation']
        
        # Get available checkpoints
        if checkpoints is None:
            available_checkpoints = self.checkpoint_selector.list_available_checkpoints()
            checkpoints = [cp['path'] for cp in available_checkpoints[:2]]  # Top 2 checkpoints
        
        self.progress_bridge.start_evaluation(scenarios, checkpoints, "Research Evaluation")
        
        try:
            evaluation_results = {}
            
            # Prepare scenarios
            self.logger.info(f"🎯 Preparing {len(scenarios)} scenarios")
            self.scenario_manager.prepare_all_scenarios()
            
            for scenario_idx, scenario_name in enumerate(scenarios):
                self._check_shutdown()  # Check for graceful shutdown
                self.progress_bridge.update_scenario(scenario_idx, scenario_name, "Preparing scenario data")
                
                scenario_results = {}
                
                for checkpoint_idx, checkpoint_path in enumerate(checkpoints):
                    self._check_shutdown()  # Check for graceful shutdown
                    self.progress_bridge.update_checkpoint(checkpoint_idx, Path(checkpoint_path).name, "Loading checkpoint")
                    
                    # Load checkpoint
                    checkpoint_info = self._load_checkpoint(checkpoint_path)
                    if not checkpoint_info:
                        continue
                    
                    # Run scenario evaluation using dedicated evaluator
                    scenario_result = self._evaluate_scenario_with_evaluator(
                        scenario_name, checkpoint_info, 
                        checkpoint_idx, len(checkpoints)
                    )
                    
                    # Store results
                    backbone = checkpoint_info.get('backbone', 'unknown')
                    scenario_results[backbone] = {
                        'checkpoint_info': checkpoint_info,
                        'metrics': scenario_result['metrics'],
                        'additional_data': scenario_result.get('additional_data', {})
                    }
                    
                    # Add to aggregator
                    self.results_aggregator.add_scenario_results(
                        scenario_name, backbone, checkpoint_info,
                        scenario_result['metrics'], scenario_result.get('additional_data', {})
                    )
                
                evaluation_results[scenario_name] = scenario_results
            
            # Generate comprehensive summary
            self.progress_bridge.update_metrics(90, "Generating comprehensive summary")
            summary = self._generate_evaluation_summary(evaluation_results)
            
            # Export results
            self.progress_bridge.update_metrics(95, "Exporting results")
            export_files = self.results_aggregator.export_results()
            
            # Generate evaluation charts
            chart_files = self._generate_charts(evaluation_results, scenarios, checkpoints, summary)
            
            self.progress_bridge.complete_evaluation("Evaluation completed successfully!")
            
            final_results = {
                'status': 'success',
                'evaluation_results': evaluation_results,
                'summary': summary,
                'export_files': export_files,
                'chart_files': chart_files,
                'scenarios_evaluated': len(scenarios),
                'checkpoints_evaluated': len(checkpoints)
            }
            
            # Call metrics callback
            if metrics_callback:
                metrics_callback(final_results)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.progress_bridge.evaluation_error(error_msg)
            self.logger.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'error': error_msg,
                'partial_results': getattr(self, 'evaluation_results', {})
            }
    
    def run_scenario(self, scenario_name: str, checkpoint_path: str) -> Dict[str, Any]:
        """🎯 Run single scenario evaluation"""
        
        self.logger.info(f"🎯 Running {scenario_name} evaluation")
        
        # Load checkpoint
        checkpoint_info = self._load_checkpoint(checkpoint_path)
        if not checkpoint_info:
            return {'status': 'error', 'error': 'Failed to load checkpoint'}
        
        # Evaluate scenario using dedicated evaluator
        result = self._evaluate_scenario_with_evaluator(scenario_name, checkpoint_info, 0, 1)
        
        # Generate charts for single scenario
        chart_files = self._generate_single_scenario_charts(scenario_name, checkpoint_info, result)
        
        return {
            'status': 'success',
            'scenario_name': scenario_name,
            'checkpoint_info': checkpoint_info,
            'metrics': result['metrics'],
            'additional_data': result.get('additional_data', {}),
            'chart_files': chart_files
        }
    
    def _evaluate_scenario_with_evaluator(self, scenario_name: str, checkpoint_info: Dict[str, Any], 
                                        checkpoint_idx: int, total_checkpoints: int) -> Dict[str, Any]:
        """🧪 Evaluate scenario using dedicated scenario evaluator"""
        
        # Create scenario evaluator
        scenario_evaluator = create_scenario_evaluator(
            scenario_manager=self.scenario_manager,
            inference_timer=self.inference_timer,
            num_classes=17
        )
        
        # Create progress callback
        def progress_callback(progress: int, message: str):
            self.progress_bridge.update_metrics(progress, message)
        
        # Evaluate scenario
        return scenario_evaluator.evaluate_scenario(
            scenario_name=scenario_name,
            checkpoint_info=checkpoint_info,
            model_api=self.model_api,
            progress_callback=progress_callback
        )
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """📥 Load checkpoint with validation for latest model architecture"""
        try:
            checkpoint_info = self.checkpoint_selector.select_checkpoint(checkpoint_path)
            
            # Skip checkpoint loading if model is already loaded (optimization)
            if checkpoint_info.get('model_loaded', False) and checkpoint_info.get('optimized_run', False):
                self.logger.debug(f"🚀 Skipping checkpoint reload - model already loaded (optimization)")
                return checkpoint_info
            
            if self.model_api:
                # Load model with checkpoint using latest API
                load_result = self.model_api.load_checkpoint(checkpoint_path)
                if load_result.get('success', False):
                    checkpoint_info['model_loaded'] = True
                    checkpoint_info['architecture_type'] = load_result.get('architecture_type', 'yolov5')
                    checkpoint_info['model_info'] = load_result.get('model_info', {})
                    self.logger.info(f"✅ Model loaded: {checkpoint_info['display_name']} ({checkpoint_info['architecture_type']})")
                else:
                    self.logger.warning(f"⚠️ Model load failed: {checkpoint_path}")
                    checkpoint_info['model_loaded'] = False
            else:
                # Create model API using checkpoint metadata
                checkpoint_info['model_loaded'] = self._create_model_api_from_checkpoint(checkpoint_info, checkpoint_path)
            
            return checkpoint_info
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint load error: {str(e)}")
            return None
    
    def _create_model_api_from_checkpoint(self, checkpoint_info: Dict[str, Any], checkpoint_path: str) -> bool:
        """🔧 Create model API from checkpoint metadata"""
        try:
            from smartcash.model.api.core import create_api
            
            # Extract model configuration from checkpoint
            model_config = self.model_config_extractor.extract_model_config_from_checkpoint(checkpoint_info)
            
            # Create API with extracted configuration
            self.model_api = create_api(
                config=model_config,
                use_yolov5_integration=True
            )
            
            # Build model first with proper configuration
            build_result = self.model_api.build_model(
                backbone=checkpoint_info.get('backbone', 'cspdarknet'),
                num_classes=17,  # Force hierarchical prediction
                img_size=model_config.get('img_size', 640),
                layer_mode=checkpoint_info.get('layer_mode', 'multi'),
                detection_layers=['layer_1', 'layer_2', 'layer_3'],
                pretrained=False  # We'll load weights from checkpoint
            )
            
            if build_result.get('success', False):
                # Now load checkpoint weights
                load_result = self.model_api.load_checkpoint(checkpoint_path)
                if load_result.get('success', False):
                    checkpoint_info['architecture_type'] = load_result.get('architecture_type', 'yolov5')
                    checkpoint_info['model_info'] = load_result.get('model_info', {})
                    self.logger.info(f"✅ Model API created and loaded: {checkpoint_info['display_name']}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Failed to load checkpoint weights: {checkpoint_path}")
                    return False
            else:
                self.logger.warning(f"⚠️ Failed to build model: {build_result.get('error', 'Unknown error')}")
                return False
                
        except Exception as api_error:
            self.logger.warning(f"⚠️ Failed to create model API: {api_error}")
            return False
    
    def _generate_charts(self, evaluation_results: Dict[str, Any], scenarios: List[str], 
                        checkpoints: List[str], summary: Dict[str, Any]) -> List[str]:
        """📊 Generate evaluation charts"""
        chart_files = []
        if self.chart_generator:
            try:
                self.progress_bridge.update_metrics(95, "Generating evaluation charts")
                self.logger.info("📊 Generating evaluation visualization charts...")
                
                # Prepare data for chart generation
                chart_data = {
                    'results': evaluation_results,
                    'evaluation_info': {
                        'timestamp': summary.get('evaluation_completed_at', ''),
                        'total_scenarios': len(scenarios),
                        'total_checkpoints': len(checkpoints)
                    }
                }
                
                # Generate all charts
                chart_files = self.chart_generator.generate_all_charts(chart_data)
                
                if chart_files:
                    self.logger.info(f"✅ Generated {len(chart_files)} evaluation charts")
                    chart_summary = self.chart_generator.get_chart_summary()
                    self.logger.info(f"📊 Charts saved to: {chart_summary['output_directory']}")
                else:
                    self.logger.warning("⚠️ No charts were generated")
                    
            except Exception as e:
                self.logger.error(f"❌ Chart generation failed: {str(e)}")
                chart_files = []
        
        return chart_files
    
    def _generate_single_scenario_charts(self, scenario_name: str, checkpoint_info: Dict[str, Any], 
                                       result: Dict[str, Any]) -> List[str]:
        """📊 Generate charts for single scenario"""
        chart_files = []
        if self.chart_generator:
            try:
                self.logger.info("📊 Generating single scenario evaluation charts...")
                
                # Prepare data for chart generation (single scenario format)
                chart_data = {
                    'results': [{
                        'scenario_name': scenario_name,
                        'checkpoint_info': checkpoint_info,
                        'metrics': result['metrics'],
                        'additional_data': result.get('additional_data', {})
                    }],
                    'evaluation_info': {
                        'timestamp': result.get('timestamp', ''),
                        'total_scenarios': 1,
                        'total_checkpoints': 1,
                        'single_scenario': True
                    }
                }
                
                # Generate charts (will adapt to single scenario)
                chart_files = self.chart_generator.generate_all_charts(chart_data)
                
                if chart_files:
                    self.logger.info(f"✅ Generated {len(chart_files)} evaluation charts")
                    chart_summary = self.chart_generator.get_chart_summary()
                    self.logger.info(f"📊 Charts saved to: {chart_summary['output_directory']}")
                    
            except Exception as e:
                self.logger.error(f"❌ Chart generation failed: {str(e)}")
                chart_files = []
        
        return chart_files
    
    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """📋 Generate comprehensive evaluation summary"""
        
        # Aggregate results
        aggregated = self.results_aggregator.aggregate_metrics()
        
        # Generate summary
        summary = self.results_aggregator.generate_summary()
        
        return {
            'evaluation_overview': summary.get('evaluation_overview', {}),
            'aggregated_metrics': aggregated,
            'key_findings': summary.get('key_findings', []),
            'recommendations': summary.get('recommendations', []),
            'best_configurations': aggregated.get('best_configurations', {}),
            'backbone_comparison': aggregated.get('backbone_comparison', {}),
            'scenario_comparison': aggregated.get('scenario_comparison', {})
        }


# Factory functions
def create_evaluation_service(model_api=None, config: Dict[str, Any] = None) -> EvaluationService:
    """🏭 Factory for EvaluationService"""
    return EvaluationService(model_api, config)


def run_evaluation_pipeline(scenarios: List[str] = None, checkpoints: List[str] = None,
                           model_api=None, config: Dict[str, Any] = None,
                           progress_callback=None, ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🚀 One-liner for run complete evaluation pipeline"""
    service = create_evaluation_service(model_api, config)
    return service.run_evaluation(scenarios, checkpoints, progress_callback, ui_components=ui_components)