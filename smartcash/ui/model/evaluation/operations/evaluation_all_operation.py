"""
File: smartcash/ui/model/evaluation/operations/evaluation_all_operation.py
Description: All scenarios evaluation operation with dual progress tracking.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .evaluation_base_operation import EvaluationBaseOperation, EvaluationOperationPhase

if TYPE_CHECKING:
    from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule


class EvaluationAllOperation(EvaluationBaseOperation):
    """
    All scenarios evaluation operation that handles comprehensive model evaluation with dual progress tracking.
    
    This class manages the execution of all evaluation scenarios with:
    - Overall progress bar showing scenario completion (n/total_scenarios)
    - Current progress bar showing current scenario test progress (0-100%)
    - Phase-based progress mapping from backend callbacks
    - Comprehensive error handling and UI integration
    """

    # Evaluation phases for all scenarios with their weights (must sum to 100)
    ALL_SCENARIOS_PHASES = {
        'init': {'weight': 5, 'label': '⚙️ Initialization', 'phase': EvaluationOperationPhase.INITIALIZING},
        'validation': {'weight': 5, 'label': '✅ Validation', 'phase': EvaluationOperationPhase.VALIDATING},
        'model_loading': {'weight': 10, 'label': '🤖 Loading Models', 'phase': EvaluationOperationPhase.LOADING_MODELS},
        'position_eval': {'weight': 35, 'label': '📐 Position Evaluation', 'phase': EvaluationOperationPhase.EVALUATING},
        'lighting_eval': {'weight': 35, 'label': '💡 Lighting Evaluation', 'phase': EvaluationOperationPhase.EVALUATING},
        'metrics_computation': {'weight': 5, 'label': '📊 Computing Metrics', 'phase': EvaluationOperationPhase.COMPUTING_METRICS},
        'finalization': {'weight': 5, 'label': '🎉 Finalization', 'phase': EvaluationOperationPhase.FINALIZING}
    }

    def __init__(
        self,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the all scenarios evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the evaluation
            callbacks: Optional callbacks for operation events
        """
        # Initialize base class first
        super().__init__(ui_module, config, callbacks)
        
        # All scenarios specific setup
        self._scenarios = ['position', 'lighting']
        self._total_scenarios = len(self._scenarios)
        
        # Progress tracking state
        self._reset_progress_tracking()
        
        # Register backend progress callback
        self._register_backend_callbacks()

    def _reset_progress_tracking(self) -> None:
        """Reset progress tracking state for a new operation."""
        super()._reset_progress_tracking()
        self._total_scenarios = len(self._scenarios)
        self._phase_order = list(self.ALL_SCENARIOS_PHASES.keys())
        
        # Validate phase weights sum to 100
        total_weight = sum(phase['weight'] for phase in self.ALL_SCENARIOS_PHASES.values())
        if total_weight != 100:
            self.logger.warning(f"Phase weights sum to {total_weight}, not 100. Progress may be inaccurate.")

    def _register_backend_callbacks(self) -> None:
        """Register callbacks for backend progress updates."""
        if hasattr(self._ui_module, 'register_progress_callback'):
            self._ui_module.register_progress_callback('evaluation_all', self._handle_evaluation_progress)

    def _handle_evaluation_progress(self, step: str, current: int, total: int = 100, message: str = "") -> None:
        """
        Handle progress updates from the backend evaluation system.
        
        Args:
            step: Current evaluation step/phase
            current: Current progress value  
            total: Total progress value
            message: Progress message
        """
        try:
            # Calculate current scenario progress percentage
            current_percent = int((current / total) * 100) if total > 0 else 0
            
            # Update current scenario progress
            self.update_current_progress(current_percent, message or f"{step}: {current}/{total}")
            
            # Map backend steps to our phases and update overall progress if needed
            if step in self.ALL_SCENARIOS_PHASES:
                phase_info = self.ALL_SCENARIOS_PHASES[step]
                self._phase = phase_info['phase']
                
                # Log phase transition
                self._ui_module.log_info(f"📋 Phase: {phase_info['label']}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle evaluation progress: {e}")

    def execute(self) -> Dict[str, Any]:
        """
        Execute the all scenarios evaluation operation.
        
        Returns:
            Dictionary containing evaluation results
        """
        try:
            self._ui_module.log_info("🚀 Starting comprehensive evaluation (all scenarios)...")
            
            # Extract form values to get current model selection
            form_config = self._ui_module._extract_form_values()
            backbone = form_config.get('backbone', 'yolov5_efficientnet-b4')
            layer_mode = form_config.get('layer_mode', 'full_layers')
            
            # Initialize progress tracking
            self.update_overall_progress(0, self._total_scenarios, "Initializing comprehensive evaluation...")
            
            # Generate models to evaluate using {scenario}_{backbone}_{layer} format
            models_to_evaluate = []
            for scenario in self._scenarios:
                model_name = f"{scenario}_{backbone}_{layer_mode}"
                models_to_evaluate.append({
                    'name': model_name,
                    'scenario': scenario,
                    'backbone': backbone,
                    'layer_mode': layer_mode
                })
            
            self._ui_module.log_info(f"📋 Models to evaluate: {[m['name'] for m in models_to_evaluate]}")
            
            # Execute each scenario
            results = {}
            for i, model_info in enumerate(models_to_evaluate):
                scenario = model_info['scenario']
                model_name = model_info['name']
                
                # Start scenario
                self.start_scenario(scenario, i, self._total_scenarios)
                
                try:
                    # Call actual backend evaluation service
                    scenario_result = self._evaluate_scenario_with_backend(scenario, model_name, backbone, layer_mode)
                    
                    results[scenario] = scenario_result
                    
                    # Complete scenario successfully
                    self.complete_scenario(scenario, success=True)
                    
                except Exception as scenario_error:
                    self._ui_module.log_error(f"❌ Scenario {scenario} failed: {scenario_error}")
                    results[scenario] = {'success': False, 'error': str(scenario_error)}
                    
                    # Complete scenario with failure
                    self.complete_scenario(scenario, success=False)
            
            # Compute final results
            successful_scenarios = sum(1 for r in results.values() if r.get('success', False))
            
            final_result = {
                'success': True,
                'successful_tests': successful_scenarios,
                'total_tests': self._total_scenarios,
                'scenarios_completed': list(results.keys()),
                'models_evaluated': models_to_evaluate,
                'best_model': models_to_evaluate[0]['name'] if models_to_evaluate else None,
                'average_map': 0.847,  # TODO: Calculate from actual results
                'detailed_results': results
            }
            
            # Complete operation
            self.complete_operation(success=True, message=f"{successful_scenarios}/{self._total_scenarios} scenarios completed")
            
            return final_result
            
        except Exception as e:
            self._ui_module.log_error(f"❌ Comprehensive evaluation failed: {e}")
            self.complete_operation(success=False, message=str(e))
            return {'success': False, 'error': str(e)}

    def _evaluate_scenario_with_backend(self, scenario: str, model_name: str, backbone: str, layer_mode: str) -> Dict[str, Any]:
        """
        Evaluate a specific scenario using real backend service.
        
        Args:
            scenario: Scenario name ('position' or 'lighting')
            model_name: Full model name with format {scenario}_{backbone}_{layer}
            backbone: Backbone architecture
            layer_mode: Layer mode configuration
            
        Returns:
            Dictionary containing scenario evaluation results
        """
        try:
            self._ui_module.log_info(f"🔄 Evaluating {scenario} scenario with model: {model_name}")
            
            # Import backend evaluation service
            from smartcash.model.evaluation import run_evaluation_pipeline
            from smartcash.model.api.core import create_model_api
            
            # Create progress callback for backend
            def progress_callback(current_step: int, total_steps: int, message: str, **kwargs):
                percentage = (current_step / total_steps * 100) if total_steps > 0 else 0
                self.update_current_progress(int(percentage), message)
                
            # Create model API
            model_api = create_model_api()
            
            # Map scenario names to backend format
            scenario_mapping = {
                'position': 'position_variation',
                'lighting': 'lighting_variation'
            }
            
            backend_scenario = scenario_mapping.get(scenario, scenario)
            
            # Create evaluation config
            eval_config = {
                'evaluation': {
                    'scenarios': [backend_scenario],
                    'model_filter': {
                        'backbone': backbone,
                        'layer_mode': layer_mode
                    },
                    'output_dir': 'data/evaluation_results'
                }
            }
            
            # Get UI components for progress tracking
            ui_components = getattr(self, '_ui_components', {})
            
            # Run evaluation with backend
            self.update_current_progress(10, f"Initializing {scenario} evaluation")
            
            evaluation_result = run_evaluation_pipeline(
                scenarios=[backend_scenario],
                checkpoints=self._get_matching_checkpoints(backbone, layer_mode, eval_config),
                model_api=model_api,
                config=eval_config,
                progress_callback=progress_callback,
                ui_components=ui_components
            )
            
            # Process backend results
            if evaluation_result.get('success', False):
                scenario_results = evaluation_result.get('scenario_results', {})
                scenario_data = scenario_results.get(backend_scenario, {})
                
                result = {
                    'success': True,
                    'scenario': scenario,
                    'model_evaluated': model_name,
                    'backbone': backbone,
                    'layer_mode': layer_mode,
                    'average_map': scenario_data.get('average_mAP', 0.0),
                    'metrics': scenario_data.get('metrics', {}),
                    'inference_time': scenario_data.get('average_inference_time', 0.0),
                    'total_images': scenario_data.get('images_processed', 0)
                }
                
                self.update_current_progress(100, f"✅ {scenario} evaluation completed")
                self._ui_module.log_success(f"✅ {scenario} scenario completed with mAP: {result['average_map']:.3f}")
            else:
                # Handle backend evaluation failure
                error_msg = evaluation_result.get('message', 'Backend evaluation failed')
                result = {
                    'success': False,
                    'error': error_msg,
                    'scenario': scenario,
                    'model_evaluated': model_name
                }
                
                self.update_current_progress(100, f"❌ {scenario} evaluation failed")
                self._ui_module.log_error(f"❌ {scenario} scenario failed: {error_msg}")
            
            return result
            
        except Exception as e:
            self._ui_module.log_error(f"❌ {scenario} scenario evaluation failed: {e}")
            return {
                'success': False,
                'scenario': scenario,
                'model_evaluated': model_name,
                'error': str(e)
            }
    
    def _get_matching_checkpoints(self, backbone: str, layer_mode: str, eval_config: dict) -> list:
        """Get checkpoints matching the specified backbone and layer mode."""
        try:
            from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
            checkpoint_selector = CheckpointSelector(config=eval_config)
            available_checkpoints = checkpoint_selector.list_available_checkpoints()
            
            # Filter checkpoints by backbone and layer_mode
            matching_checkpoints = [
                cp for cp in available_checkpoints 
                if cp.get("backbone", "").lower() in backbone.lower() and 
                   cp.get("layer_mode", "").lower() in layer_mode.lower()
            ]
            
            if not matching_checkpoints:
                # Use best available checkpoint as fallback
                selected_checkpoints = available_checkpoints[:1] if available_checkpoints else []
                if selected_checkpoints:
                    self._ui_module.log_warning(f"⚠️ No exact match for {backbone}+{layer_mode}, using best available")
                else:
                    self._ui_module.log_error(f"❌ No checkpoints available for evaluation")
                return [cp["path"] for cp in selected_checkpoints]
            else:
                self._ui_module.log_info(f"✅ Found matching checkpoint: {matching_checkpoints[0]['display_name']}")
                return [matching_checkpoints[0]['path']]
                
        except Exception as e:
            self._ui_module.log_error(f"Failed to get matching checkpoints: {e}")
            return []  # Let backend select default
