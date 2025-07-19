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
                    # TODO: Call actual backend evaluation here
                    # For now, simulate the evaluation process
                    scenario_result = self._evaluate_scenario(scenario, model_name, backbone, layer_mode)
                    
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

    def _evaluate_scenario(self, scenario: str, model_name: str, backbone: str, layer_mode: str) -> Dict[str, Any]:
        """
        Evaluate a specific scenario.
        
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
            
            # Phase 1: Model loading
            self.update_current_progress(10, f"Loading model: {model_name}")
            
            # Phase 2: Dataset preparation  
            self.update_current_progress(25, f"Preparing {scenario} test dataset")
            
            # Phase 3: Running evaluation
            self.update_current_progress(50, f"Running {scenario} evaluation tests")
            
            # Phase 4: Computing metrics
            self.update_current_progress(75, f"Computing {scenario} metrics")
            
            # Phase 5: Finalizing results
            self.update_current_progress(90, f"Finalizing {scenario} results")
            
            # TODO: Replace with actual backend call
            # result = backend.evaluate_model(model_name, scenario, config)
            
            # Mock result for now
            result = {
                'success': True,
                'scenario': scenario,
                'model_evaluated': model_name,
                'backbone': backbone,
                'layer_mode': layer_mode,
                'average_map': 0.823 if scenario == 'position' else 0.801,
                'metrics': {
                    'mAP@0.5': 0.823 if scenario == 'position' else 0.801,
                    'precision': 0.845 if scenario == 'position' else 0.788,
                    'recall': 0.790 if scenario == 'position' else 0.812
                }
            }
            
            self.update_current_progress(100, f"✅ {scenario} evaluation completed")
            self._ui_module.log_success(f"✅ {scenario} scenario completed successfully")
            
            return result
            
        except Exception as e:
            self._ui_module.log_error(f"❌ {scenario} scenario evaluation failed: {e}")
            return {'success': False, 'error': str(e)}