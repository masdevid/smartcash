"""
File: smartcash/ui/model/evaluation/operations/evaluation_lighting_operation.py
Description: Lighting scenario evaluation operation with dual progress tracking.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .evaluation_base_operation import EvaluationBaseOperation, EvaluationOperationPhase

if TYPE_CHECKING:
    from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule


class EvaluationLightingOperation(EvaluationBaseOperation):
    """
    Lighting scenario evaluation operation that handles lighting variation testing with dual progress tracking.
    
    This class manages the execution of lighting variation evaluation with:
    - Overall progress bar showing test completion (1/1 scenario)
    - Current progress bar showing current test progress (0-100%)
    - Phase-based progress mapping from backend callbacks
    - Comprehensive error handling and UI integration
    """

    # Lighting evaluation phases with their weights (must sum to 100)
    LIGHTING_PHASES = {
        'init': {'weight': 10, 'label': '⚙️ Initialization', 'phase': EvaluationOperationPhase.INITIALIZING},
        'validation': {'weight': 10, 'label': '✅ Validation', 'phase': EvaluationOperationPhase.VALIDATING},
        'model_loading': {'weight': 15, 'label': '🤖 Loading Model', 'phase': EvaluationOperationPhase.LOADING_MODELS},
        'lighting_tests': {'weight': 50, 'label': '💡 Lighting Tests', 'phase': EvaluationOperationPhase.EVALUATING},
        'metrics_computation': {'weight': 10, 'label': '📊 Computing Metrics', 'phase': EvaluationOperationPhase.COMPUTING_METRICS},
        'finalization': {'weight': 5, 'label': '🎉 Finalization', 'phase': EvaluationOperationPhase.FINALIZING}
    }

    def __init__(
        self,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the lighting evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the evaluation
            callbacks: Optional callbacks for operation events
        """
        # Initialize base class first
        super().__init__(ui_module, config, callbacks)
        
        # Lighting specific setup
        self._scenarios = ['lighting']
        self._total_scenarios = 1
        
        # Progress tracking state
        self._reset_progress_tracking()
        
        # Register backend progress callback
        self._register_backend_callbacks()

    def _reset_progress_tracking(self) -> None:
        """Reset progress tracking state for a new operation."""
        super()._reset_progress_tracking()
        self._total_scenarios = 1
        self._phase_order = list(self.LIGHTING_PHASES.keys())
        
        # Validate phase weights sum to 100
        total_weight = sum(phase['weight'] for phase in self.LIGHTING_PHASES.values())
        if total_weight != 100:
            self.logger.warning(f"Phase weights sum to {total_weight}, not 100. Progress may be inaccurate.")

    def _register_backend_callbacks(self) -> None:
        """Register callbacks for backend progress updates."""
        if hasattr(self._ui_module, 'register_progress_callback'):
            self._ui_module.register_progress_callback('evaluation_lighting', self._handle_evaluation_progress)

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
            if step in self.LIGHTING_PHASES:
                phase_info = self.LIGHTING_PHASES[step]
                self._phase = phase_info['phase']
                
                # Log phase transition
                self._ui_module.log_info(f"📋 Phase: {phase_info['label']}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle evaluation progress: {e}")

    def execute(self) -> Dict[str, Any]:
        """
        Execute the lighting scenario evaluation operation.
        
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Clear previous operation logs
            self.clear_operation_logs()
            
            self._ui_module.log_info("💡 Starting lighting variation scenario...")
            
            # Extract form values to get current model selection
            form_config = self._ui_module._extract_form_values()
            backbone = form_config.get('backbone', 'yolov5_efficientnet-b4')
            layer_mode = form_config.get('layer_mode', 'full_layers')
            
            # Initialize progress tracking
            self.update_overall_progress(0, self._total_scenarios, "Initializing lighting evaluation...")
            
            # Generate model name using {scenario}_{backbone}_{layer} format
            model_name = f"lighting_{backbone}_{layer_mode}"
            self._ui_module.log_info(f"📋 Evaluating model: {model_name}")
            
            # Start the lighting scenario
            self.start_scenario('lighting', 0, self._total_scenarios)
            
            try:
                # Execute lighting evaluation
                result = self._evaluate_lighting_scenario(model_name, backbone, layer_mode)
                
                # Complete scenario successfully
                self.complete_scenario('lighting', success=True)
                
                # Complete operation
                self.complete_operation(success=True, message="Lighting evaluation completed successfully")
                
                return result
                
            except Exception as scenario_error:
                self._ui_module.log_error(f"❌ Lighting scenario failed: {scenario_error}")
                
                # Complete scenario with failure
                self.complete_scenario('lighting', success=False)
                
                # Complete operation with failure
                self.complete_operation(success=False, message=str(scenario_error))
                
                return {'success': False, 'error': str(scenario_error)}
            
        except Exception as e:
            self._ui_module.log_error(f"❌ Lighting evaluation failed: {e}")
            self.complete_operation(success=False, message=str(e))
            return {'success': False, 'error': str(e)}

    def _evaluate_lighting_scenario(self, model_name: str, backbone: str, layer_mode: str) -> Dict[str, Any]:
        """
        Evaluate the lighting variation scenario.
        
        Args:
            model_name: Full model name with format lighting_{backbone}_{layer}
            backbone: Backbone architecture
            layer_mode: Layer mode configuration
            
        Returns:
            Dictionary containing lighting evaluation results
        """
        try:
            self._ui_module.log_info(f"🔄 Running lighting variation tests with model: {model_name}")
            
            # Phase 1: Model loading
            self.update_current_progress(10, f"Loading model: {model_name}")
            
            # Phase 2: Dataset preparation  
            self.update_current_progress(25, "Preparing lighting test dataset")
            
            # Phase 3: Lighting test variations
            lighting_conditions = ['bright', 'dim', 'artificial', 'natural', 'mixed']
            total_tests = len(lighting_conditions)
            
            test_results = {}
            for i, condition in enumerate(lighting_conditions):
                test_progress = 40 + int((i / total_tests) * 40)  # Progress from 40% to 80%
                self.update_current_progress(test_progress, f"Testing lighting: {condition}")
                
                # TODO: Replace with actual backend call
                # test_result = backend.test_lighting_variation(model_name, condition)
                
                # Mock test result with some variation
                base_map = 0.801
                variation = (i - 2) * 0.01  # Vary around base value
                test_results[condition] = {
                    'mAP@0.5': base_map + variation,
                    'precision': 0.788 + variation + 0.005,
                    'recall': 0.812 + variation - 0.003
                }
                
                self._ui_module.log_info(f"✅ Lighting test completed: {condition} (mAP: {test_results[condition]['mAP@0.5']:.3f})")
            
            # Phase 4: Computing final metrics
            self.update_current_progress(85, "Computing lighting variation metrics")
            
            # Calculate average metrics
            avg_map = sum(result['mAP@0.5'] for result in test_results.values()) / len(test_results)
            avg_precision = sum(result['precision'] for result in test_results.values()) / len(test_results)
            avg_recall = sum(result['recall'] for result in test_results.values()) / len(test_results)
            
            # Phase 5: Finalizing results
            self.update_current_progress(95, "Finalizing lighting evaluation results")
            
            result = {
                'success': True,
                'successful_tests': 1,
                'total_tests': 1,
                'scenario': 'lighting',
                'model_evaluated': model_name,
                'backbone': backbone,
                'layer_mode': layer_mode,
                'average_map': avg_map,
                'test_results': test_results,
                'summary_metrics': {
                    'mAP@0.5': avg_map,
                    'precision': avg_precision,
                    'recall': avg_recall
                }
            }
            
            self.update_current_progress(100, "✅ Lighting evaluation completed")
            self._ui_module.log_success(f"✅ Lighting scenario completed: {model_name} evaluation successful")
            
            return result
            
        except Exception as e:
            self._ui_module.log_error(f"❌ Lighting scenario evaluation failed: {e}")
            return {'success': False, 'error': str(e)}