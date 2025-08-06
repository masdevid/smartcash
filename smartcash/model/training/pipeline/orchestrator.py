"""
Training Pipeline Orchestrator

This module consolidates all high-level orchestration logic for the training pipeline,
following SRP by focusing solely on coordinating the training workflow sequence.
"""

import time
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.model.training.phases.mixins.callbacks import CallbacksMixin
from smartcash.model.training.utils.setup_utils import prepare_training_environment
from smartcash.model.training.utils.resume_utils import handle_resume_training_pipeline
from smartcash.model.training.utils.summary_utils import generate_markdown_summary
from smartcash.model.training.phases import TrainingPhaseExecutor
from smartcash.model.core.weight_transfer_manager import create_weight_transfer_manager
from smartcash.model.config.model_config_manager import (
    create_model_configuration_manager,
)
from smartcash.model.training.phases import create_phase_setup_manager
from smartcash.model.training.visualization_manager import create_visualization_manager

logger = get_logger(__name__)


class PipelineOrchestrator(CallbacksMixin):
    """
    Orchestrates the complete training pipeline workflow.

    Responsibilities:
    - Coordinate training pipeline phases in correct sequence
    - Manage high-level workflow state transitions
    - Handle resume and continuation logic
    - Generate final training summaries

    Does NOT handle:
    - Individual phase training loops (delegated to TrainingPhaseManager)
    - Model building/validation (delegated to ModelManager)
    - Data loading specifics (delegated to components)
    - Model configuration (delegated to ModelConfigurationManager)
    - Training components setup (delegated to PhaseSetupManager)
    """

    def __init__(
        self,
        progress_tracker,
        log_callback=None,
        metrics_callback=None,
        live_chart_callback=None,
        progress_callback=None,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            progress_tracker: Progress tracking instance
            log_callback: Callback for log messages
            metrics_callback: Callback for metrics updates
            live_chart_callback: Callback for live chart updates
            progress_callback: Callback for progress updates
        """
        super().__init__()

        self.progress_tracker = progress_tracker

        # Set callbacks using mixin
        self.set_callbacks(
            log_callback=log_callback,
            metrics_callback=metrics_callback,
            live_chart_callback=live_chart_callback,
            progress_callback=progress_callback,
        )

        # Pipeline state
        self.config = None
        self.model_api = None
        self.model = None
        self.training_results = {}
        self.pipeline_start_time = None

        # Component managers (initialized when pipeline starts)
        self.model_config_manager = None
        self.phase_setup_manager = None
        self.training_visualization_manager = None

    def execute_pipeline(
        self, config: Dict[str, Any], model_api, model
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        Args:
            config: Training configuration
            model_api: Model API instance
            model: PyTorch model instance

        Returns:
            Dictionary containing pipeline execution results
        """
        self.pipeline_start_time = time.time()
        self.config = config
        self.model_api = model_api
        self.model = model

        # Initialize component managers
        self.model_config_manager = create_model_configuration_manager(model, config)
        self.phase_setup_manager = create_phase_setup_manager(model, config)
        
        # Initialize training visualization manager
        num_classes_per_layer = {
            'layer_1': 7,  # Banknote denominations
            'layer_2': 7,  # Denomination features  
            'layer_3': 3   # Common features
        }
        self.training_visualization_manager = create_visualization_manager(
            num_classes_per_layer=num_classes_per_layer,
            save_dir="outputs/training_visualizations",
            verbose=config.get("verbose", False)
        )

        try:
            self.emit_log("info", "🚀 Starting training pipeline orchestration")

            # Phase 1: Environment preparation
            self.progress_tracker.update_overall_progress("Preparation", 1, 7)
            self._execute_preparation_phase(config)

            # Phase 2: Handle resume if applicable
            self.progress_tracker.update_overall_progress("Handle Resume", 2, 7)
            resume_info = self._execute_resume_phase(config)

            # Phase 3: Execute training phases
            self.progress_tracker.update_overall_progress("Running Phase 1", 3, 7)
            training_results = self._execute_training_phases(config, resume_info)

            # Phase 4: Generate final summary
            self.progress_tracker.update_overall_progress("Summary Generation", 7, 7)
            final_results = self._execute_finalization_phase(training_results)

            self.emit_log("info", "✅ Pipeline orchestration completed successfully")

            return final_results

        except Exception as e:
            self.emit_log("error", f"❌ Pipeline orchestration failed: {str(e)}")
            raise

        finally:
            self._cleanup_pipeline_resources()

    def _execute_preparation_phase(self, config: Dict[str, Any]):
        """Execute environment preparation phase."""
        self.emit_log("info", "🔧 Executing preparation phase")

        try:
            prepare_training_environment(
                backbone=config.get("backbone"),
                pretrained=config.get("pretrained"),
                phase_1_epochs=config.get("phase_1_epochs"),
                phase_2_epochs=config.get("phase_2_epochs"),
                checkpoint_dir=config.get("checkpoint_dir"),
                data_dir=config.get("data_dir"),
                output_dir="outputs",
            )

            self.emit_log("info", "✅ Preparation phase completed")

        except Exception as e:
            self.emit_log("error", f"❌ Preparation phase failed: {str(e)}")
            raise

    def _execute_resume_phase(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute resume phase if applicable."""
        self.emit_log("info", "🔍 Checking for resume capabilities")

        resume_from_checkpoint = config.get("resume_from_checkpoint", False)

        if not resume_from_checkpoint:
            self.emit_log("info", "📋 No resume requested - starting fresh training")
            return None

        try:
            # This would integrate with resume handling logic
            self.emit_log("info", "🔄 Resume functionality would be executed here")
            return None  # Placeholder

        except Exception as e:
            self.emit_log("error", f"❌ Resume phase failed: {str(e)}")
            raise

    def _execute_training_phases(
        self, config: Dict[str, Any], resume_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute the actual training phases using TrainingPhaseExecutor."""
        self.emit_log("info", "🎯 Executing training phases")

        try:
            # Create training phase executor
            phase_executor = TrainingPhaseExecutor(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                log_callback=self.log_callback,
                metrics_callback=self.metrics_callback,
                live_chart_callback=self.live_chart_callback,
                progress_callback=self.progress_callback,
            )
            
            # Pass training visualization manager to phase executor for real-time updates
            if self.training_visualization_manager:
                phase_executor.set_training_visualization_manager(self.training_visualization_manager)

            # Create weight transfer manager
            weight_transfer_manager = create_weight_transfer_manager()

            # Determine training phases to execute
            training_mode = config.get("training", {}).get("training_mode", "two_phase")

            if training_mode == "single_phase":
                training_results = self._execute_single_phase_training(
                    phase_executor, config, resume_info
                )
            else:
                # Two-phase training
                training_results = self._execute_two_phase_training(
                    phase_executor, config, resume_info, weight_transfer_manager
                )

            # Update overall progress after training phases
            if training_mode == "two_phase":
                self.progress_tracker.update_overall_progress("Rebuild Model", 4, 7)
                self.progress_tracker.update_overall_progress("Validate Model", 5, 7)
                self.progress_tracker.update_overall_progress("Running Phase 2", 6, 7)

            self.emit_log("info", "✅ Training phases completed successfully")

            return training_results

        except Exception as e:
            self.emit_log("error", f"❌ Training phases failed: {str(e)}")
            raise

    def _execute_single_phase_training(
        self,
        phase_executor: TrainingPhaseExecutor,
        config: Dict[str, Any],
        resume_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute single-phase training workflow."""
        self.emit_log("info", "🎯 Executing single-phase training workflow")

        # Get phase configuration
        phase_1_config = config["training_phases"]["phase_1"]
        epochs = phase_1_config["epochs"]
        start_epoch = resume_info.get("epoch", 0) if resume_info else 0

        # Configure model and set up Phase 1 components
        self.model_config_manager.configure_model_for_phase(1)
        phase_1_components = self.phase_setup_manager.setup_phase_components(
            1, epochs, checkpoint_manager=phase_executor.checkpoint_manager)

        # Execute Phase 1 only
        phase_1_result = phase_executor.execute_phase(
            phase_num=1,
            epochs=epochs,
            start_epoch=start_epoch,
            training_components=phase_1_components,
        )

        return {
            "training_mode": "single_phase",
            "phase_1": phase_1_result,
            "total_epochs": epochs,
            "final_metrics": phase_1_result.get("best_metrics", {}),
        }

    def _execute_two_phase_training(
        self,
        phase_executor: TrainingPhaseExecutor,
        config: Dict[str, Any],
        resume_info: Optional[Dict[str, Any]],
        weight_transfer_manager,
    ) -> Dict[str, Any]:
        """Execute two-phase training workflow."""
        self.emit_log("info", "🎯 Executing two-phase training workflow")

        results = {"training_mode": "two_phase"}

        # Phase 1: Feature extraction training
        phase_1_epochs = config.get("phase_1_epochs")
        phase_1_start_epoch = (
            resume_info.get("epoch", 0)
            if resume_info and resume_info.get("phase") == 1
            else 0
        )

        if not resume_info or resume_info.get("phase", 1) <= 1:
            self.emit_log("info", "🎯 Starting Phase 1: Feature extraction training")

            # Configure model and set up Phase 1 components
            self.model_config_manager.configure_model_for_phase(1)
            phase_1_components = self.phase_setup_manager.setup_phase_components(
                1, phase_1_epochs, checkpoint_manager=phase_executor.checkpoint_manager
            )

            phase_1_result = phase_executor.execute_phase(
                phase_num=1,
                epochs=phase_1_epochs,
                start_epoch=phase_1_start_epoch,
                training_components=phase_1_components,
            )

            results["phase_1"] = phase_1_result

            # Handle phase transition with model rebuild
            self.emit_log("info", "🔄 Transitioning from Phase 1 to Phase 2")
            phase_executor.handle_phase_transition(2, {})

            # Rebuild model for Phase 2 with proper architecture and weight transfer
            if weight_transfer_manager and phase_1_result.get("final_checkpoint"):
                self.emit_log("info", "🔧 Rebuilding model for Phase 2 with weight transfer...")
                
                rebuild_success, rebuilt_model = weight_transfer_manager.rebuild_model_for_phase2(
                    self.model_api, phase_1_result["final_checkpoint"]
                )
                
                if rebuild_success and rebuilt_model:
                    self.model = rebuilt_model  # Update orchestrator's model reference
                    phase_executor.model = rebuilt_model  # Update phase executor's model reference
                    self.emit_log("info", "✅ Model successfully rebuilt for Phase 2")
                else:
                    self.emit_log("warning", "⚠️ Model rebuild failed, using fresh Phase 2 model")
                    # Fallback: build fresh Phase 2 model without weight transfer
                    build_result = self.model_api.build_model(
                        layer_mode='multi',
                        detection_layers=['layer_1', 'layer_2', 'layer_3']
                    )
                    if build_result['success']:
                        self.model = build_result['model']
                        phase_executor.model = build_result['model']  # Update phase executor's model reference

        # Phase 2: Fine-tuning training
        phase_2_epochs = config.get("phase_2_epochs")
        phase_2_start_epoch = (
            resume_info.get("epoch", 0)
            if resume_info and resume_info.get("phase") == 2
            else 0
        )

        self.emit_log("info", "🎯 Starting Phase 2: Fine-tuning training")

        # Configure model and set up Phase 2 components
        self.model_config_manager.configure_model_for_phase(2)
        phase_2_components = self.phase_setup_manager.setup_phase_components(
            2, phase_2_epochs, checkpoint_manager=phase_executor.checkpoint_manager
        )

        phase_2_result = phase_executor.execute_phase(
            phase_num=2,
            epochs=phase_2_epochs,
            start_epoch=phase_2_start_epoch,
            training_components=phase_2_components,
        )

        results["phase_2"] = phase_2_result

        # Determine final metrics (Phase 2 takes precedence)
        results["final_metrics"] = phase_2_result.get(
            "best_metrics", results.get("phase_1", {}).get("best_metrics", {})
        )
        results["total_epochs"] = phase_1_epochs + phase_2_epochs

        return results

    def _execute_finalization_phase(
        self, training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute finalization phase with summary generation."""
        self.emit_log("info", "📋 Executing finalization phase")

        try:
            # Generate training summary
            pipeline_duration = time.time() - self.pipeline_start_time

            summary_data = {
                **training_results,
                "pipeline_duration": pipeline_duration,
                "config": self.config,
            }

            # Generate markdown summary
            summary_path = generate_markdown_summary(summary_data, self.config)

            # Generate comprehensive training visualizations
            self.emit_log("info", "📊 Generating comprehensive training visualizations...")
            visualization_results = {}
            
            if self.training_visualization_manager:
                try:
                    # Determine current phase for context
                    current_phase = self._determine_training_phase(training_results)
                    session_id = f"training_{self.config.get('backbone', 'unknown')}_{int(time.time())}"
                    
                    # Generate all comprehensive charts
                    generated_charts = self.training_visualization_manager.generate_comprehensive_charts(
                        session_id=session_id,
                        phase_num=current_phase
                    )
                    
                    if generated_charts:
                        visualization_results.update(generated_charts)
                        self.emit_log("info", f"✅ Generated {len(generated_charts)} comprehensive training visualizations")
                        for chart_type, path in generated_charts.items():
                            self.emit_log("info", f"   📊 {chart_type}: {path}")
                    else:
                        self.emit_log("warning", "⚠️ No training visualizations generated (insufficient data)")
                        
                except Exception as e:
                    self.emit_log("error", f"❌ Failed to generate training visualizations: {str(e)}")
                    visualization_results["chart_error"] = str(e)
            else:
                self.emit_log("warning", "⚠️ Training visualization manager not available")

            final_results = {
                **training_results,
                "pipeline_duration": pipeline_duration,
                "summary_path": summary_path,
                "visualization_results": visualization_results,
                "success": True,
            }

            self.emit_log(
                "info",
                "✅ Finalization phase completed",
                {
                    "duration_minutes": pipeline_duration / 60,
                    "summary_path": summary_path,
                },
            )

            return final_results

        except Exception as e:
            self.emit_log("error", f"❌ Finalization phase failed: {str(e)}")
            # Don't raise here - training was successful even if summary fails
            return {
                **training_results,
                "pipeline_duration": time.time() - self.pipeline_start_time,
                "summary_error": str(e),
                "success": True,  # Training itself was successful
            }

    def _determine_training_phase(self, training_results: Dict[str, Any]) -> int:
        """Determine the current training phase from results."""
        try:
            # Check if we have phase 2 results
            if "phase_2" in training_results and training_results["phase_2"]:
                return 2
            # Check training mode
            elif training_results.get("training_mode") == "two_phase":
                return 2  # Two-phase training completed both phases
            else:
                return 1  # Single phase or only phase 1 completed
        except Exception:
            return 1  # Default to phase 1

    def _cleanup_pipeline_resources(self):
        """Clean up pipeline resources."""
        try:
            self.cleanup_callbacks()
            self.config = None
            self.model_api = None
            self.model = None
            self.training_results = {}
            self.model_config_manager = None
            self.phase_setup_manager = None
            self.training_visualization_manager = None

            self.emit_log("info", "🧹 Pipeline resources cleaned up")

        except Exception as e:
            logger.warning(f"⚠️ Error during pipeline cleanup: {e}")
