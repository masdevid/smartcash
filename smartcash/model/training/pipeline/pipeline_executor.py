#!/usr/bin/env python3
"""
Pipeline Executor for Training Pipeline

This module handles the execution flow of training pipeline phases,
coordinating between different components and managing state transitions.
"""

import time
from typing import Dict, Any, Callable, Optional

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.resume_utils import handle_resume_training_pipeline
from smartcash.model.training.utils.setup_utils import prepare_training_environment
from smartcash.model.training.utils.summary_utils import generate_markdown_summary
from smartcash.model.training.phases import TrainingPhaseManager
from smartcash.model.training.utils.weight_transfer import create_weight_transfer_manager
from smartcash.model.training.pipeline.configuration_builder import ConfigurationBuilder
from smartcash.model.core.checkpoints.checkpoint_utils import CheckpointUtils
from smartcash.model.core.model_utils import ModelUtils

# Delayed import to avoid circular dependency - imported where used
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
import torch

logger = get_logger(__name__)


class PipelineExecutor:
    """Executes training pipeline phases in the correct order."""
    
    def __init__(self, 
                 progress_tracker,
                 log_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 use_yolov5_integration: bool = True):
        """
        Initialize pipeline executor.
        
        Args:
            progress_tracker: Progress tracking instance
            log_callback: Callback for logging events
            metrics_callback: Callback for metrics updates
            use_yolov5_integration: Enable YOLOv5 integration
        """
        self.progress_tracker = progress_tracker
        self.log_callback = log_callback  
        self.metrics_callback = metrics_callback
        self.use_yolov5_integration = use_yolov5_integration
        
        # Pipeline state
        self.config = None
        self.model_api = None
        self.model = None
        self.memory_optimizer = None
        self.current_phase = None
        self.phase_start_time = None
    
    def execute_fresh_pipeline(self, config: Dict[str, Any], 
                             session_id: str, training_start_time: float) -> Dict[str, Any]:
        """
        Execute fresh training pipeline.
        
        Args:
            config: Training configuration
            session_id: Training session identifier
            training_start_time: Training start timestamp
            
        Returns:
            Pipeline execution result
        """
        # Phase 1: Preparation
        self.progress_tracker.start_phase('preparation', 5)
        preparation_result = self._phase_preparation(config)
        if not preparation_result['success']:
            return preparation_result
        
        # Phase 2: Build Model
        self.progress_tracker.start_phase('build_model', 4)
        build_result = self._phase_build_model()
        if not build_result['success']:
            return build_result
        
        # Phase 3: Validate Model
        self.progress_tracker.start_phase('validate_model', 3)
        validate_result = self._phase_validate_model()
        if not validate_result['success']:
            return validate_result
        
        # Phase 4 & 5: Training phases
        training_result = self._run_training_phases(config)
        if not training_result['success']:
            return training_result
        
        # Final Phase: Finalize
        self.progress_tracker.start_phase('finalize', 3)
        summary_result = self._phase_finalize(training_result, session_id, training_start_time)
        
        # Return final success result
        return {
            'success': True,
            'session_id': session_id,
            'architecture_type': 'yolov5',
            'training_duration': time.time() - training_start_time,
            'checkpoint_path': training_result.get('checkpoint_path'),
            'metrics': training_result.get('final_metrics', {}),
            'model_info': build_result.get('model_info', {}),
            'summary_path': summary_result.get('summary_path')
        }
    
    def execute_resume_pipeline(self, resume_info: Dict[str, Any], config: Dict[str, Any],
                              session_id: str, training_start_time: float) -> Dict[str, Any]:
        """
        Execute resumed training pipeline.
        
        Args:
            resume_info: Resume information from checkpoint
            config: Training configuration
            session_id: Training session identifier
            training_start_time: Training start timestamp
            
        Returns:
            Pipeline execution result
        """
        # Execute phases directly for resume (bypass old resume utils)
        # Phase 1: Preparation
        self.progress_tracker.start_phase('preparation', 5)
        preparation_result = self._phase_preparation(config)
        if not preparation_result['success']:
            return preparation_result
        
        # Phase 2: Build Model
        self.progress_tracker.start_phase('build_model', 4)
        build_result = self._phase_build_model()
        if not build_result['success']:
            return build_result
        
        # Load checkpoint state into model
        if resume_info.get('model_state_dict') and self.model:
            try:
                self.model.load_state_dict(resume_info['model_state_dict'])
                logger.info("✅ Model state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model state: {e}")
        
        # Phase 3: Validate Model
        self.progress_tracker.start_phase('validate_model', 3)
        validate_result = self._phase_validate_model()
        if not validate_result['success']:
            return validate_result
        
        # Phase 4 & 5: Resume training with correct epoch offsets
        resume_epoch = resume_info.get('epoch', 1)  # 1-based epoch from checkpoint
        resume_phase = resume_info.get('phase', 1)
        
        training_result = self._run_training_phases_resume(config, resume_epoch, resume_phase)
        if not training_result['success']:
            return training_result
        
        # Final Phase: Finalize
        self.progress_tracker.start_phase('finalize', 3)
        summary_result = self._phase_finalize(training_result, session_id, training_start_time)
        
        return {
            'success': True,
            'session_id': session_id,
            'architecture_type': 'yolov5',
            'training_duration': time.time() - training_start_time,
            'checkpoint_path': training_result.get('checkpoint_path'),
            'metrics': training_result.get('final_metrics', {}),
            'model_info': build_result.get('model_info', {}),
            'summary_path': summary_result.get('summary_path'),
            'resumed_from': resume_info
        }
    
    def _phase_preparation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Preparation - Setup environment and configuration."""
        self.current_phase = 'preparation'
        self.phase_start_time = time.time()
        
        self._emit_log('info', 'Starting preparation phase', {
            'backbone': config['backbone'],
            'phase_1_epochs': config['phase_1_epochs'],
            'phase_2_epochs': config['phase_2_epochs']
        })
        
        try:
            self.progress_tracker.update_phase(1, 5, "🔧 Preparing training environment...")
            
            # Use setup utils for environment preparation
            result = prepare_training_environment(
                config['backbone'], 
                config['pretrained'], 
                config['phase_1_epochs'], 
                config['phase_2_epochs'], 
                str(config['checkpoint_dir']), 
                config['force_cpu'], 
                config['training_mode']
            )
            
            if result.get('success'):
                self.config = result['config']
                self.progress_tracker.update_phase(5, 5, "✅ Preparation complete")
                
                return {
                    'success': True,
                    'config': self.config,
                    'message': 'Environment preparation completed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': f"Environment preparation failed: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"❌ Preparation phase failed: {str(e)}")
            return {
                'success': False,
                'error': f"Preparation failed: {str(e)}"
            }
    
    def _phase_build_model(self) -> Dict[str, Any]:
        """Phase 2: Model building with phase-specific configuration."""
        try:
            self.progress_tracker.update_phase(1, 4, "🔧 Initializing model API...")
            
            # Create SmartCash API with YOLOv5 integration
            # Delayed import to avoid circular dependency
            from smartcash.model.api.core import create_api
            
            # DEBUG: Check what config is being passed to create_api
            logger.info(f"🔍 DEBUG: self.config for API creation: {self.config.get('model', {})}")
            
            self.model_api = create_api(
                config=self.config,
                use_yolov5_integration=self.use_yolov5_integration
            )
            
            # CRITICAL: Use phase-specific model configuration for Phase 1
            training_mode = self.config.get('training_mode', 'two_phase')  # Default to two_phase
            logger.info(f"🔍 DEBUG: Detected training_mode = '{training_mode}'")
            
            if training_mode == 'two_phase':
                # For two-phase training, build Phase 1 model with single-layer configuration
                session_id = self.config.get('session_id', 'default')
                config_builder = ConfigurationBuilder(session_id)
                model_config = config_builder.get_phase_specific_model_config(self.config, phase_num=1)
                logger.info("🔧 Building Phase 1 model with single-layer configuration")
                logger.info(f"📊 Phase 1 model config: {model_config}")
            else:
                # For single-phase training, use the base model configuration
                model_config = self.config.get('model', {})
                logger.info("🔧 Building single-phase model with base configuration")
                logger.info(f"📊 Single-phase model config: {model_config}")
            
            # DEBUG: Log what we're actually passing to build_model
            logger.info(f"🔍 DEBUG: model_config being passed to build_model: {model_config}")
            logger.info(f"🔍 DEBUG: Expected to produce {'36 channels (single-layer)' if model_config.get('layer_mode') == 'single' else '66 channels (multi-layer)'}")
            
            self.progress_tracker.update_phase(2, 4, f"🏗️ Building yolov5 model...")
            
            # Build model with phase-specific configuration
            build_result = self.model_api.build_model(
                model_config=model_config
            )
            
            if not build_result['success']:
                return build_result
            
            self.model = build_result['model']
            
            # DEBUG: Check actual model output channels
            try:
                state_dict = self.model.state_dict()
                detection_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
                if detection_keys:
                    actual_channels = state_dict[detection_keys[0]].shape[0]
                    logger.info(f"🔍 DEBUG: Actual model output channels: {actual_channels}")
                    if model_config.get('layer_mode') == 'single' and actual_channels != 36:
                        logger.warning(f"⚠️ MISMATCH: Expected 36 channels for single-layer, got {actual_channels}")
                    elif model_config.get('layer_mode') == 'multi' and actual_channels != 66:
                        logger.warning(f"⚠️ MISMATCH: Expected 66 channels for multi-layer, got {actual_channels}")
                    else:
                        logger.info(f"✅ Channel count matches expectation for {model_config.get('layer_mode', 'unknown')} mode")
                else:
                    logger.warning("⚠️ Could not find detection head layers for channel verification")
            except Exception as e:
                logger.debug(f"Debug channel check failed: {e}")
            
            self.progress_tracker.update_phase(3, 4, "🔧 Setting up training components...")
            
            # Use device from model API build result
            device_str = build_result.get('device', 'cpu')
            device = torch.device(device_str)
            
            # Model is already on the correct device from build_model
            self.memory_optimizer = get_memory_optimizer(device)
            
            self.progress_tracker.update_phase(4, 4, "✅ Model building complete")
            
            if self.log_callback:
                self.log_callback('info', f"Model built successfully", {
                    'architecture_type': 'yolov5',
                    'backbone': build_result.get('model_info', {}).get('backbone', 'unknown'),
                    'parameters': build_result.get('model_info', {}).get('total_parameters', 0)
                })
            
            return {
                'success': True,
                'model': self.model,
                'model_info': build_result.get('model_info', {}),
                'architecture_type': 'yolov5',
                'device': str(device)
            }
            
        except Exception as e:
            logger.error(f"❌ Build phase failed: {str(e)}")
            return {
                'success': False,
                'error': f"Model building failed: {str(e)}"
            }
    
    def _phase_validate_model(self) -> Dict[str, Any]:
        """Phase 3: Model validation."""
        try:
            self.progress_tracker.update_phase(1, 3, "🔍 Validating model...")
            
            # Use API validation if available
            if self.model_api:
                validation_result = self.model_api.validate_model()
                
                if not validation_result['success']:
                    return {
                        'success': False,
                        'error': f"Model validation failed: {validation_result.get('error')}"
                    }
                
                self.progress_tracker.update_phase(2, 3, "📊 Analyzing model architecture...")
                
                # Log validation results
                if self.log_callback:
                    self.log_callback('info', "Model validation successful", {
                        'architecture_type': 'yolov5',
                        'validation_info': validation_result
                    })
            
            self.progress_tracker.update_phase(3, 3, "✅ Model validation complete")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                'success': False,
                'error': f"Model validation failed: {str(e)}"
            }
    
    def _run_training_phases(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training phases with architecture-specific handling."""
        
        try:
            # Create visualization manager for comprehensive training charts
            from smartcash.model.training.visualization_manager import create_visualization_manager
            
            # Define layer configurations for different training modes
            layer_configs = {
                'two_phase': {'layer_1': 7},  # Phase 1: single-layer only
                'single': {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}  # Single-phase: all layers
            }
            training_mode = config.get('training_mode', 'two_phase')
            num_classes_per_layer = layer_configs[training_mode]
            
            visualization_manager = create_visualization_manager(
                num_classes_per_layer=num_classes_per_layer,
                save_dir="data/visualization",
                verbose=True
            )
            # Create training phase manager
            phase_manager = TrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback,
                visualization_manager=visualization_manager
            )
            # Store reference for finalization phase
            self.training_phase_manager = phase_manager
            
            # Run training phases based on mode and start_phase configuration
            start_phase = config.get('start_phase', 1)
            
            if config['training_mode'] == 'two_phase':
                if start_phase == 1:
                    # Normal two-phase training: Start from Phase 1
                    logger.info("🚀 Starting normal two-phase training from Phase 1")
                    
                    # Phase 1: Frozen backbone training
                    self.progress_tracker.start_phase('training_phase_1', 6)
                    phase1_result = phase_manager.run_training_phase(
                        phase_num=1, 
                        epochs=config['phase_1_epochs'], 
                        start_epoch=0
                    )
                    
                    if not phase1_result.get('success', False):
                        # Check if this is an early shutdown - should not continue to Phase 2
                        if phase1_result.get('early_shutdown', False):
                            logger.info("🛑 Phase 1 early shutdown detected - stopping training pipeline")
                        return {
                            'success': phase1_result.get('success', False),
                            'error': phase1_result.get('error', 'Phase 1 training failed'),
                            **phase1_result  # Include any additional fields from the original result
                        }
                    
                    # Phase 1 backup is now created automatically by CheckpointManager when best model is saved
                    # phase1_backup_path = self._create_phase_backup(phase1_result.get('best_checkpoint'), 1)
                    
                    # Transition to Phase 2
                    phase2_result = self._transition_to_phase2_and_train(phase_manager, config, phase1_result)
                    
                    # Phase 2 backup is now created automatically by CheckpointManager when best model is saved
                    # self._create_phase_backup(phase2_result.get('best_checkpoint'), 2)
                    
                    return phase2_result
                    
                elif start_phase == 2:
                    # Jump directly to Phase 2
                    logger.info("🚀 Jumping directly to Phase 2 training")
                    
                    # Check if we should override standard best model with backup
                    if not config.get('resume_checkpoint'):
                        # Starting new Phase 2 training - override standard best.pt with backup if available
                        phase1_backup_path = CheckpointUtils.find_phase_backup_model(1, config)
                        if phase1_backup_path:
                            logger.info(f"📦 Overriding standard best.pt with Phase 1 backup for new Phase 2 training")
                            CheckpointUtils.override_standard_best_with_backup(phase1_backup_path, config)
                        else:
                            # Check if there's an existing standard best model
                            standard_best_path = CheckpointUtils.find_standard_best_model(config)
                            if standard_best_path:
                                logger.warning("⚠️ No Phase 1 backup found - using existing standard best.pt")
                            else:
                                # Neither backup nor standard best model exists - cannot jump to Phase 2
                                error_msg = (
                                    "❌ Cannot jump to Phase 2: No Phase 1 backup model found and no standard best model exists. "
                                    "Please either:\n"
                                    "  1. Run Phase 1 training first (--start-phase 1), or\n"
                                    "  2. Provide an existing standard best model in the checkpoint directory"
                                )
                                logger.error(error_msg)
                                return {
                                    'success': False,
                                    'error': error_msg
                                }
                    else:
                        # Resume mode - keep standard best.pt as-is
                        logger.info("🔄 Resume mode: keeping standard best.pt unchanged")
                    
                    # Jump directly to Phase 2 training (always uses standard best.pt)
                    phase2_result = self._run_phase2_only(phase_manager, config)
                    
                    # Phase 2 backup is now created automatically by CheckpointManager when best model is saved
                    # self._create_phase_backup(phase2_result.get('best_checkpoint'), 2)
                    
                    return phase2_result
            else:
                # Single phase training
                self.progress_tracker.start_phase('training_phase_single', 6)
                result = phase_manager.run_training_phase(
                    phase_num=1, 
                    epochs=config.get('phase_1_epochs', 10), 
                    start_epoch=0
                )
                # Ensure result has 'success' key
                if 'success' not in result:
                    return {
                        'success': False,
                        'error': 'Training phase did not return success status',
                        **result  # Include any additional fields from the original result
                    }
                return result
                
        except ImportError:
            logger.error("Training phase manager not available")
            return {
                'success': False,
                'error': 'Training phase manager not available'
            }
        except Exception as e:
            logger.error(f"❌ Training phases failed: {str(e)}")
            return {
                'success': False,
                'error': f"Training phases failed: {str(e)}"
            }
    
    def _run_training_phases_resume(self, config: Dict[str, Any], resume_epoch: int, resume_phase: int) -> Dict[str, Any]:
        """Run training phases with resume logic and correct epoch offsets."""
        
        try:
            # Create visualization manager for comprehensive training charts
            from smartcash.model.training.visualization_manager import create_visualization_manager
            
            # Use phase-aware layer configuration based on training mode
            training_mode = config.get('training_mode', 'two_phase')
            if training_mode == 'two_phase':
                # Phase 1 starts with single-layer (layer_1 only)
                num_classes_per_layer = {'layer_1': 7}
            else:
                # Single-phase uses all layers
                num_classes_per_layer = {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}
            
            visualization_manager = create_visualization_manager(
                num_classes_per_layer=num_classes_per_layer,
                save_dir="data/visualization",
                verbose=True
            )
            # Create training phase manager
            phase_manager = TrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback,
                visualization_manager=visualization_manager
            )
            # Store reference for finalization phase
            self.training_phase_manager = phase_manager
            
            # Run training phases based on mode and resume info
            if config['training_mode'] == 'two_phase':
                if resume_phase == 1:
                    # Resume from Phase 1
                    self.progress_tracker.start_phase('training_phase_1', 6)
                    start_epoch = resume_epoch - 1  # Convert to 0-based
                    logger.info(f"🔄 Resuming Phase 1 from epoch {resume_epoch}")
                    phase1_result = phase_manager.run_training_phase(
                        phase_num=1, 
                        epochs=config['phase_1_epochs'], 
                        start_epoch=start_epoch
                    )
                    
                    if not phase1_result.get('success', False):
                        # Check if this is an early shutdown - should not continue to Phase 2
                        if phase1_result.get('early_shutdown', False):
                            logger.info("🛑 Phase 1 resume early shutdown detected - stopping training pipeline")
                        phase1_result['completed_phase'] = 1
                        phase1_result['training_mode'] = 'two_phase'
                        return phase1_result
                    
                    # Phase 1 backup is now created automatically by CheckpointManager when best model is saved
                    # self._create_phase_backup(phase1_result.get('best_checkpoint'), 1)
                    
                    # Transition to Phase 2
                    phase2_result = self._transition_to_phase2_and_train(phase_manager, config, phase1_result)
                    
                    # Phase 2 backup is now created automatically by CheckpointManager when best model is saved
                    # self._create_phase_backup(phase2_result.get('best_checkpoint'), 2)
                    
                    # Add phase information for finalization
                    phase2_result['completed_phase'] = 2
                    phase2_result['training_mode'] = 'two_phase'
                    return phase2_result
                    
                elif resume_phase == 2:
                    # Skip Phase 1, resume from Phase 2
                    logger.info(f"🔄 Skipping Phase 1, resuming Phase 2 from epoch {resume_epoch}")
                    
                    # Unfreeze backbone for Phase 2 resume
                    ModelUtils.unfreeze_backbone_for_phase2(self.model)
                    
                    # CRITICAL: Propagate Phase 2 state to all model components for resume
                    logger.info("🔄 Propagating Phase 2 state to model components (resume mode)")
                    try:
                        if hasattr(phase_manager, 'phase_orchestrator'):
                            phase_manager.phase_orchestrator.propagate_phase_to_model(self.model, 2)
                            logger.info("✅ Phase 2 state propagated via phase orchestrator (resume)")
                        else:
                            # Fallback: direct phase setting
                            if hasattr(self.model, 'current_phase'):
                                self.model.current_phase = 2
                                logger.info("✅ Set model.current_phase = 2 (resume)")
                            else:
                                logger.warning("⚠️ Model doesn't have current_phase attribute (resume)")
                    except Exception as e:
                        logger.warning(f"⚠️ Phase propagation failed (resume): {e}")
                    
                    # CRITICAL FIX: Reset best metrics tracking for resumed Phase 2
                    # This prevents Phase 2 from comparing against Phase 1's best metrics
                    logger.info("🔄 Preparing for resumed Phase 2: resetting best metrics tracking")
                    phase_manager.handle_phase_transition(2, {})
                    
                    self.progress_tracker.start_phase('training_phase_2', 6)
                    phase2_result = phase_manager.run_training_phase(
                        phase_num=2, 
                        epochs=config['phase_2_epochs'], 
                        start_epoch=resume_epoch - 1  # Convert to 0-based
                    )
                    
                    # Phase 2 backup is now created automatically by CheckpointManager when best model is saved
                    # self._create_phase_backup(phase2_result.get('best_checkpoint'), 2)
                    
                    # Add phase information for finalization
                    phase2_result['completed_phase'] = 2
                    phase2_result['training_mode'] = 'two_phase'
                    return phase2_result
                else:
                    # Invalid phase, start fresh
                    logger.warning(f"⚠️ Invalid resume phase {resume_phase}, starting fresh training")
                    return self._run_training_phases(config)
            else:
                # Single phase training resume
                logger.info(f"🔄 Resuming single phase training from epoch {resume_epoch}")
                self.progress_tracker.start_phase('training_phase_single', 6)
                single_result = phase_manager.run_training_phase(
                    phase_num=1, 
                    epochs=config.get('phase_1_epochs', 10), 
                    start_epoch=resume_epoch - 1  # Convert to 0-based
                )
                
                # Add phase information for finalization
                single_result['completed_phase'] = 1
                single_result['training_mode'] = 'single_phase'
                return single_result
                
        except ImportError:
            logger.error("Training phase manager not available")
            return {
                'success': False,
                'error': 'Training phase manager not available'
            }
    
    def _phase_finalize(self, training_result: Dict[str, Any], 
                       session_id: str, training_start_time: float) -> Dict[str, Any]:
        """Phase 6: Finalization with summary and visualization."""
        try:
            self.progress_tracker.update_phase(1, 3, "📊 Generating summary...")
            
            # Get model summary
            model_summary = {}
            if self.model_api:
                model_summary = self.model_api.get_model_summary()
            
            self.progress_tracker.update_phase(2, 3, "📈 Creating visualizations...")
            
            # Generate visualization charts
            generated_charts = {}
            if hasattr(self, 'training_phase_manager') and self.training_phase_manager:
                progress_manager = getattr(self.training_phase_manager, 'progress_manager', None)
                if progress_manager and hasattr(progress_manager, 'visualization_manager'):
                    visualization_manager = progress_manager.visualization_manager
                    if visualization_manager:
                        try:
                            logger.info("📊 Generating comprehensive training visualization charts...")
                            generated_charts = visualization_manager.generate_comprehensive_charts(session_id)
                            logger.info(f"✅ Generated {len(generated_charts)} visualization charts")
                        except Exception as viz_error:
                            logger.warning(f"⚠️ Error generating visualization charts: {viz_error}")
                            # Continue with finalization even if visualization fails
                    else:
                        logger.debug("No visualization manager available for chart generation")
                else:
                    logger.debug("No progress manager or visualization manager available")
            else:
                logger.debug("No training phase manager available for visualization")
            
            # Generate markdown summary
            markdown_summary = generate_markdown_summary(
                config=self.config,
                phase_results=getattr(self, 'phase_results', None),
                training_session_id=session_id,
                training_start_time=training_start_time
            )
            
            # Final memory cleanup
            if self.memory_optimizer:
                self.memory_optimizer.cleanup_memory(aggressive=True)
                logger.info("🧹 Final memory cleanup completed")
            
            self.progress_tracker.update_phase(3, 3, "✅ Finalization complete")
            
            return {
                'success': True,
                'architecture_type': 'yolov5',
                'model_summary': model_summary,
                'markdown_summary': markdown_summary,
                'generated_charts': generated_charts,
                'charts_count': len(generated_charts),
                'summary_path': None  # Could be enhanced to save summary to file
            }
            
        except Exception as e:
            logger.error(f"❌ Summary phase failed: {str(e)}")
            return {
                'success': False,
                'error': f"Summary generation failed: {str(e)}"
            }
    
    def _emit_log(self, level: str, message: str, data: dict = None):
        """Emit log event to UI via log callback."""
        if self.log_callback:
            try:
                log_data = {
                    'timestamp': time.time(),
                    'phase': self.current_phase,
                    'message': message,
                    'data': data or {}
                }
                self.log_callback(level, message, log_data)
            except Exception as e:
                logger.warning(f"Log callback error: {e}")
    
    def _transition_to_phase2_and_train(self, phase_manager, config: Dict[str, Any], phase1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transition from Phase 1 to Phase 2 and run Phase 2 training."""
        # CRITICAL: Phase 1 to Phase 2 transition logic
        
        # 1. Save Phase 1 checkpoint data for loading into rebuilt model
        phase1_checkpoint_data = None
        if phase1_result.get('best_checkpoint'):
            try:
                checkpoint_path = phase1_result['best_checkpoint']
                logger.info(f"🔄 Loading Phase 1 best checkpoint data for Phase 2 transition: {checkpoint_path}")
                
                import torch
                # Use weights_only=False for compatibility with PyTorch 2.6+
                phase1_checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                logger.info("✅ Phase 1 checkpoint data loaded for model rebuilding")
            except Exception as e:
                logger.error(f"❌ Failed to load Phase 1 checkpoint: {e}")
                # Continue anyway - Phase 2 will start from current model state
        
        # 2. Determine Phase 2 model configuration using phase-specific logic
        logger.info("🔧 Determining Phase 2 model configuration for single→multi transition")
        
        # Get phase-specific configuration for Phase 2 (multi-layer)
        session_id = config.get('session_id', 'default')
        config_builder = ConfigurationBuilder(session_id)
        phase2_model_config = config_builder.get_phase_specific_model_config(config, phase_num=2)
        
        # Update the config for Phase 2 rebuild
        config = config.copy()
        config['model'] = phase2_model_config
        
        logger.info(f"🔧 Phase 2 model config: {phase2_model_config}")
        
        # 3. Rebuild model with Phase 2 configuration (multi-layer, unfrozen backbone)
        logger.info("🏗️ Rebuilding model with Phase 2 configuration (multi-layer, unfrozen backbone)")
        phase2_model = ModelUtils.rebuild_model_for_phase2(self.model_api, self.model, config)
        
        # 4. Handle weight transfer from Phase 1 (single-layer) to Phase 2 (multi-layer)
        if phase1_checkpoint_data and 'model_state_dict' in phase1_checkpoint_data:
            # CRITICAL FIX: Extract model configuration from checkpoint if available
            checkpoint_model_config = None
            if 'model_config' in phase1_checkpoint_data and phase1_checkpoint_data['model_config']:
                saved_config = phase1_checkpoint_data['model_config']
                logger.info(f"🔧 Found model config in checkpoint: {saved_config}")
                
                # CRITICAL: Validate that saved config matches actual model architecture
                # Check if the saved config num_classes matches the model's output channels
                state_dict = phase1_checkpoint_data.get('model_state_dict', {})
                detection_head_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
                
                if detection_head_keys:
                    actual_output_channels = state_dict[detection_head_keys[0]].shape[0]
                    saved_num_classes = saved_config.get('num_classes', 0)
                    
                    # For multi-layer: 17 classes should give 66 output channels
                    # For single-layer: 17 classes should give 102 output channels  
                    expected_channels_multi = (7 + 7 + 3) * 6  # 102, but YOLOv5 uses different calculation = 66
                    expected_channels_single = 17 * 6  # 102
                    
                    if actual_output_channels == 66 and saved_num_classes != 17:
                        logger.warning(f"⚠️ Config mismatch: saved num_classes={saved_num_classes}, but architecture suggests 17 classes (66 channels)")
                        logger.info("🔍 Using inference to get correct configuration")
                        checkpoint_path = phase1_result.get('best_checkpoint', '')
                        inferred_config = ModelUtils.infer_model_config_from_checkpoint(phase1_checkpoint_data, checkpoint_path)
                        if inferred_config:
                            logger.info(f"🔧 Using inferred config instead: {inferred_config}")
                            checkpoint_model_config = inferred_config
                        else:
                            checkpoint_model_config = saved_config
                    else:
                        checkpoint_model_config = saved_config
                else:
                    checkpoint_model_config = saved_config
            else:
                # Infer configuration from checkpoint state dict structure
                logger.info("🔍 Model config empty in checkpoint, inferring from state dict structure")
                # Pass checkpoint path context for better backbone inference
                checkpoint_path = phase1_result.get('best_checkpoint', '')
                checkpoint_model_config = ModelUtils.infer_model_config_from_checkpoint(phase1_checkpoint_data, checkpoint_path)
                if checkpoint_model_config:
                    logger.info(f"🔧 Inferred model config: {checkpoint_model_config}")
            
            if checkpoint_model_config:
                # Update the config used for Phase 2 rebuild with checkpoint config
                if 'model' not in config:
                    config['model'] = {}
                config['model'].update(checkpoint_model_config)
                logger.info("✅ Updated Phase 2 config with checkpoint model configuration")
                
                # Rebuild model again with correct configuration from checkpoint
                logger.info("🔄 Rebuilding model with correct checkpoint configuration")
                corrected_model = ModelUtils.rebuild_model_for_phase2(self.model_api, self.model, config)
                phase2_model = corrected_model
            
            # Use WeightTransferManager for intelligent single→multi weight transfer
            logger.info("🔄 Using WeightTransferManager for single→multi architecture transition")
            
            weight_transfer_manager = create_weight_transfer_manager()
            
            # First validate compatibility
            compatibility = weight_transfer_manager.validate_transfer_compatibility(
                phase1_checkpoint_data, phase2_model
            )
            
            logger.info(f"🔍 Weight transfer compatibility: {compatibility}")
            
            if compatibility['compatible']:
                # Perform the weight transfer
                transfer_success, transfer_info = weight_transfer_manager.transfer_single_to_multi_weights(
                    phase1_checkpoint_data, 
                    phase2_model,
                    transfer_mode=compatibility.get('transfer_strategy', 'expand')
                )
                
                if transfer_success:
                    logger.info("✅ Single→Multi weight transfer completed successfully")
                    logger.info(f"📊 Transfer details:")
                    logger.info(f"   • Mode: {transfer_info.get('transfer_mode')}")
                    logger.info(f"   • Transferred layers: {len(transfer_info.get('transferred_layers', []))}")
                    logger.info(f"   • Initialized layers: {len(transfer_info.get('initialized_layers', []))}")
                    logger.info(f"   • Single-layer info: {transfer_info.get('single_info', {})}")
                    logger.info(f"   • Multi-layer info: {transfer_info.get('multi_info', {})}")
                else:
                    logger.warning(f"⚠️ Weight transfer failed: {transfer_info}")
                    logger.info("🔄 Phase 2 will use newly initialized model weights")
            else:
                logger.warning(f"⚠️ Weight transfer not compatible: {compatibility}")
                logger.info("🔄 Phase 2 will use newly initialized model weights")
                logger.info("🔍 Recommendation: Check if multi-layer configuration was properly preserved during rebuild")
        
        # 4. Replace the current model with the rebuilt Phase 2 model
        self.model = phase2_model
        logger.info("🔄 Model replaced with Phase 2 configuration")
        
        # CRITICAL FIX: Reset best metrics tracking for Phase 2
        # This prevents Phase 2 from comparing against Phase 1's best metrics
        logger.info("🔄 Preparing for Phase 2: resetting best metrics tracking")
        phase_manager.handle_phase_transition(2, {})
        
        # Phase 2: Fine-tuning training
        self.progress_tracker.start_phase('training_phase_2', 6)
        return phase_manager.run_training_phase(
            phase_num=2, 
            epochs=config['phase_2_epochs'], 
            start_epoch=0
        )
    
    
    def _run_phase2_only(self, phase_manager, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Phase 2 training only (for direct Phase 2 start)."""
        # Save standard best model data before rebuilding
        standard_checkpoint_data = None
        standard_best_path = CheckpointUtils.find_standard_best_model(config)
        
        if standard_best_path:
            try:
                import torch
                logger.info(f"🔄 Loading standard best model data for Phase 2: {standard_best_path}")
                # Use weights_only=False for compatibility with PyTorch 2.6+
                standard_checkpoint_data = torch.load(standard_best_path, map_location='cpu', weights_only=False)
                logger.info("✅ Standard best model data loaded for model rebuilding")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load standard best model: {e}")
        
        # Rebuild model with unfrozen backbone configuration for Phase 2
        logger.info("🏗️ Rebuilding model with unfrozen backbone configuration for Phase 2")
        phase2_model = ModelUtils.rebuild_model_for_phase2(self.model_api, self.model, config)
        
        # Load standard best model weights into the rebuilt model
        if standard_checkpoint_data and 'model_state_dict' in standard_checkpoint_data:
            # CRITICAL FIX: Extract model configuration from checkpoint if available
            checkpoint_model_config = None
            if 'model_config' in standard_checkpoint_data and standard_checkpoint_data['model_config']:
                saved_config = standard_checkpoint_data['model_config']
                logger.info(f"🔧 Found model config in standard checkpoint: {saved_config}")
                
                # CRITICAL: Validate that saved config matches actual model architecture
                # Check if the saved config num_classes matches the model's output channels
                state_dict = standard_checkpoint_data.get('model_state_dict', {})
                detection_head_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
                
                if detection_head_keys:
                    actual_output_channels = state_dict[detection_head_keys[0]].shape[0]
                    saved_num_classes = saved_config.get('num_classes', 0)
                    
                    # For multi-layer: 17 classes should give 66 output channels
                    # For single-layer: 17 classes should give 102 output channels  
                    expected_channels_multi = (7 + 7 + 3) * 6  # 102, but YOLOv5 uses different calculation = 66
                    expected_channels_single = 17 * 6  # 102
                    
                    if actual_output_channels == 66 and saved_num_classes != 17:
                        logger.warning(f"⚠️ Config mismatch: saved num_classes={saved_num_classes}, but architecture suggests 17 classes (66 channels)")
                        logger.info("🔍 Using inference to get correct configuration")
                        inferred_config = ModelUtils.infer_model_config_from_checkpoint(standard_checkpoint_data, standard_best_path)
                        if inferred_config:
                            logger.info(f"🔧 Using inferred config instead: {inferred_config}")
                            checkpoint_model_config = inferred_config
                        else:
                            checkpoint_model_config = saved_config
                    else:
                        checkpoint_model_config = saved_config
                else:
                    checkpoint_model_config = saved_config
            else:
                # Infer configuration from checkpoint state dict structure
                logger.info("🔍 Model config empty in standard checkpoint, inferring from state dict structure")
                checkpoint_model_config = ModelUtils.infer_model_config_from_checkpoint(standard_checkpoint_data, standard_best_path)
                if checkpoint_model_config:
                    logger.info(f"🔧 Inferred model config: {checkpoint_model_config}")
            
            if checkpoint_model_config:
                # Update the config used for Phase 2 rebuild with checkpoint config
                if 'model' not in config:
                    config['model'] = {}
                config['model'].update(checkpoint_model_config)
                logger.info("✅ Updated Phase 2 config with standard checkpoint model configuration")
                
                # Rebuild model again with correct configuration from checkpoint
                logger.info("🔄 Rebuilding model with correct checkpoint configuration")
                corrected_model = ModelUtils.rebuild_model_for_phase2(self.model_api, self.model, config)
                phase2_model = corrected_model
            
            try:
                # Attempt to load state dict with strict=False to handle minor mismatches
                missing_keys, unexpected_keys = phase2_model.load_state_dict(standard_checkpoint_data['model_state_dict'], strict=False)
                
                if missing_keys or unexpected_keys:
                    logger.warning(f"⚠️ Partial weight loading - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        logger.debug(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                    logger.info("✅ Standard best model weights loaded with some architecture differences (non-critical layers skipped)")
                else:
                    logger.info("✅ Standard best model weights loaded completely into rebuilt Phase 2 model")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load standard best weights into rebuilt model: {e}")
                logger.info("🔄 Phase 2 will use newly initialized model weights")
                
                # Additional debugging information
                if "size mismatch" in str(e):
                    logger.info("🔍 Architecture mismatch detected - this may indicate configuration preservation issue")
                    logger.info("🔍 Recommendation: Check if multi-layer configuration was properly preserved during rebuild")
        
        # Replace the current model with the rebuilt Phase 2 model
        self.model = phase2_model
        logger.info("🔄 Model replaced with Phase 2 configuration")
        
        # CRITICAL: Propagate Phase 2 state to all model components after model replacement
        logger.info("🔄 Propagating Phase 2 state to model components")
        try:
            # Use phase orchestrator to properly set phase state
            if hasattr(phase_manager, 'phase_orchestrator'):
                phase_manager.phase_orchestrator.propagate_phase_to_model(self.model, 2)
                logger.info("✅ Phase 2 state propagated via phase orchestrator")
            else:
                # Fallback: direct phase setting
                if hasattr(self.model, 'current_phase'):
                    self.model.current_phase = 2
                    logger.info("✅ Set model.current_phase = 2")
                else:
                    logger.warning("⚠️ Model doesn't have current_phase attribute")
        except Exception as e:
            logger.warning(f"⚠️ Phase propagation failed: {e}")
        
        # CRITICAL FIX: Reset best metrics tracking for Phase 2
        # This prevents Phase 2 from comparing against Phase 1's best metrics
        logger.info("🔄 Preparing for Phase 2: resetting best metrics tracking")
        phase_manager.handle_phase_transition(2, {})
        
        # Phase 2: Fine-tuning training
        self.progress_tracker.start_phase('training_phase_2', 6)
        return phase_manager.run_training_phase(
            phase_num=2, 
            epochs=config['phase_2_epochs'], 
            start_epoch=0
        )
    