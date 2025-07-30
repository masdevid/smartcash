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
from smartcash.model.training.training_phase_manager import TrainingPhaseManager
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
        """Phase 2: Model building."""
        try:
            self.progress_tracker.update_phase(1, 4, "🔧 Initializing model API...")
            
            # Create SmartCash API with YOLOv5 integration
            # Delayed import to avoid circular dependency
            from smartcash.model.api.core import create_api
            self.model_api = create_api(
                config=self.config,
                use_yolov5_integration=self.use_yolov5_integration
            )
            
            model_config = self.config.get('model', {})
            
            self.progress_tracker.update_phase(2, 4, f"🏗️ Building yolov5 model...")
            
            # Build model
            build_result = self.model_api.build_model(
                model_config=model_config
            )
            
            if not build_result['success']:
                return build_result
            
            self.model = build_result['model']
            
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
            # Create training phase manager
            phase_manager = TrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback
            )
            
            # Run training phases based on mode
            if config['training_mode'] == 'two_phase':
                # Phase 1: Frozen backbone training
                self.progress_tracker.start_phase('training_phase_1', 6)
                phase1_result = phase_manager.run_training_phase(
                    phase_num=1, 
                    epochs=config['phase_1_epochs'], 
                    start_epoch=0
                )
                
                if not phase1_result.get('success', False):
                    return phase1_result
                
                # Phase 2: Fine-tuning training
                self.progress_tracker.start_phase('training_phase_2', 6)
                phase2_result = phase_manager.run_training_phase(
                    phase_num=2, 
                    epochs=config['phase_2_epochs'], 
                    start_epoch=0
                )
                
                return phase2_result
            else:
                # Single phase training
                self.progress_tracker.start_phase('training_phase_single', 6)
                return phase_manager.run_training_phase(
                    phase_num=1, 
                    epochs=config.get('phase_1_epochs', 10), 
                    start_epoch=0
                )
                
        except ImportError:
            logger.error("Training phase manager not available")
            return {
                'success': False,
                'error': 'Training phase manager not available'
            }
    
    def _run_training_phases_resume(self, config: Dict[str, Any], resume_epoch: int, resume_phase: int) -> Dict[str, Any]:
        """Run training phases with resume logic and correct epoch offsets."""
        
        try:
            # Create training phase manager
            phase_manager = TrainingPhaseManager(
                model=self.model,
                model_api=self.model_api,
                config=config,
                progress_tracker=self.progress_tracker,
                emit_metrics_callback=self.metrics_callback,
                emit_live_chart_callback=self.log_callback
            )
            
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
                        return phase1_result
                    
                    # Phase 2: Fine-tuning training (start from beginning)
                    self.progress_tracker.start_phase('training_phase_2', 6)
                    phase2_result = phase_manager.run_training_phase(
                        phase_num=2, 
                        epochs=config['phase_2_epochs'], 
                        start_epoch=0
                    )
                    
                    return phase2_result
                    
                elif resume_phase == 2:
                    # Skip Phase 1, resume from Phase 2
                    logger.info(f"🔄 Skipping Phase 1, resuming Phase 2 from epoch {resume_epoch}")
                    self.progress_tracker.start_phase('training_phase_2', 6)
                    return phase_manager.run_training_phase(
                        phase_num=2, 
                        epochs=config['phase_2_epochs'], 
                        start_epoch=resume_epoch - 1  # Convert to 0-based
                    )
                else:
                    # Invalid phase, start fresh
                    logger.warning(f"⚠️ Invalid resume phase {resume_phase}, starting fresh training")
                    return self._run_training_phases(config)
            else:
                # Single phase training resume
                logger.info(f"🔄 Resuming single phase training from epoch {resume_epoch}")
                self.progress_tracker.start_phase('training_phase_single', 6)
                return phase_manager.run_training_phase(
                    phase_num=1, 
                    epochs=config.get('phase_1_epochs', 10), 
                    start_epoch=resume_epoch - 1  # Convert to 0-based
                )
                
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