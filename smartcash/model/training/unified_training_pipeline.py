#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/smartcash/model/training/unified_training_pipeline.py

Unified training pipeline that merges model building and training with comprehensive progress tracking.

This module provides the main API `run_full_training_pipeline` that handles the complete
training workflow from preparation to visualization in a unified manner.

Progress Structure (6 phases):
1. Preparation -> Setup environment, check prerequisites
2. Build Model -> Create and configure model architecture  
3. Validate Model -> Verify model is ready for training
4. Training Phase 1 -> Train detection heads (backbone frozen)
5. Training Phase 2 -> Fine-tune entire model (backbone unfrozen)
6. Summary & Visualization -> Generate charts and final summary

Features:
- Platform-aware configuration using presets
- Unified checkpoint management under /data/checkpoints
- Configurable phase 1 and phase 2 epochs
- Comprehensive progress tracking with batch-level updates
- Automatic visualization generation
- Error handling and recovery
"""

import time
import json
import torch
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from smartcash.common.logger import get_logger
from smartcash.model.api.core import create_model_api
from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.model.training.visualization_manager import create_visualization_manager
from smartcash.model.training.utils.progress_tracker import UnifiedProgressTracker
from smartcash.model.training.utils.summary_utils import generate_markdown_summary
from smartcash.model.training.utils.checkpoint_utils import generate_checkpoint_name, save_checkpoint_to_disk
from smartcash.model.training.utils.resume_utils import (
    handle_resume_training_pipeline, setup_training_session, validate_training_mode_and_params
)
from smartcash.model.training.utils.setup_utils import prepare_training_environment
from smartcash.model.training.training_phase_manager import TrainingPhaseManager
from smartcash.model.utils.device_utils import setup_device, model_to_device

logger = get_logger(__name__)

# Functions moved to separate utility files - see utils/metrics_utils.py and utils/progress_tracker.py

class UnifiedTrainingPipeline:
    """Unified training pipeline with comprehensive progress tracking and UI callbacks."""
    
    def __init__(self, 
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None,
                 live_chart_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize unified training pipeline with comprehensive callbacks.
        
        Args:
            progress_callback: Callback for progress updates (phase, current, total, message, **kwargs)
            log_callback: Callback for logging events (level, message, data)
            live_chart_callback: Callback for live chart updates (chart_type, data, config)
            metrics_callback: Callback for metrics updates (phase, epoch, metrics)
            verbose: Enable verbose logging
        """
        self.progress_tracker = UnifiedProgressTracker(progress_callback, verbose)
        self.verbose = verbose
        self.config = None
        self.model_api = None
        self.model = None
        self.visualization_manager = None
        
        # UI Integration callbacks
        self.log_callback = log_callback
        self.live_chart_callback = live_chart_callback
        self.metrics_callback = metrics_callback
        
        # Training state for UI reporting
        self.current_phase = None
        self.training_session_id = None
        self.phase_start_time = None
        self.training_start_time = None
        
    def run_full_training_pipeline(self, 
                                  backbone: str = 'cspdarknet',
                                  phase_1_epochs: int = 1,
                                  phase_2_epochs: int = 1,
                                  checkpoint_dir: str = 'data/checkpoints',
                                  resume_from_checkpoint: bool = True,
                                  force_cpu: bool = False,
                                  training_mode: str = 'two_phase',
                                  single_phase_layer_mode: str = 'multi',
                                  single_phase_freeze_backbone: bool = False,
                                  **kwargs) -> Dict[str, Any]:
        """
        Run the complete unified training pipeline with automatic resume capability.
        
        Args:
            backbone: Model backbone ('cspdarknet' or 'efficientnet_b4')
            phase_1_epochs: Number of epochs for phase 1 (frozen backbone)
            phase_2_epochs: Number of epochs for phase 2 (fine-tuning)
            checkpoint_dir: Directory for checkpoint management
            resume_from_checkpoint: Automatically resume from latest checkpoint if available
            force_cpu: Force CPU usage instead of auto-detecting GPU/MPS (default: False)
            training_mode: Training mode ('single_phase', 'two_phase') (default: 'two_phase')
            single_phase_layer_mode: Layer mode for single-phase training ('single', 'multi') (default: 'multi')
            single_phase_freeze_backbone: Whether to freeze backbone in single-phase training (default: False)
            **kwargs: Additional configuration overrides
            
        Returns:
            Complete training results with all phase information
        """
        try:
            # Validate training mode and parameters
            validate_training_mode_and_params(
                training_mode, single_phase_layer_mode, single_phase_freeze_backbone, phase_2_epochs
            )
            
            # Setup training session with resume capability
            self.training_session_id, resume_info = setup_training_session(
                resume_from_checkpoint, checkpoint_dir, backbone
            )
            
            self.training_start_time = time.time()
            
            logger.info("🚀 Starting Unified Training Pipeline")
            logger.info(f"   Backbone: {backbone}")
            logger.info(f"   Training mode: {training_mode}")
            if training_mode == 'two_phase':
                logger.info(f"   Phase 1 epochs: {phase_1_epochs}")
                logger.info(f"   Phase 2 epochs: {phase_2_epochs}")
            else:
                total_epochs = phase_1_epochs + phase_2_epochs
                logger.info(f"   Single phase epochs: {total_epochs}")
                logger.info(f"   Layer mode: {single_phase_layer_mode}")
                logger.info(f"   Backbone frozen: {single_phase_freeze_backbone}")
            logger.info(f"   Checkpoint dir: {checkpoint_dir}")
            if resume_info:
                logger.info(f"   Resuming from: Phase {resume_info['phase']}, Epoch {resume_info['epoch']}")
            
            # Emit initial log with configuration
            initial_config = {
                'backbone': backbone,
                'phase_1_epochs': phase_1_epochs,
                'phase_2_epochs': phase_2_epochs,
                'checkpoint_dir': checkpoint_dir,
                'session_id': self.training_session_id,
                'resume_info': resume_info,
                **kwargs
            }
            self._emit_log('info', 'Training pipeline started', {'config': initial_config})
            
            # Execute phases based on resume status
            if resume_info:
                # Resume from specific phase/epoch
                prep_result, build_result, validate_result, phase1_result, phase2_result = handle_resume_training_pipeline(
                    resume_info, backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, 
                    training_mode, single_phase_layer_mode, single_phase_freeze_backbone, 
                    pipeline_instance=self, **kwargs
                )
            else:
                # Fresh training - execute all phases
                # Phase 1: Preparation
                prep_result = self._phase_preparation(backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs)
                if not prep_result.get('success'):
                    return prep_result
                
                # Phase 2: Build Model
                build_result = self._phase_build_model()
                if not build_result.get('success'):
                    return build_result
                
                # Phase 3: Validate Model
                validate_result = self._phase_validate_model()
                if not validate_result.get('success'):
                    return validate_result
                
                # Create training phase manager
                training_manager = TrainingPhaseManager(
                    model=self.model,
                    model_api=self.model_api,
                    config=self.config,
                    progress_tracker=self.progress_tracker,
                    emit_metrics_callback=self._emit_metrics,
                    emit_live_chart_callback=self._emit_live_chart,
                    visualization_manager=self.visualization_manager
                )
                
                # Execute training based on mode
                if training_mode == 'two_phase':
                    # Phase 4: Training Phase 1
                    phase1_result = self._phase_training_1_with_manager(training_manager)
                    if not phase1_result.get('success'):
                        return phase1_result
                    
                    # Phase 5: Training Phase 2
                    phase2_result = self._phase_training_2_with_manager(training_manager)
                    if not phase2_result.get('success'):
                        return phase2_result
                else:
                    # Single phase training - only use phase_1_epochs
                    total_epochs = phase_1_epochs
                    phase1_result = {'success': True, 'message': 'Skipped in single phase mode'}
                    phase2_result = self._phase_single_training(
                        total_epochs, 
                        layer_mode=single_phase_layer_mode,
                        freeze_backbone=single_phase_freeze_backbone
                    )
                    if not phase2_result.get('success'):
                        return phase2_result
            
            # Phase 6: Summary & Visualization
            summary_result = self._phase_summary_visualization()
            
            # Complete pipeline
            pipeline_summary = self.progress_tracker.get_summary()
            
            # Generate final markdown summary using utils
            markdown_summary = generate_markdown_summary(
                config=self.config,
                phase_results=getattr(self, 'phase_results', None),
                training_session_id=self.training_session_id,
                training_start_time=self.training_start_time
            )
            
            # Emit final log with complete results
            final_log_data = {
                'pipeline_summary': pipeline_summary,
                'final_training_result': phase2_result,
                'visualization_result': summary_result,
                'markdown_summary': markdown_summary
            }
            self._emit_log('info', 'Training pipeline completed successfully', final_log_data)
            
            return {
                'success': True,
                'message': 'Unified training pipeline completed successfully',
                'pipeline_summary': pipeline_summary,
                'final_training_result': phase2_result,
                'visualization_result': summary_result,
                'config_used': self.config,
                'markdown_summary': markdown_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Unified training pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_summary': self.progress_tracker.get_summary()
            }
    
    def _phase_preparation(self, backbone: str, phase_1_epochs: int, phase_2_epochs: int, 
                          checkpoint_dir: str, force_cpu: bool = False, **kwargs) -> Dict[str, Any]:
        """Phase 1: Preparation - Setup environment and configuration."""
        self.current_phase = 'preparation'
        self.phase_start_time = time.time()
        self.progress_tracker.start_phase('preparation', 100, "Setting up training environment")
        
        self._emit_log('info', 'Starting preparation phase', {
            'backbone': backbone,
            'phase_1_epochs': phase_1_epochs,
            'phase_2_epochs': phase_2_epochs
        })
        
        try:
            # Use setup utils for environment preparation
            self.progress_tracker.update_phase(20, 100, "Preparing training environment")
            result = prepare_training_environment(
                backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs
            )
            
            if result.get('success'):
                self.config = result['config']
                self.progress_tracker.update_phase(100, 100, "Preparation completed")
                self.progress_tracker.complete_phase(result)
                return result
            else:
                self.progress_tracker.complete_phase(result)
                return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Preparation failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _phase_build_model(self) -> Dict[str, Any]:
        """Phase 2: Build Model - Create and configure model architecture."""
        self.progress_tracker.start_phase('build_model', 100, "Building model architecture")
        
        try:
            # Step 1: Create model API
            self.progress_tracker.update_phase(25, 100, "Creating model API")
            self.model_api = create_model_api()
            if not self.model_api:
                raise RuntimeError("Failed to create model API")
            
            # Step 2: Build model with configuration
            self.progress_tracker.update_phase(50, 100, "Building model architecture")
            model_config = self.config['model']
            build_result = self.model_api.build_model(**model_config)
            
            if build_result.get('status') != 'built':
                raise RuntimeError(f"Model build failed: {build_result.get('message', 'Unknown error')}")
            
            # Handle different model API response formats
            if 'model' in build_result:
                self.model = build_result['model']
            elif hasattr(self.model_api, 'model'):
                self.model = self.model_api.model
            else:
                raise RuntimeError("Model not found in build result")
            
            # Step 3: Move model to device
            self.progress_tracker.update_phase(75, 100, "Moving model to device")
            device_config = self.config['device']
            device = setup_device({
                'auto_detect': device_config.get('auto_detect', True), 
                'preferred': device_config['device']
            })
            self.model = model_to_device(self.model, device)
            
            # Step 4: Setup model compilation if supported
            self.progress_tracker.update_phase(90, 100, "Applying model optimizations")
            training_config = self.config['training']
            if training_config.get('compile_model', False) and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("✅ Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"⚠️ Model compilation failed: {e}")
            
            self.progress_tracker.update_phase(100, 100, "Model build completed")
            
            result = {
                'success': True,
                'model_info': build_result,
                'device': str(device),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Model build failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _phase_validate_model(self) -> Dict[str, Any]:
        """Phase 3: Validate Model - Verify model is ready for training."""
        self.progress_tracker.start_phase('validate_model', 100, "Validating model configuration")
        
        try:
            # Step 1: Check model structure
            self.progress_tracker.update_phase(25, 100, "Checking model structure")
            if self.model is None:
                raise RuntimeError("Model not built")
            
            # Step 2: Create data loaders for validation
            self.progress_tracker.update_phase(50, 100, "Creating data loaders")
            data_factory = DataLoaderFactory(self.config)
            train_loader = data_factory.create_train_loader()
            val_loader = data_factory.create_val_loader()
            
            if len(train_loader) == 0:
                raise RuntimeError("No training data available")
            if len(val_loader) == 0:
                logger.warning("⚠️ No validation data available")
            
            # Step 3: Test forward pass
            self.progress_tracker.update_phase(75, 100, "Testing forward pass")
            self.model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                images, targets = sample_batch
                device = next(self.model.parameters()).device
                images = images.to(device)
                
                try:
                    outputs = self.model(images)
                    logger.info("✅ Forward pass successful")
                except Exception as e:
                    raise RuntimeError(f"Forward pass failed: {str(e)}")
            
            self.progress_tracker.update_phase(100, 100, "Model validation completed")
            
            result = {
                'success': True,
                'train_batches': len(train_loader),
                'val_batches': len(val_loader),
                'model_device': str(device),
                'forward_pass_successful': True
            }
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Model validation failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _phase_training_1(self, start_epoch: int = 0) -> Dict[str, Any]:
        """Phase 4: Training Phase 1 - Train detection heads with frozen backbone."""
        # Create training manager and delegate
        training_manager = TrainingPhaseManager(
            model=self.model, model_api=self.model_api, config=self.config,
            progress_tracker=self.progress_tracker, emit_metrics_callback=self._emit_metrics,
            emit_live_chart_callback=self._emit_live_chart, visualization_manager=self.visualization_manager
        )
        return self._phase_training_1_with_manager(training_manager, start_epoch)
    
    def _phase_training_1_with_manager(self, training_manager: TrainingPhaseManager, start_epoch: int = 0) -> Dict[str, Any]:
        """Phase 4: Training Phase 1 using training manager."""
        phase_config = self.config['training_phases']['phase_1']
        epochs = phase_config['epochs']
        
        if start_epoch > 0:
            remaining_epochs = epochs - start_epoch
            self.progress_tracker.start_phase(
                'training_phase_1', remaining_epochs, 
                f"Resuming Phase 1 from epoch {start_epoch + 1} - {remaining_epochs} remaining epochs"
            )
        else:
            self.progress_tracker.start_phase(
                'training_phase_1', epochs, 
                f"Training detection heads (backbone frozen) - {epochs} epochs"
            )
        
        try:
            # Freeze backbone
            self._freeze_backbone()
            
            # Run training using manager
            result = training_manager.run_training_phase(1, epochs, start_epoch=start_epoch)
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Training Phase 1 failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _phase_training_2(self, start_epoch: int = 0) -> Dict[str, Any]:
        """Phase 5: Training Phase 2 - Fine-tune entire model."""
        # Create training manager and delegate
        training_manager = TrainingPhaseManager(
            model=self.model, model_api=self.model_api, config=self.config,
            progress_tracker=self.progress_tracker, emit_metrics_callback=self._emit_metrics,
            emit_live_chart_callback=self._emit_live_chart, visualization_manager=self.visualization_manager
        )
        return self._phase_training_2_with_manager(training_manager, start_epoch)
    
    def _phase_training_2_with_manager(self, training_manager: TrainingPhaseManager, start_epoch: int = 0) -> Dict[str, Any]:
        """Phase 5: Training Phase 2 using training manager."""
        phase_config = self.config['training_phases']['phase_2']
        epochs = phase_config['epochs']
        
        if start_epoch > 0:
            remaining_epochs = epochs - start_epoch
            self.progress_tracker.start_phase(
                'training_phase_2', remaining_epochs,
                f"Resuming Phase 2 from epoch {start_epoch + 1} - {remaining_epochs} remaining epochs"
            )
        else:
            self.progress_tracker.start_phase(
                'training_phase_2', epochs,
                f"Fine-tuning entire model - {epochs} epochs"
            )
        
        try:
            # Unfreeze backbone
            self._unfreeze_backbone()
            
            # Run training using manager
            result = training_manager.run_training_phase(2, epochs, start_epoch=start_epoch)
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Training Phase 2 failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    # Method removed - functionality moved to TrainingPhaseManager
    
    def _phase_single_training(self, total_epochs: int, start_epoch: int = 0, 
                              layer_mode: str = 'multi', freeze_backbone: bool = False) -> Dict[str, Any]:
        """Single phase training using TrainingPhaseManager.
        
        Args:
            total_epochs: Total number of epochs to train for
            start_epoch: Epoch to start training from (0-based). Defaults to 0.
            layer_mode: Layer mode ('single', 'multi') for detection layers
            freeze_backbone: Whether to freeze backbone during training
            
        Returns:
            Dictionary containing training results and metrics
        """
        if start_epoch > 0:
            remaining_epochs = total_epochs - start_epoch
            self.progress_tracker.start_phase(
                'training_phase_single', remaining_epochs,
                f"Resuming single phase training from epoch {start_epoch + 1} - {remaining_epochs} remaining epochs"
            )
        else:
            self.progress_tracker.start_phase(
                'training_phase_single', total_epochs,
                f"Single phase unified training - {total_epochs} epochs"
            )
        
        try:
            # Handle backbone freezing based on configuration
            if freeze_backbone:
                self._freeze_backbone()
                logger.info(f"🔒 Backbone frozen for single phase training")
            else:
                self._unfreeze_backbone()
                logger.info(f"🔓 Backbone unfrozen for single phase training")
            
            # Store original configurations for restoration (with safe access)
            original_loss_config = {}
            if 'training' in self.config and 'loss' in self.config['training']:
                original_loss_config = self.config['training']['loss'].copy()
            
            # Configure loss type based on layer mode using setup utils
            from smartcash.model.training.utils.setup_utils import configure_single_phase_settings
            self.config = configure_single_phase_settings(self.config, layer_mode)
            
            try:
                # Create training manager and run single phase training
                training_manager = TrainingPhaseManager(
                    model=self.model, model_api=self.model_api, config=self.config,
                    progress_tracker=self.progress_tracker, emit_metrics_callback=self._emit_metrics,
                    emit_live_chart_callback=self._emit_live_chart, visualization_manager=self.visualization_manager
                )
                training_manager.set_single_phase_mode(True)
                
                # Run training using phase 1 configuration (phase number doesn't matter for single phase)
                result = training_manager.run_training_phase(1, total_epochs, start_epoch=start_epoch)
                
            finally:
                # Restore original loss configuration if it existed
                if original_loss_config and 'training' in self.config:
                    if 'loss' not in self.config['training']:
                        self.config['training']['loss'] = {}
                    self.config['training']['loss'].update(original_loss_config)
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Single phase training failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _phase_summary_visualization(self) -> Dict[str, Any]:
        """Phase 6: Summary & Visualization - Generate charts and final summary."""
        self.progress_tracker.start_phase('summary_visualization', 100, "Generating training summary and visualizations")
        
        try:
            # Step 1: Create visualization manager
            self.progress_tracker.update_phase(25, 100, "Setting up visualization manager")
            if not self.visualization_manager:
                model_config = self.config['model']
                self.visualization_manager = create_visualization_manager(
                    num_classes_per_layer=model_config['num_classes'],
                    save_dir=self.config['paths']['visualization'],
                    verbose=self.verbose
                )
            
            # Step 2: Generate comprehensive charts
            self.progress_tracker.update_phase(50, 100, "Generating training charts")
            session_id = f"unified_training_{int(time.time())}"
            generated_charts = {}
            
            if self.visualization_manager:
                generated_charts = self.visualization_manager.generate_comprehensive_charts(session_id)
            
            # Step 3: Save training summary
            self.progress_tracker.update_phase(75, 100, "Saving training summary")
            summary = self.progress_tracker.get_summary()
            
            summary_path = Path(self.config['paths']['logs']) / f'training_summary_{session_id}.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.progress_tracker.update_phase(100, 100, "Summary and visualization completed")
            
            result = {
                'success': True,
                'generated_charts': generated_charts,
                'charts_count': len(generated_charts),
                'session_id': session_id,
                'summary_file': str(summary_path),
                'pipeline_summary': summary,
                'visualization_directory': str(self.config['paths']['visualization'] / session_id),
                'chart_paths': {
                    'full_paths': generated_charts,
                    'session_directory': str(self.config['paths']['visualization'] / session_id),
                    'quick_access': {
                        'dashboard': generated_charts.get('dashboard', ''),
                        'training_curves': generated_charts.get('training_curves', ''),
                        'confusion_matrices': {k: v for k, v in generated_charts.items() if 'confusion_matrix' in k}
                    }
                }
            }
            
            if self.verbose:
                logger.info(f"📊 Generated {len(generated_charts)} visualization charts")
                for chart_type, chart_path in generated_charts.items():
                    chart_name = Path(chart_path).name
                    logger.info(f"   • {chart_type.replace('_', ' ').title()}: {chart_name}")
                logger.info(f"📁 Charts saved to: {self.config['paths']['visualization']}/{session_id}/")
                logger.info(f"📄 Training summary saved: {summary_path}")
            
            self.progress_tracker.complete_phase(result)
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Summary & visualization failed: {str(e)}"}
            self.progress_tracker.complete_phase(result)
            return result
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for phase 1."""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            logger.info("🔒 Backbone frozen for phase 1")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters for phase 2."""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            logger.info("🔓 Backbone unfrozen for phase 2")
    
    # Methods removed - functionality moved to TrainingPhaseManager
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], phase_num: int) -> str:
        """Save checkpoint using checkpoint utils."""
        try:
            checkpoint_dir = Path(self.config['paths']['checkpoints'])
            backbone = self.config['model']['backbone']
            
            # Generate checkpoint filename using utils
            checkpoint_name = generate_checkpoint_name(backbone, self.model, self.config, is_best=True)
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            # Try model API first
            if self.model_api:
                checkpoint_info = {
                    'epoch': epoch,
                    'phase': phase_num,
                    'metrics': metrics,
                    'is_best': True,
                    'config': self.config
                }
                
                saved_path = self.model_api.save_checkpoint(**checkpoint_info)
                if saved_path:
                    device = next(self.model.parameters()).device
                    device_type = 'cpu' if device.type == 'cpu' else ('mps' if device.type == 'mps' else 'gpu')
                    layer_mode = self.config['model'].get('layer_mode', 'multi')
                    logger.info(f"💾 Best checkpoint saved: {Path(saved_path).name}")
                    logger.info(f"   Epoch: {epoch + 1}, Phase: {phase_num}, Device: {device_type}, Layer mode: {layer_mode}")
                    return saved_path
            
            # Fallback: use utils for direct save
            success = save_checkpoint_to_disk(
                checkpoint_path=checkpoint_path,
                model_state_dict=self.model.state_dict(),
                epoch=epoch,
                phase=phase_num,
                metrics=metrics,
                config=self.config,
                session_id=self.training_session_id
            )
            
            if success:
                device = next(self.model.parameters()).device
                device_type = 'cpu' if device.type == 'cpu' else ('mps' if device.type == 'mps' else 'gpu')
                layer_mode = self.config['model'].get('layer_mode', 'multi')
                logger.info(f"   Epoch: {epoch + 1}, Phase: {phase_num}, Device: {device_type}, Layer mode: {layer_mode}")
                return str(checkpoint_path)
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def _deep_merge_dict(self, base_dict: dict, override_dict: dict) -> dict:
        """Deep merge two dictionaries, with override_dict taking precedence."""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _emit_log(self, level: str, message: str, data: dict = None):
        """Emit log event to UI via log callback."""
        if self.log_callback:
            try:
                log_data = {
                    'timestamp': time.time(),
                    'phase': self.current_phase,
                    'session_id': self.training_session_id,
                    'message': message,
                    'data': data or {}
                }
                self.log_callback(level, message, log_data)
            except Exception as e:
                logger.warning(f"Log callback error: {e}")
    
    def _emit_live_chart(self, chart_type: str, data: dict, config: dict = None):
        """Emit live chart update to UI via live chart callback."""
        if self.live_chart_callback:
            try:
                chart_data = {
                    'timestamp': time.time(),
                    'phase': self.current_phase,
                    'session_id': self.training_session_id,
                    'chart_type': chart_type,
                    'data': data,
                    'config': config or {}
                }
                self.live_chart_callback(chart_type, chart_data, config)
            except Exception as e:
                logger.warning(f"Live chart callback error: {e}")
    
    def _emit_metrics(self, phase: str, epoch: int, metrics: dict):
        """Emit metrics update to UI via metrics callback."""
        if self.metrics_callback:
            try:
                metrics_data = {
                    'timestamp': time.time(),
                    'phase': phase,
                    'epoch': epoch,
                    'session_id': self.training_session_id,
                    'metrics': metrics,
                    'phase_duration': time.time() - self.phase_start_time if self.phase_start_time else 0,
                    'total_duration': time.time() - self.training_start_time if self.training_start_time else 0
                }
                self.metrics_callback(phase, epoch, metrics_data)
            except Exception as e:
                logger.warning(f"Metrics callback error: {e}")
    
    # Markdown summary generation moved to utils/summary_utils.py
    
    # Checkpoint management moved to utils/checkpoint_utils.py
    
    def _resume_training_pipeline(self, resume_info: Dict[str, Any], backbone: str, 
                                 phase_1_epochs: int, phase_2_epochs: int, 
                                 checkpoint_dir: str, force_cpu: bool = False, 
                                 training_mode: str = 'two_phase',
                                 single_phase_layer_mode: str = 'multi',
                                 single_phase_freeze_backbone: bool = False, **kwargs) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Resume training pipeline from checkpoint.
        
        Args:
            resume_info: Resume information from checkpoint
            backbone: Model backbone
            phase_1_epochs: Phase 1 epochs
            phase_2_epochs: Phase 2 epochs  
            checkpoint_dir: Checkpoint directory
            force_cpu: Force CPU usage instead of auto-detecting GPU/MPS
            training_mode: Training mode ('single_phase', 'two_phase')
            single_phase_layer_mode: Layer mode for single-phase training ('single', 'multi')
            single_phase_freeze_backbone: Whether to freeze backbone in single-phase training
            **kwargs: Additional configuration
            
        Returns:
            Tuple of phase results (prep, build, validate, phase1, phase2)
        """
        try:
            logger.info(f"🔄 Resuming training from checkpoint")
            logger.info(f"   Checkpoint: {resume_info['checkpoint_name']}")
            logger.info(f"   Phase: {resume_info['phase']}, Epoch: {resume_info['epoch']}")
            
            resume_phase = resume_info['phase']
            resume_epoch = resume_info['epoch']
            
            # Phase 1: Preparation (always execute for config setup)
            prep_result = self._phase_preparation(backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs)
            if not prep_result.get('success'):
                raise RuntimeError(f"Preparation failed during resume: {prep_result.get('error')}")
            
            # Phase 2: Build Model (always execute to rebuild model)
            build_result = self._phase_build_model()
            if not build_result.get('success'):
                raise RuntimeError(f"Model build failed during resume: {build_result.get('error')}")
            
            # Load checkpoint state into model
            if resume_info.get('model_state_dict') and self.model:
                try:
                    self.model.load_state_dict(resume_info['model_state_dict'])
                    logger.info("✅ Model state loaded from checkpoint")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load model state: {e}")
            
            # Phase 3: Validate Model (always execute to ensure model is ready)
            validate_result = self._phase_validate_model()
            if not validate_result.get('success'):
                raise RuntimeError(f"Model validation failed during resume: {validate_result.get('error')}")
            
            # Phase 4 & 5: Resume training based on training mode
            phase1_result = {'success': True, 'message': 'Skipped (resumed from later phase)'}
            phase2_result = {'success': True, 'message': 'Skipped (resumed from later phase)'}
            
            if training_mode == 'two_phase':
                # Two-phase training resume logic
                if resume_phase == 1:
                    # Resume from Phase 1
                    logger.info(f"🔄 Resuming Phase 1 from epoch {resume_epoch + 1}")
                    phase1_result = self._phase_training_1(start_epoch=resume_epoch + 1)
                    if phase1_result.get('success'):
                        phase2_result = self._phase_training_2()
                        
                elif resume_phase == 2:
                    # Skip Phase 1, resume from Phase 2
                    logger.info(f"🔄 Skipping Phase 1, resuming Phase 2 from epoch {resume_epoch + 1}")
                    phase1_result = {'success': True, 'message': 'Completed (loaded from checkpoint)'}
                    phase2_result = self._phase_training_2(start_epoch=resume_epoch + 1)
                
                else:
                    # Invalid phase, start fresh
                    logger.warning(f"⚠️ Invalid resume phase {resume_phase}, starting fresh training")
                    phase1_result = self._phase_training_1()
                    if phase1_result.get('success'):
                        phase2_result = self._phase_training_2()
            
            else:
                # Single-phase training resume logic
                total_epochs = phase_1_epochs + phase_2_epochs
                logger.info(f"🔄 Resuming single phase training from epoch {resume_epoch + 1}")
                phase1_result = {'success': True, 'message': 'Skipped in single phase mode'}
                phase2_result = self._phase_single_training(
                    total_epochs, 
                    start_epoch=resume_epoch + 1,
                    layer_mode=single_phase_layer_mode,
                    freeze_backbone=single_phase_freeze_backbone
                )
            
            return prep_result, build_result, validate_result, phase1_result, phase2_result
            
        except Exception as e:
            logger.error(f"❌ Resume failed: {str(e)}")
            # Fall back to fresh training
            logger.info("🔄 Falling back to fresh training")
            
            prep_result = self._phase_preparation(backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs)
            if not prep_result.get('success'):
                return prep_result, {}, {}, {}, {}
            
            build_result = self._phase_build_model()
            if not build_result.get('success'):
                return prep_result, build_result, {}, {}, {}
            
            validate_result = self._phase_validate_model()
            if not validate_result.get('success'):
                return prep_result, build_result, validate_result, {}, {}
            
            phase1_result = self._phase_training_1()
            if not phase1_result.get('success'):
                return prep_result, build_result, validate_result, phase1_result, {}
            
            phase2_result = self._phase_training_2()
            return prep_result, build_result, validate_result, phase1_result, phase2_result


# Main API function moved to smartcash.model.api.core