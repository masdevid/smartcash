#!/usr/bin/env python3
"""
Resume utilities for the unified training pipeline.

This module handles training resumption logic and checkpoint-based training continuation.
"""

import uuid
from typing import Dict, Any, Tuple
from smartcash.common.logger import get_logger
from smartcash.model.training.utils.checkpoint_utils import check_for_resumable_checkpoint

logger = get_logger(__name__)


def handle_resume_training_pipeline(
    resume_info: Dict[str, Any], 
    backbone: str,
    phase_1_epochs: int, 
    phase_2_epochs: int, 
    checkpoint_dir: str, 
    force_cpu: bool = False,
    training_mode: str = 'two_phase',
    single_phase_layer_mode: str = 'multi',
    single_phase_freeze_backbone: bool = False,
    pipeline_instance=None,
    **kwargs
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
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
        pipeline_instance: UnifiedTrainingPipeline instance for calling methods
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
        prep_result = pipeline_instance._phase_preparation(backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs)
        if not prep_result.get('success'):
            raise RuntimeError(f"Preparation failed during resume: {prep_result.get('error')}")
        
        # Phase 2: Build Model (always execute to rebuild model)
        build_result = pipeline_instance._phase_build_model()
        if not build_result.get('success'):
            raise RuntimeError(f"Model build failed during resume: {build_result.get('error')}")
        
        # Load checkpoint state into model
        if resume_info.get('model_state_dict') and pipeline_instance.model:
            try:
                pipeline_instance.model.load_state_dict(resume_info['model_state_dict'])
                logger.info("✅ Model state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model state: {e}")
        
        # Phase 3: Validate Model (always execute to ensure model is ready)
        validate_result = pipeline_instance._phase_validate_model()
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
                phase1_result = pipeline_instance._phase_training_1(start_epoch=resume_epoch + 1)
                if phase1_result.get('success'):
                    phase2_result = pipeline_instance._phase_training_2()
                    
            elif resume_phase == 2:
                # Skip Phase 1, resume from Phase 2
                logger.info(f"🔄 Skipping Phase 1, resuming Phase 2 from epoch {resume_epoch + 1}")
                phase1_result = {'success': True, 'message': 'Completed (loaded from checkpoint)'}
                phase2_result = pipeline_instance._phase_training_2(start_epoch=resume_epoch + 1)
            
            else:
                # Invalid phase, start fresh
                logger.warning(f"⚠️ Invalid resume phase {resume_phase}, starting fresh training")
                phase1_result = pipeline_instance._phase_training_1()
                if phase1_result.get('success'):
                    phase2_result = pipeline_instance._phase_training_2()
        
        else:
            # Single-phase training resume logic
            total_epochs = phase_1_epochs  # Only use phase_1_epochs for single phase
            logger.info(f"🔄 Resuming single phase training from epoch {resume_epoch + 1}")
            phase1_result = {'success': True, 'message': 'Skipped in single phase mode'}
            phase2_result = pipeline_instance._phase_single_training(
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
        
        prep_result = pipeline_instance._phase_preparation(backbone, phase_1_epochs, phase_2_epochs, checkpoint_dir, force_cpu, **kwargs)
        if not prep_result.get('success'):
            return prep_result, {}, {}, {}, {}
        
        build_result = pipeline_instance._phase_build_model()
        if not build_result.get('success'):
            return prep_result, build_result, {}, {}, {}
        
        validate_result = pipeline_instance._phase_validate_model()
        if not validate_result.get('success'):
            return prep_result, build_result, validate_result, {}, {}
        
        if training_mode == 'two_phase':
            phase1_result = pipeline_instance._phase_training_1()
            if not phase1_result.get('success'):
                return prep_result, build_result, validate_result, phase1_result, {}
            
            phase2_result = pipeline_instance._phase_training_2()
        else:
            total_epochs = phase_1_epochs
            phase1_result = {'success': True, 'message': 'Skipped in single phase mode'}
            phase2_result = pipeline_instance._phase_single_training(
                total_epochs,
                layer_mode=single_phase_layer_mode,
                freeze_backbone=single_phase_freeze_backbone
            )
        
        return prep_result, build_result, validate_result, phase1_result, phase2_result


def setup_training_session(resume_from_checkpoint: bool, checkpoint_dir: str, backbone: str) -> Tuple[str, Dict]:
    """
    Setup training session with optional resume capability.
    
    Args:
        resume_from_checkpoint: Whether to check for resumable checkpoints
        checkpoint_dir: Directory to search for checkpoints
        backbone: Model backbone name
        
    Returns:
        Tuple of (session_id, resume_info or None)
    """
    resume_info = None
    
    if resume_from_checkpoint:
        resume_info = check_for_resumable_checkpoint(checkpoint_dir, backbone)
        if resume_info:
            logger.info(f"🔄 Found resumable checkpoint: {resume_info['checkpoint_path']}")
            logger.info(f"   Phase: {resume_info['phase']}, Epoch: {resume_info['epoch']}")
            session_id = resume_info.get('session_id', str(uuid.uuid4())[:8])
        else:
            logger.info("📋 No resumable checkpoint found, starting fresh training")
            session_id = str(uuid.uuid4())[:8]
    else:
        session_id = str(uuid.uuid4())[:8]
    
    return session_id, resume_info


def validate_training_mode_and_params(training_mode: str, single_phase_layer_mode: str, 
                                     single_phase_freeze_backbone: bool, phase_2_epochs: int) -> None:
    """
    Validate training mode and parameters.
    
    Args:
        training_mode: Training mode ('single_phase', 'two_phase')
        single_phase_layer_mode: Layer mode for single-phase training
        single_phase_freeze_backbone: Whether to freeze backbone in single-phase training
        phase_2_epochs: Number of epochs for phase 2
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate training mode
    valid_modes = ['single_phase', 'two_phase']
    if training_mode not in valid_modes:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be one of: {valid_modes}")
    
    # Validate single-phase specific parameters
    if training_mode == 'single_phase':
        valid_layer_modes = ['single', 'multi']
        if single_phase_layer_mode not in valid_layer_modes:
            raise ValueError(f"Invalid single_phase_layer_mode: {single_phase_layer_mode}. Must be one of: {valid_layer_modes}")
        
        # Warn about ignored phase_2_epochs in single-phase mode
        if phase_2_epochs > 0:
            logger.warning(f"⚠️ Single phase mode: ignoring phase2_epochs ({phase_2_epochs}), using only phase1_epochs")
    else:
        # Log warning if single-phase parameters are used in two-phase mode
        if single_phase_layer_mode != 'multi' or single_phase_freeze_backbone != False:
            logger.warning("⚠️ single_phase_layer_mode and single_phase_freeze_backbone are ignored in two_phase mode")