#!/usr/bin/env python3
"""
Session Manager for Training Pipeline

This module manages training sessions, including session identification,
resume capability, and session state tracking.
"""

import uuid
import time
from typing import Dict, Any, Optional, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.resume_utils import (
    setup_training_session, validate_training_mode_and_params
)

logger = get_logger(__name__)


class SessionManager:
    """Manages training session lifecycle and state."""
    
    def __init__(self):
        """Initialize session manager."""
        self.session_id = None
        self.training_start_time = None
        self.current_phase = None
        self.phase_start_time = None
        self.session_metadata = {}
    
    def create_session(self, backbone: str, training_mode: str, 
                      resume_from_checkpoint: bool = False,
                      checkpoint_dir: str = 'data/checkpoints',
                      resume_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Create or resume a training session.
        
        Args:
            backbone: Model backbone type
            training_mode: Training mode ('single_phase' or 'two_phase')
            resume_from_checkpoint: Whether to resume from checkpoint
            checkpoint_dir: Checkpoint directory path
            resume_info: Pre-loaded resume info (for legacy checkpoint support)
            
        Returns:
            Tuple of (session_id, resume_info)
        """
        # Generate session ID
        self.session_id = str(uuid.uuid4())[:8]
        self.training_start_time = time.time()
        
        # Setup session metadata
        self.session_metadata = {
            'backbone': backbone,
            'training_mode': training_mode,
            'resume_from_checkpoint': resume_from_checkpoint,
            'checkpoint_dir': checkpoint_dir,
            'created_at': self.training_start_time
        }
        
        # Handle resume capability
        if resume_from_checkpoint:
            # Use pre-loaded resume_info if provided (legacy checkpoint support)
            if resume_info:
                self.session_id = resume_info.get('session_id', self.session_id)
                self.session_metadata['resumed_from'] = resume_info
                logger.info(f"📁 Session resumed from legacy checkpoint: Phase {resume_info['phase']}, Epoch {resume_info['epoch']}")
            else:
                # Try standard checkpoint discovery
                try:
                    _, resume_info = setup_training_session(
                        resume_from_checkpoint, checkpoint_dir, backbone
                    )
                    if resume_info:
                        self.session_metadata['resumed_from'] = resume_info
                        logger.info(f"📁 Session resumed from checkpoint: Phase {resume_info['phase']}, Epoch {resume_info['epoch']}")
                except Exception as e:
                    logger.warning(f"Failed to setup resume session: {e}")
        
        logger.info(f"🆔 Training session created: {self.session_id}")
        return self.session_id, resume_info
    
    def validate_session_params(self, training_mode: str, 
                               single_phase_layer_mode: str = 'multi',
                               single_phase_freeze_backbone: bool = False,
                               phase_2_epochs: int = 1) -> bool:
        """
        Validate session parameters.
        
        Args:
            training_mode: Training mode to validate
            single_phase_layer_mode: Layer mode for single phase
            single_phase_freeze_backbone: Whether to freeze backbone
            phase_2_epochs: Number of phase 2 epochs
            
        Returns:
            True if parameters are valid
        """
        try:
            validate_training_mode_and_params(
                training_mode, single_phase_layer_mode, 
                single_phase_freeze_backbone, phase_2_epochs
            )
            return True
        except Exception as e:
            logger.error(f"Session parameter validation failed: {e}")
            return False
    
    def start_phase(self, phase_name: str):
        """
        Start a new training phase.
        
        Args:
            phase_name: Name of the phase being started
        """
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        phase_duration = 0
        if self.training_start_time:
            phase_duration = self.phase_start_time - self.training_start_time
        
        logger.info(f"📍 Phase started: {phase_name} (Session: {self.session_id}, Elapsed: {phase_duration:.1f}s)")
    
    def complete_phase(self, phase_name: str):
        """
        Complete current training phase.
        
        Args:
            phase_name: Name of the phase being completed
        """
        if self.phase_start_time:
            phase_duration = time.time() - self.phase_start_time
            logger.info(f"✅ Phase completed: {phase_name} (Duration: {phase_duration:.1f}s)")
        
        self.current_phase = None
        self.phase_start_time = None
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.
        
        Returns:
            Dictionary containing session information
        """
        info = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'metadata': self.session_metadata.copy()
        }
        
        if self.training_start_time:
            info['total_duration'] = time.time() - self.training_start_time
        
        if self.phase_start_time:
            info['phase_duration'] = time.time() - self.phase_start_time
        
        return info
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive session summary.
        
        Returns:
            Dictionary containing session summary
        """
        summary = {
            'session_id': self.session_id,
            'total_duration': time.time() - self.training_start_time if self.training_start_time else 0,
            'phases_completed': [],
            'metadata': self.session_metadata.copy(),
            'status': 'completed' if self.current_phase is None else 'running'
        }
        
        # Add current phase info if running
        if self.current_phase:
            summary['current_phase'] = {
                'name': self.current_phase,
                'duration': time.time() - self.phase_start_time if self.phase_start_time else 0
            }
        
        return summary
    
    def cleanup_session(self):
        """Clean up session resources."""
        if self.session_id:
            total_duration = time.time() - self.training_start_time if self.training_start_time else 0
            logger.info(f"🧹 Session cleanup: {self.session_id} (Duration: {total_duration:.1f}s)")
        
        self.session_id = None
        self.training_start_time = None
        self.current_phase = None
        self.phase_start_time = None
        self.session_metadata.clear()