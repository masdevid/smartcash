#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/progress_tracker.py

Progress tracking utilities for unified training pipeline.
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class UnifiedProgressTracker:
    """
    Progress tracker for the unified training pipeline with 3-level progress:
    1. Overall: Preparation -> Build Model -> Validate Model -> Start Train Phase 1 -> [Start Train Phase 2] -> Finalize
    2. Epoch: Current epoch progress (with early stopping support)
    3. Batch: Current batch progress within epoch
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None, verbose: bool = True, training_mode: str = 'two_phase'):
        """Initialize the progress tracker.
        
        Args:
            progress_callback: Optional callback function for progress updates
            verbose: Whether to enable verbose logging
            training_mode: 'single_phase' or 'two_phase' to determine total phases
        """
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.training_mode = training_mode
        
        # Overall phases - 5 phases for single, 6 phases for two_phase
        if training_mode == 'single_phase':
            self.phases = [
                'preparation',
                'build_model', 
                'validate_model',
                'training_phase_1',
                'finalize'
            ]
        else:  # two_phase
            self.phases = [
                'preparation',
                'build_model', 
                'validate_model',
                'training_phase_1',
                'training_phase_2',
                'finalize'
            ]
        
        # Overall progress tracking
        self.current_phase = None
        self.current_phase_index = -1
        self.phase_start_time = None
        self.phase_results = {}
        self.pipeline_start_time = time.time()
        
        # Epoch progress tracking
        self.current_epoch = 0
        self.total_epochs = 0
        self.epoch_completed = False
        self.early_stopping_triggered = False
        
        # Batch progress tracking
        self.current_batch = 0
        self.total_batches = 0
        self.batch_progress_active = False
    
    def start_phase(self, phase_name: str, total_steps: int, description: str = ""):
        """Start a new phase."""
        self.current_phase = phase_name
        self.current_phase_index = self.phases.index(phase_name) if phase_name in self.phases else -1
        self.phase_start_time = time.time()
        
        logger.info(f"🚀 Starting {phase_name.replace('_', ' ').title()}")
        logger.info(f"   {description}")
        
        if self.progress_callback:
            self.progress_callback(phase_name, 0, total_steps, f"Starting {phase_name.replace('_', ' ').title()}")
    
    def update_phase(self, current_step: int, total_steps: int, message: str = "", **kwargs):
        """Update progress within current phase.
        
        Args:
            current_step: Current step in the phase (0-based)
            total_steps: Total steps in the phase
            message: Optional status message
            **kwargs: Additional arguments to pass to the callback
            
        Note:
            Ensures progress stays within 0-100% range to prevent tqdm warnings.
        """
        if not self.current_phase:
            return
            
        # Calculate overall progress
        phase_progress = (self.current_phase_index / max(1, len(self.phases))) * 100  # Avoid division by zero
        step_progress = (current_step / max(1, total_steps)) * (100 / max(1, len(self.phases)))  # Avoid division by zero
        overall_progress = min(100.0, max(0.0, phase_progress + step_progress))  # Clamp between 0 and 100
            
        if self.progress_callback:
            self.progress_callback('overall', overall_progress, 100, message, **kwargs)
    
    def start_epoch_tracking(self, total_epochs: int):
        """Start epoch progress tracking."""
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.epoch_completed = False
        self.early_stopping_triggered = False
        
        if self.progress_callback:
            self.progress_callback('epoch', 0, total_epochs, f"Starting epoch training ({total_epochs} epochs)")
    
    def update_epoch_progress(self, current_epoch: int, total_epochs: int = None, message: str = ""):
        """Update epoch progress."""
        self.current_epoch = current_epoch
        if total_epochs:
            self.total_epochs = total_epochs
            
        if self.progress_callback:
            self.progress_callback('epoch', current_epoch, self.total_epochs, message)
    
    def complete_epoch_early_stopping(self, final_epoch: int, message: str = "Early stopping triggered"):
        """Complete epoch tracking due to early stopping."""
        self.early_stopping_triggered = True
        self.current_epoch = final_epoch
        
        # Set progress to 100% when early stopping
        if self.progress_callback:
            self.progress_callback('epoch', 100, 100, message)
    
    def start_batch_tracking(self, total_batches: int):
        """Start batch progress tracking."""
        self.total_batches = total_batches
        self.current_batch = 0
        self.batch_progress_active = True
        
        if self.progress_callback:
            self.progress_callback('batch', 0, total_batches, f"Starting batch processing ({total_batches} batches)")
    
    def update_batch_progress(self, current_batch: int, total_batches: int = None, message: str = "", loss: float = None, **kwargs):
        """Update batch progress."""
        if not self.batch_progress_active:
            return
            
        self.current_batch = current_batch
        if total_batches:
            self.total_batches = total_batches
        
        # Add loss to message if provided
        if loss is not None:
            message = f"{message} (Loss: {loss:.4f})" if message else f"Loss: {loss:.4f}"
            
        if self.progress_callback:
            # Pass through additional kwargs (like epoch) to the callback
            self.progress_callback('batch', current_batch, self.total_batches, message, loss=loss, **kwargs)
    
    def complete_batch_tracking(self):
        """Complete batch progress tracking."""
        self.batch_progress_active = False
        
        if self.progress_callback:
            self.progress_callback('batch', 100, 100, "Batch processing completed")
    
    def complete_phase(self, result: Dict[str, Any]):
        """Complete the current phase."""
        if not self.current_phase:
            return
            
        duration = time.time() - self.phase_start_time if self.phase_start_time else 0
        result['duration'] = duration
        self.phase_results[self.current_phase] = result
        
        success = result.get('success', False)
        status = "✅" if success else "❌"
        phase_display = self.current_phase.replace('_', ' ').title()
        
        logger.info(f"{status} {phase_display} completed in {duration:.1f}s")
        
        if self.progress_callback:
            if success:
                # Successful completion - report 100%
                message = f"✅ {phase_display} completed"
                self.progress_callback(self.current_phase, 100, 100, message)
            else:
                # Failed completion - report partial progress to avoid confusion
                error_msg = result.get('error', 'Unknown error')
                message = f"❌ {phase_display} failed: {error_msg}"
                # Report 99% instead of 100% to indicate incomplete/failed state
                self.progress_callback(self.current_phase, 99, 100, message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete pipeline summary."""
        total_duration = time.time() - self.pipeline_start_time
        phases_completed = len([r for r in self.phase_results.values() if r.get('success', False)])
        
        return {
            'total_duration': total_duration,
            'phases_completed': phases_completed,
            'total_phases': len(self.phases),
            'success': phases_completed == len(self.phases),
            'phase_results': self.phase_results,
            'phases': self.phases
        }
    
    def get_phase_result(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific phase."""
        return self.phase_results.get(phase_name)
    
    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase was completed successfully."""
        result = self.phase_results.get(phase_name)
        return result is not None and result.get('success', False)
    
    def get_current_phase_info(self) -> Dict[str, Any]:
        """Get information about current phase."""
        if not self.current_phase:
            return {}
            
        return {
            'phase': self.current_phase,
            'phase_index': self.current_phase_index,
            'phase_display': self.current_phase.replace('_', ' ').title(),
            'duration': time.time() - self.phase_start_time if self.phase_start_time else 0
        }