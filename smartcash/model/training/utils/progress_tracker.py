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
    """Progress tracker for the unified training pipeline with 6 phases."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, verbose: bool = True):
        """Initialize the progress tracker.
        
        Args:
            progress_callback: Optional callback function for progress updates
            verbose: Whether to enable verbose logging
        """
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.phases = [
            'preparation',
            'build_model', 
            'validate_model',
            'training_phase_1',
            'training_phase_2',
            'summary_visualization'
        ]
        self.current_phase = None
        self.current_phase_index = -1
        self.phase_start_time = None
        self.phase_results = {}
        self.pipeline_start_time = time.time()
    
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
        """Update progress within current phase."""
        if not self.current_phase:
            return
            
        if self.progress_callback:
            self.progress_callback(self.current_phase, current_step, total_steps, message, **kwargs)
    
    def complete_phase(self, result: Dict[str, Any]):
        """Complete the current phase."""
        if not self.current_phase:
            return
            
        duration = time.time() - self.phase_start_time if self.phase_start_time else 0
        result['duration'] = duration
        self.phase_results[self.current_phase] = result
        
        status = "✅" if result.get('success', False) else "❌"
        phase_display = self.current_phase.replace('_', ' ').title()
        
        logger.info(f"{status} {phase_display} completed in {duration:.1f}s")
        
        if self.progress_callback:
            message = f"{phase_display} completed"
            if not result.get('success', False):
                message = f"{phase_display} failed: {result.get('error', 'Unknown error')}"
            self.progress_callback(self.current_phase, 100, 100, message)
    
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