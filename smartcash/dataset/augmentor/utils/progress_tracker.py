"""
File: smartcash/dataset/augmentor/utils/progress_tracker.py
Deskripsi: SRP progress tracker dengan UI communicator integration yang mulus
"""

from typing import Optional, Callable, Dict, Any

class ProgressTracker:
    """SRP progress tracker dengan UI integration dan real-time updates"""
    
    def __init__(self, communicator=None):
        self.communicator = communicator
        self.current_operation = None
        self.steps_completed = 0
        self.total_steps = 100
        
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Start operation dengan UI notification"""
        self.current_operation = operation_name
        self.total_steps = total_steps
        self.steps_completed = 0
        
        if self.communicator:
            self.communicator.start_operation(operation_name, total_steps)
    
    def progress(self, level: str, current: int, total: int, message: str = ""):
        """Update progress dengan level specificity"""
        if self.communicator:
            self.communicator.progress(level, current, total, message)
        
        # Update internal state
        if level == 'overall':
            self.steps_completed = current
    
    def complete(self, message: str = ""):
        """Complete operation dengan success message"""
        if self.communicator:
            self.communicator.complete_operation(self.current_operation or "Operation", message)
        self.current_operation = None
    
    def error(self, error_message: str):
        """Handle operation error"""
        if self.communicator:
            self.communicator.error_operation(self.current_operation or "Operation", error_message)
        self.current_operation = None
    
    def log_info(self, message: str):
        """Log info message"""
        if self.communicator:
            self.communicator.log_info(message)
    
    def log_success(self, message: str):
        """Log success message"""
        if self.communicator:
            self.communicator.log_success(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        if self.communicator:
            self.communicator.log_warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        if self.communicator:
            self.communicator.log_error(message)

def create_progress_tracker(communicator=None) -> ProgressTracker:
    """Factory untuk progress tracker dengan communicator"""
    return ProgressTracker(communicator)