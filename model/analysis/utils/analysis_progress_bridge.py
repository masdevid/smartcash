"""
File: smartcash/model/analysis/utils/analysis_progress_bridge.py
Deskripsi: Progress tracking bridge untuk analysis operations dengan UI integration
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class AnalysisProgressBridge:
    """Bridge untuk progress tracking analysis operations dengan UI integration"""
    
    def __init__(self, progress_callback: Optional[Callable] = None, logger=None):
        self.progress_callback = progress_callback
        self.logger = logger or get_logger('analysis_progress')
        self.current_operation = None
        self.total_steps = 0
        self.current_step = 0
        
    def start_analysis(self, operation_name: str = "Analysis", total_steps: int = 6) -> None:
        """Start analysis tracking dengan total steps"""
        self.current_operation = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        
        self.logger.info(f"🚀 Starting {operation_name} dengan {total_steps} steps")
        
        if self.progress_callback:
            try:
                self.progress_callback({
                    'operation': operation_name,
                    'step_name': 'Initialization',
                    'current_step': 0,
                    'total_steps': total_steps,
                    'progress_percent': 0,
                    'message': f'Memulai {operation_name}...'
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Progress callback error: {str(e)}")
    
    def update_step(self, step_name: str, progress_percent: float, message: str) -> None:
        """Update progress untuk specific step"""
        self.current_step += 1
        
        self.logger.info(f"📊 Step {self.current_step}/{self.total_steps}: {step_name} ({progress_percent:.1f}%)")
        
        if self.progress_callback:
            try:
                self.progress_callback({
                    'operation': self.current_operation,
                    'step_name': step_name,
                    'current_step': self.current_step,
                    'total_steps': self.total_steps,
                    'progress_percent': progress_percent,
                    'message': message,
                    'overall_progress': (self.current_step / self.total_steps) * 100
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Progress callback error: {str(e)}")
    
    def update_substep(self, substep_name: str, substep_progress: float, detail_message: str) -> None:
        """Update sub-step progress dalam current step"""
        if self.progress_callback:
            try:
                self.progress_callback({
                    'operation': self.current_operation,
                    'step_name': f"Step {self.current_step}",
                    'substep_name': substep_name,
                    'current_step': self.current_step,
                    'total_steps': self.total_steps,
                    'substep_progress': substep_progress,
                    'message': detail_message,
                    'overall_progress': ((self.current_step - 1 + substep_progress / 100) / self.total_steps) * 100
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Substep callback error: {str(e)}")
    
    def complete_analysis(self, completion_message: str = "Analysis completed successfully!") -> None:
        """Mark analysis sebagai complete"""
        self.logger.info(f"✅ {self.current_operation} completed: {completion_message}")
        
        if self.progress_callback:
            try:
                self.progress_callback({
                    'operation': self.current_operation,
                    'step_name': 'Complete',
                    'current_step': self.total_steps,
                    'total_steps': self.total_steps,
                    'progress_percent': 100,
                    'overall_progress': 100,
                    'message': completion_message,
                    'status': 'completed'
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Completion callback error: {str(e)}")
    
    def analysis_error(self, error_message: str, current_step: Optional[str] = None) -> None:
        """Report analysis error dengan context"""
        step_context = current_step or f"Step {self.current_step}"
        self.logger.error(f"❌ Error in {self.current_operation} at {step_context}: {error_message}")
        
        if self.progress_callback:
            try:
                self.progress_callback({
                    'operation': self.current_operation,
                    'step_name': step_context,
                    'current_step': self.current_step,
                    'total_steps': self.total_steps,
                    'progress_percent': (self.current_step / self.total_steps) * 100,
                    'message': f"Error: {error_message}",
                    'status': 'error',
                    'error': error_message
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Error callback error: {str(e)}")
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status"""
        return {
            'operation': self.current_operation,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
            'is_complete': self.current_step >= self.total_steps
        }