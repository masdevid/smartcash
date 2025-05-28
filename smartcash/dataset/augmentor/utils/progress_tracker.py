"""
File: smartcash/dataset/augmentor/utils/progress_tracker.py
Deskripsi: Fixed progress tracker dengan granular step tracking dan UI sync
"""

from smartcash.common.logger import get_logger

class ProgressTracker:
    """Fixed progress tracker dengan granular step tracking"""
    
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.current_step = None
        self.step_start_time = None
    
    def progress(self, step: str, current: int, total: int, msg: str = ""):
        """Progress dengan step context tracking"""
        # Track step changes
        if step != self.current_step:
            self._log_step_change(step, msg)
            self.current_step = step
            import time
            self.step_start_time = time.time()
        
        # Report progress dengan communicator
        if self.comm and hasattr(self.comm, 'progress'):
            self.comm.progress(step, current, total, msg)
        
        # Backup progress via callback
        if self.comm and hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback(None, step, current, total, msg)
        
        # Log milestone progress
        if current in [0, total//4, total//2, 3*total//4, total]:
            percentage = int((current / max(total, 1)) * 100)
            self._log_milestone(step, percentage, msg)
    
    def _log_step_change(self, new_step: str, message: str):
        """Log step transitions untuk visibility"""
        step_emojis = {
            'overall': '🎯', 'step': '📊', 'current': '⚡',
            'analysis': '🔍', 'augmentation': '🔄', 'normalization': '🔧',
            'cleanup': '🧹', 'validation': '✅'
        }
        
        emoji = step_emojis.get(new_step, '📈')
        step_msg = f"{emoji} {new_step.title()} Phase: {message}"
        
        if self.comm:
            self.comm.log_info(step_msg)
        else:
            print(f"ℹ️ {step_msg}")
    
    def _log_milestone(self, step: str, percentage: int, message: str):
        """Log milestone progress untuk granular feedback"""
        if percentage in [0, 25, 50, 75, 100]:
            milestone_msg = f"📊 {step.title()}: {percentage}% - {message}"
            if self.comm:
                self.comm.log_info(milestone_msg)
    
    # Enhanced log methods dengan step context
    def log_info(self, msg: str):
        """Log info dengan step context"""
        context_msg = f"[{self.current_step.upper() if self.current_step else 'SYSTEM'}] {msg}"
        if self.comm:
            self.comm.log_info(context_msg)
        else:
            print(f"ℹ️ {context_msg}")
    
    def log_success(self, msg: str):
        """Log success dengan step context"""
        context_msg = f"[{self.current_step.upper() if self.current_step else 'SYSTEM'}] {msg}"
        if self.comm:
            self.comm.log_success(context_msg)
        else:
            print(f"✅ {context_msg}")
    
    def log_error(self, msg: str):
        """Log error dengan step context"""
        context_msg = f"[{self.current_step.upper() if self.current_step else 'SYSTEM'}] {msg}"
        if self.comm:
            self.comm.log_error(context_msg)
        else:
            print(f"❌ {context_msg}")
    
    def log_warning(self, msg: str):
        """Log warning dengan step context"""
        context_msg = f"[{self.current_step.upper() if self.current_step else 'SYSTEM'}] {msg}"
        if self.comm:
            self.comm.log_warning(context_msg)
        else:
            print(f"⚠️ {context_msg}")
    
    def log_debug(self, msg: str):
        """Log debug dengan step context"""
        context_msg = f"[{self.current_step.upper() if self.current_step else 'SYSTEM'}] {msg}"
        if self.comm and hasattr(self.comm, 'log_debug'):
            self.comm.log_debug(context_msg)
        else:
            print(f"🔍 {context_msg}")
    
    def finish_step(self, step: str, success: bool = True, message: str = ""):
        """Finish current step dengan summary"""
        if step == self.current_step and self.step_start_time:
            import time
            duration = time.time() - self.step_start_time
            
            status_emoji = "✅" if success else "❌"
            status_text = "completed" if success else "failed"
            
            summary_msg = f"{status_emoji} {step.title()} {status_text} in {duration:.1f}s"
            if message:
                summary_msg += f" - {message}"
            
            if self.comm:
                self.comm.log_success(summary_msg) if success else self.comm.log_error(summary_msg)
            
            self.current_step = None
            self.step_start_time = None

# Factory function dengan enhanced tracking
create_progress_tracker = lambda communicator=None: ProgressTracker(communicator)