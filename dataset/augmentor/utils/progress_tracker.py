"""
File: smartcash/dataset/augmentor/utils/progress_tracker.py
Deskripsi: SRP module untuk progress tracking dengan real-time UI updates
"""

from smartcash.common.logger import get_logger

class ProgressTracker:
    """Fixed progress tracker dengan immediate UI updates"""
    
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
    
    def progress(self, step: str, current: int, total: int, msg: str = ""):
        """Fixed progress dengan immediate UI updates"""
        if self.comm and hasattr(self.comm, 'progress'):
            self.comm.progress(step, current, total, msg)
        
        # Backup progress via direct callback
        if self.comm and hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback(None, step, current, total, msg)
    
    # One-liner log methods
    log_info = lambda self, msg: self.comm.log_info(msg) if self.comm else print(f"ℹ️ {msg}")
    log_success = lambda self, msg: self.comm.log_success(msg) if self.comm else print(f"✅ {msg}")
    log_error = lambda self, msg: self.comm.log_error(msg) if self.comm else print(f"❌ {msg}")
    log_warning = lambda self, msg: self.comm.log_warning(msg) if self.comm else print(f"⚠️ {msg}")
    log_debug = lambda self, msg: self.comm.log_debug(msg) if self.comm else print(f"🔍 {msg}")

# Factory function
create_progress_tracker = lambda communicator=None: ProgressTracker(communicator)