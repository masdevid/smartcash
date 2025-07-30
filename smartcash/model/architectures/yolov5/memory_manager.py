"""
Memory Management for YOLOv5 Integration
Handles memory cleanup and optimization operations
"""

import gc
import torch
from smartcash.common.logger import SmartCashLogger


class YOLOv5MemoryManager:
    """
    Handles memory management operations for YOLOv5 integration
    """
    
    def __init__(self, logger=None):
        """
        Initialize memory manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
    
    def initial_cleanup(self):
        """
        Perform comprehensive memory cleanup before initialization.
        
        This ensures a clean memory state when starting the YOLOv5 integration,
        which is especially important for repeated model creation or in memory-constrained environments.
        """
        try:
            # Force garbage collection multiple times for thorough cleanup
            for _ in range(3):
                gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.debug("🧹 Cleared CUDA cache during initialization")
            
            # Clear MPS cache if available (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    self.logger.debug("🧹 Cleared MPS cache during initialization")
                except Exception:
                    # MPS cache clearing can sometimes fail, but it's not critical
                    pass
            
            self.logger.debug("🧹 Initial memory cleanup completed")
            
        except Exception as e:
            # Memory cleanup should never fail the initialization
            self.logger.warning(f"⚠️ Initial memory cleanup encountered an issue: {e}")
    
    def cleanup_after_model_creation(self):
        """
        Cleanup memory after model creation
        """
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("🧹 Post-model-creation memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"⚠️ Post-model-creation cleanup issue: {e}")