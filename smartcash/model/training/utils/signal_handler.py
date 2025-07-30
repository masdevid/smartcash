#!/usr/bin/env python3
"""
Signal handling utilities for graceful training interruption.

This module provides proper signal handling to ensure clean shutdown
when training is interrupted with Ctrl+C.
"""

import signal
import sys
import threading
from typing import Optional, Callable
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class TrainingSignalHandler:
    """Handles graceful shutdown of training processes."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.cleanup_callbacks = []
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        self._interrupt_count = 0
        self._lock = threading.Lock()
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback to run on shutdown."""
        if callback not in self.cleanup_callbacks:
            self.cleanup_callbacks.append(callback)
    
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        # Store original handlers
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🔧 Signal handlers installed for graceful shutdown")
    
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if self.original_sigterm_handler:
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        
        logger.debug("🔧 Original signal handlers restored")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        with self._lock:
            self._interrupt_count += 1
            
            if self._interrupt_count == 1:
                logger.info("🛑 Graceful shutdown requested (Ctrl+C). Cleaning up...")
                logger.info("💡 Press Ctrl+C again to force immediate exit")
                self.shutdown_requested = True
                self._run_cleanup_callbacks()
                
            elif self._interrupt_count == 2:
                logger.warning("⚠️ Force shutdown requested. Attempting immediate cleanup...")
                self._force_cleanup()
                sys.exit(1)
                
            else:
                logger.error("🔥 Multiple interrupts detected. Force exiting!")
                self._emergency_exit()
    
    def _run_cleanup_callbacks(self):
        """Run all registered cleanup callbacks."""
        logger.info(f"🧹 Running {len(self.cleanup_callbacks)} cleanup callbacks...")
        
        for i, callback in enumerate(self.cleanup_callbacks):
            try:
                logger.debug(f"Running cleanup callback {i+1}/{len(self.cleanup_callbacks)}")
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback {i+1} failed: {e}")
        
        logger.info("✅ Cleanup callbacks completed")
    
    def _force_cleanup(self):
        """Force immediate cleanup of critical resources."""
        try:
            # Cleanup DataLoaders
            from smartcash.model.training.data_loader_factory import DataLoaderFactory
            DataLoaderFactory.cleanup_all()
            
            # PyTorch cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Thread cleanup
            main_thread = threading.main_thread()
            for thread in threading.enumerate():
                if thread != main_thread and thread.is_alive():
                    logger.debug(f"Waiting for thread: {thread.name}")
                    thread.join(timeout=1.0)
            
            logger.info("✅ Force cleanup completed")
            
        except Exception as e:
            logger.error(f"Force cleanup failed: {e}")
    
    def _emergency_exit(self):
        """Emergency exit without cleanup."""
        try:
            import os
            logger.error("💥 Emergency exit - terminating process")
            os._exit(1)
        except:
            sys.exit(1)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested


# Global signal handler instance
_global_handler: Optional[TrainingSignalHandler] = None


def get_signal_handler() -> TrainingSignalHandler:
    """Get the global signal handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = TrainingSignalHandler()
    return _global_handler


def install_training_signal_handlers():
    """Install signal handlers for training processes."""
    handler = get_signal_handler()
    handler.install_signal_handlers()
    return handler


def register_cleanup_callback(callback: Callable):
    """Register a cleanup callback for shutdown."""
    handler = get_signal_handler()
    handler.register_cleanup_callback(callback)


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    handler = get_signal_handler()
    return handler.is_shutdown_requested()


def cleanup_and_exit():
    """Manually trigger cleanup and exit."""
    handler = get_signal_handler()
    handler._run_cleanup_callbacks()
    handler.restore_signal_handlers()
    sys.exit(0)