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
                
                # Set a timeout for graceful cleanup to prevent hang-ups
                import threading
                cleanup_thread = threading.Thread(target=self._run_cleanup_callbacks_with_timeout)
                cleanup_thread.daemon = True
                cleanup_thread.start()
                
                # Wait for cleanup with timeout to prevent hang-ups
                cleanup_thread.join(timeout=5.0)  # 5 second timeout
                
                if cleanup_thread.is_alive():
                    logger.warning("⚠️ Graceful cleanup timed out, forcing exit")
                    self._force_cleanup()
                else:
                    logger.info("✅ Graceful cleanup completed")
                
                sys.exit(0)
                
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
    
    def _run_cleanup_callbacks_with_timeout(self):
        """Run cleanup callbacks with individual timeouts to prevent hang-ups."""
        logger.info(f"🧹 Running {len(self.cleanup_callbacks)} cleanup callbacks with timeout...")
        
        import threading
        import time
        
        for i, callback in enumerate(self.cleanup_callbacks):
            try:
                logger.debug(f"Running cleanup callback {i+1}/{len(self.cleanup_callbacks)}")
                
                # Run each callback in a separate thread with timeout
                callback_thread = threading.Thread(target=callback)
                callback_thread.daemon = True
                callback_thread.start()
                
                # Wait for callback with 2-second timeout per callback
                callback_thread.join(timeout=2.0)
                
                if callback_thread.is_alive():
                    logger.warning(f"Cleanup callback {i+1} timed out, continuing...")
                    # Don't wait for timed out callbacks - let daemon threads handle them
                else:
                    logger.debug(f"✅ Cleanup callback {i+1} completed successfully")
                    
            except Exception as e:
                logger.warning(f"Cleanup callback {i+1} failed: {e}")
        
        logger.info("✅ Cleanup callbacks processing completed")
    
    def _force_cleanup(self):
        """Force immediate cleanup of critical resources."""
        try:
            # Cleanup DataLoaders with timeout
            try:
                import signal
                signal.alarm(2)  # 2-second timeout for DataLoader cleanup
                from smartcash.model.training.data_loader_factory import DataLoaderFactory
                DataLoaderFactory.cleanup_all()
                signal.alarm(0)  # Cancel alarm
                logger.debug("✅ DataLoader cleanup completed")
            except Exception as e:
                logger.warning(f"DataLoader cleanup failed or timed out: {e}")
                signal.alarm(0)  # Cancel alarm
            
            # PyTorch cleanup with timeout
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.debug("✅ PyTorch cleanup completed")
            except Exception as e:
                logger.warning(f"PyTorch cleanup failed: {e}")
            
            # Thread cleanup with stricter timeout
            try:
                main_thread = threading.main_thread()
                active_threads = [t for t in threading.enumerate() if t != main_thread and t.is_alive() and not t.daemon]
                
                if active_threads:
                    logger.debug(f"Waiting for {len(active_threads)} non-daemon threads...")
                    for thread in active_threads:
                        logger.debug(f"Waiting for thread: {thread.name}")
                        thread.join(timeout=0.5)  # Reduced timeout to prevent hang-ups
                        
                        if thread.is_alive():
                            logger.warning(f"Thread {thread.name} did not terminate in time")
                
                logger.debug("✅ Thread cleanup completed")
            except Exception as e:
                logger.warning(f"Thread cleanup failed: {e}")
            
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