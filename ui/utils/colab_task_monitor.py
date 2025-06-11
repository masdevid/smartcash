"""
File: smartcash/ui/utils/colab_task_monitor.py
Deskripsi: Colab-compatible task monitoring tanpa threading yang tidak didukung
"""

import time
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from IPython.display import display, Javascript

class ColabTaskMonitor:
    """Colab-compatible task monitor yang menghindari threading issues."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_tasks = {}
        self.logger = ui_components.get('logger')
    
    def submit_task(self, task_name: str, task_func: Callable, 
                   on_complete: Callable = None, on_error: Callable = None,
                   max_workers: int = 1) -> str:
        """Submit task dengan Colab-compatible monitoring."""
        
        # Create executor jika belum ada
        executor_key = f"{task_name}_executor"
        if executor_key not in self.ui_components or not self.ui_components[executor_key]:
            self.ui_components[executor_key] = ThreadPoolExecutor(max_workers=max_workers)
        
        executor = self.ui_components[executor_key]
        
        # Submit task
        future = executor.submit(task_func)
        future_key = f"{task_name}_future"
        self.ui_components[future_key] = future
        
        # Store task info
        self.active_tasks[task_name] = {
            'future': future,
            'executor': executor,
            'on_complete': on_complete,
            'on_error': on_error,
            'start_time': time.time(),
            'future_key': future_key,
            'executor_key': executor_key
        }
        
        # Start monitoring dengan simple polling
        self._start_monitoring(task_name)
        
        if self.logger:
            self.logger.debug(f"🚀 Task {task_name} submitted")
        
        return task_name
    
    def _start_monitoring(self, task_name: str):
        """Start monitoring dengan Colab-compatible approach."""
        
        def check_task():
            """Check task status dan handle completion."""
            if task_name not in self.active_tasks:
                return
            
            task_info = self.active_tasks[task_name]
            future = task_info['future']
            
            if future.done():
                # Task completed
                try:
                    result = future.result()
                    self._handle_task_complete(task_name, result)
                except Exception as e:
                    self._handle_task_error(task_name, e)
            else:
                # Task still running, schedule next check
                self._schedule_next_check(task_name)
        
        # Initial check
        check_task()
    
    def _schedule_next_check(self, task_name: str):
        """Schedule next check dengan Colab-compatible method."""
        
        # Method 1: Simple time-based approach (most reliable in Colab)
        def delayed_check():
            time.sleep(1)  # Wait 1 second
            if task_name in self.active_tasks:
                task_info = self.active_tasks[task_name]
                future = task_info['future']
                
                if future.done():
                    try:
                        result = future.result()
                        self._handle_task_complete(task_name, result)
                    except Exception as e:
                        self._handle_task_error(task_name, e)
                else:
                    self._schedule_next_check(task_name)  # Continue monitoring
        
        # Submit delayed check sebagai task baru
        try:
            # Use a minimal executor untuk monitoring
            monitor_executor = ThreadPoolExecutor(max_workers=1)
            monitor_executor.submit(delayed_check)
        except Exception:
            # Fallback: direct call dengan sleep
            time.sleep(1)
            if task_name in self.active_tasks and not self.active_tasks[task_name]['future'].done():
                self._schedule_next_check(task_name)
    
    def _handle_task_complete(self, task_name: str, result):
        """Handle task completion."""
        if task_name not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_name]
        
        try:
            # Call completion callback
            if task_info['on_complete']:
                task_info['on_complete'](result)
            
            elapsed_time = time.time() - task_info['start_time']
            if self.logger:
                self.logger.debug(f"✅ Task {task_name} completed in {elapsed_time:.1f}s")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error in completion callback for {task_name}: {str(e)}")
        
        finally:
            # Cleanup task
            self._cleanup_task(task_name)
    
    def _handle_task_error(self, task_name: str, error):
        """Handle task error."""
        if task_name not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_name]
        
        try:
            # Call error callback
            if task_info['on_error']:
                task_info['on_error'](error)
            
            if self.logger:
                self.logger.error(f"❌ Task {task_name} failed: {str(error)}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error in error callback for {task_name}: {str(e)}")
        
        finally:
            # Cleanup task
            self._cleanup_task(task_name)
    
    def _cleanup_task(self, task_name: str):
        """Cleanup task resources."""
        if task_name not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_name]
        
        try:
            # Shutdown executor
            executor = task_info['executor']
            if executor:
                executor.shutdown(wait=False)
            
            # Clear from ui_components
            future_key = task_info['future_key']
            executor_key = task_info['executor_key']
            
            if future_key in self.ui_components:
                del self.ui_components[future_key]
            if executor_key in self.ui_components:
                del self.ui_components[executor_key]
        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"🔧 Cleanup warning for {task_name}: {str(e)}")
        
        finally:
            # Remove from active tasks
            del self.active_tasks[task_name]
    
    def cancel_task(self, task_name: str) -> bool:
        """Cancel running task."""
        if task_name not in self.active_tasks:
            return False
        
        task_info = self.active_tasks[task_name]
        future = task_info['future']
        
        try:
            cancelled = future.cancel()
            if cancelled:
                self._cleanup_task(task_name)
                if self.logger:
                    self.logger.info(f"🛑 Task {task_name} cancelled")
            return cancelled
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error cancelling task {task_name}: {str(e)}")
            return False
    
    def get_running_tasks(self) -> list:
        """Get list of currently running tasks."""
        return list(self.active_tasks.keys())
    
    def cleanup_all_tasks(self):
        """Cleanup all running tasks."""
        for task_name in list(self.active_tasks.keys()):
            try:
                self.cancel_task(task_name)
            except Exception:
                pass

def get_task_monitor(ui_components: Dict[str, Any]) -> ColabTaskMonitor:
    """Factory function untuk mendapatkan task monitor."""
    if 'task_monitor' not in ui_components:
        ui_components['task_monitor'] = ColabTaskMonitor(ui_components)
    return ui_components['task_monitor']