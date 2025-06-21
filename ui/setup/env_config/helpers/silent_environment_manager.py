"""
File: smartcash/ui/setup/env_config/helpers/silent_environment_manager.py
Deskripsi: Wrapper untuk environment manager yang mencegah log leakage ke console
"""

import sys
import io
import contextlib
from typing import Dict, Any, Optional

class SilentEnvironmentManager:
    """Wrapper untuk environment manager yang mencegah output ke console"""
    
    def __init__(self):
        self._env_manager = None
        self._initialized = False
    
    @contextlib.contextmanager
    def _suppress_output(self):
        """Context manager untuk suppress semua output ke stdout/stderr"""
        # Save original streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # Redirect ke null
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _get_environment_manager(self):
        """Get environment manager dengan complete output suppression"""
        if self._env_manager is None and not self._initialized:
            try:
                with self._suppress_output():
                    from smartcash.common.environment import get_environment_manager
                    self._env_manager = get_environment_manager()
                self._initialized = True
            except Exception:
                self._initialized = True  # Mark as attempted
                self._env_manager = None
        
        return self._env_manager
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system info tanpa output ke console"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return self._get_fallback_system_info()
        
        try:
            with self._suppress_output():
                return env_manager.get_system_info()
        except Exception:
            return self._get_fallback_system_info()
    
    def refresh_drive_status(self) -> bool:
        """Refresh Drive status secara silent"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return False
        
        try:
            with self._suppress_output():
                return env_manager.refresh_drive_status()
        except Exception:
            return False
    
    def is_colab(self) -> bool:
        """Check apakah Colab environment secara silent"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return self._detect_colab_fallback()
        
        try:
            with self._suppress_output():
                return env_manager.is_colab
        except Exception:
            return self._detect_colab_fallback()
    
    def is_drive_mounted(self) -> bool:
        """Check Drive mount status secara silent"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return self._check_drive_fallback()
        
        try:
            with self._suppress_output():
                return env_manager.is_drive_mounted
        except Exception:
            return self._check_drive_fallback()
    
    def get_drive_path(self) -> Optional[str]:
        """Get Drive path secara silent"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return self._get_drive_path_fallback()
        
        try:
            with self._suppress_output():
                return str(env_manager.drive_path) if env_manager.drive_path else None
        except Exception:
            return self._get_drive_path_fallback()
    
    def mount_drive(self) -> tuple:
        """Mount Drive dengan output suppression"""
        env_manager = self._get_environment_manager()
        if env_manager is None:
            return False, "Environment manager tidak tersedia"
        
        try:
            with self._suppress_output():
                return env_manager.mount_drive()
        except Exception as e:
            return False, f"Mount error: {str(e)}"
    
    def _get_fallback_system_info(self) -> Dict[str, Any]:
        """Fallback system info tanpa dependencies external"""
        import os
        import sys
        from pathlib import Path
        
        info = {
            'environment': 'Google Colab' if self._detect_colab_fallback() else 'Local',
            'python_version': sys.version.split()[0],
            'base_directory': '/content' if self._detect_colab_fallback() else str(Path.cwd()),
            'cuda_available': False
        }
        
        # Basic GPU check
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Basic memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = round(memory.total / (1024**3), 2)
            info['available_memory_gb'] = round(memory.available / (1024**3), 2)
        except ImportError:
            pass
        
        return info
    
    def _detect_colab_fallback(self) -> bool:
        """Fallback Colab detection"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _check_drive_fallback(self) -> bool:
        """Fallback Drive check"""
        from pathlib import Path
        return Path('/content/drive/MyDrive').exists()
    
    def _get_drive_path_fallback(self) -> Optional[str]:
        """Fallback Drive path"""
        from pathlib import Path
        if self._check_drive_fallback():
            return '/content/drive/MyDrive/SmartCash'
        return None

# Singleton instance
_silent_env_manager = None

def get_silent_environment_manager() -> SilentEnvironmentManager:
    """Get singleton silent environment manager"""
    global _silent_env_manager
    if _silent_env_manager is None:
        _silent_env_manager = SilentEnvironmentManager()
    return _silent_env_manager