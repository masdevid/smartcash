"""
File: smartcash/ui/setup/env_config/utils/__init__.py

Utils Module - DEPRECATED

Utils functionality telah diintegrasikan dalam arsitektur baru:
- env_detector.py -> EnvironmentStatusChecker (services/status_checker.py)
- Other utils -> Base handlers dan operation handlers

Untuk compatibility, import dari services/status_checker.py
"""

# Deprecated imports untuk backward compatibility
from smartcash.ui.setup.env_config.services.status_checker import EnvironmentStatusChecker

# Deprecated alias
detect_environment_info = lambda: EnvironmentStatusChecker().get_overall_status()

__all__ = [
    'EnvironmentStatusChecker',
    'detect_environment_info'  # Deprecated - use EnvironmentStatusChecker instead
]