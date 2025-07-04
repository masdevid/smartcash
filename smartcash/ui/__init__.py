"""
File: smartcash/ui/__init__.py
Deskripsi: UI package exports
"""

# Import subpackages to make them available as smartcash.ui.*
from . import handlers
from . import setup
from . import types

# Initialize the initializers package
initializers = None
try:
    from . import initializers
    from .initializers import common_initializer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import initializers: {e}")

# Ensure setup package is properly imported
try:
    from . import setup
    from .setup import env_config
    from .setup import dependency
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import setup modules: {e}")

__all__ = [
    'handlers',
    'setup',
    'types',
    'initializers',
    'env_config',
    'dependency'
]
