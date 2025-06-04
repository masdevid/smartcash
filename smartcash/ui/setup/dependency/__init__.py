"""
File: smartcash/ui/setup/dependency/__init__.py
Deskripsi: Public API untuk dependency installer module
"""

# Import main initializer
from smartcash.ui.setup.dependency.dependency_initializer import (
    initialize_dependency_ui,
    DependencyInstallerInitializer
)

# Export public API
__all__ = [
    'initialize_dependency_ui',
    'DependencyInstallerInitializer'
]

# Convenience aliases
initialize_ui = initialize_dependency_ui
create_dependency_ui = initialize_dependency_ui