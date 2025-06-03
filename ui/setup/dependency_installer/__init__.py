"""
File: smartcash/ui/setup/dependency_installer/__init__.py
Deskripsi: Public API untuk dependency installer module
"""

# Import main initializer
from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import (
    initialize_dependency_installer_ui,
    DependencyInstallerInitializer
)

# Export public API
__all__ = [
    'initialize_dependency_installer_ui',
    'DependencyInstallerInitializer'
]

# Convenience aliases
initialize_ui = initialize_dependency_installer_ui
create_dependency_installer_ui = initialize_dependency_installer_ui