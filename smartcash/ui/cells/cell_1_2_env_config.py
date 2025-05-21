"""
File: smartcash/ui/cells/cell_1_2_env_config.py
Deskripsi: Entry point untuk environment configuration cell
"""

# Import environment singleton first (initialize it before anything else)
from smartcash.common.environment import get_environment_manager
env_manager = get_environment_manager()

# Then import and display the environment configuration component
from smartcash.ui.components.env_config_component import EnvConfigComponent

# Create and display environment configuration
env_config = EnvConfigComponent()
env_config.display()