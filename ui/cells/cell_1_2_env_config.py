"""
File: smartcash/ui/cells/cell_1_2_env_config.py
Deskripsi: Entry point untuk environment configuration cell
"""

def setup_env_config():
    """Setup dan tampilkan UI untuk konfigurasi environment."""
    from smartcash.ui.setup.env_config.components import EnvConfigComponent
    
    # Create and display the environment configuration component
    env_config = EnvConfigComponent()
    env_config.display()
    
    return env_config

# Eksekusi saat modul diimpor
ui_components = setup_env_config()