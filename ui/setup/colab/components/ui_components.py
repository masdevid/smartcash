"""
file_path: smartcash/ui/setup/colab/components/ui_components.py
Deskripsi: Komponen UI untuk environment configuration dengan layout lengkap
"""

from __future__ import annotations

import ipywidgets as widgets
from typing import Dict, Any

# Import shared UI components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

# Import local colab components
from smartcash.ui.setup.colab.components.setup_summary import create_setup_summary
from smartcash.ui.setup.colab.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.colab.components.tips_panel import create_tips_requirements

def create_colab_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi Colab
    
    Returns:
        Dictionary berisi semua komponen UI
    """
    # Initialize components dictionary
    ui_components: Dict[str, Any] = {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title=" Environment Setup",
        subtitle="Konfigurasi lingkungan untuk SmartCash YOLOv5-EfficientNet"
    )
    ui_components['header_container'] = header_container
    
    # 2. Create Action Container with setup button
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "setup",
                "text": " Setup Environment",
                "style": "primary",
                "order": 1
            }
        ]
    )
    ui_components['action_container'] = action_container
    
    # 3. Create Progress Tracker
    def create_progress_tracker(component_name: str = "progress_tracker") -> ProgressTracker:
        """
        Create a progress tracker component.

        Args:
            component_name: Unique name for the progress tracker component.

        Returns:
            ProgressTracker: Initialized progress tracker component.
        """
        return ProgressTracker(component_name=component_name)
    
    progress_tracker = create_progress_tracker(component_name="colab_progress_tracker")
    ui_components['progress_tracker'] = progress_tracker
    
    # 4. Create Environment Info Panel
    env_info_panel = create_env_info_panel()
    ui_components['env_info_panel'] = env_info_panel
    
    # 5. Create Tips Panel
    tips_panel = create_tips_requirements()
    ui_components['tips_panel'] = tips_panel
    
    # 6. Create Setup Summary
    setup_summary = create_setup_summary()
    ui_components['setup_summary'] = setup_summary
    
    # 7. Create Footer Container
    footer_container_instance = create_footer_container()
    ui_components['footer_container'] = footer_container_instance
    
    # 8. Create Main Container with all components
    main_container = create_main_container(
        children=[
            header_container,
            progress_tracker,
            env_info_panel,
            tips_panel,
            setup_summary,
            action_container,
            footer_container_instance
        ],
        layout={"width": "100%", "padding": "20px"}
    )
    ui_components['main_container'] = main_container
    
    return ui_components

# Backward compatibility alias for older code
create_env_config_ui = create_colab_ui
