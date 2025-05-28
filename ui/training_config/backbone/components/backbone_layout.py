"""
File: smartcash/ui/training_config/backbone/components/backbone_layout.py
Deskripsi: Layout arrangement untuk backbone configuration UI dengan responsive design
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container

def create_backbone_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create responsive layout untuk backbone configuration.
    
    Args:
        form_components: Dictionary berisi form components
        
    Returns:
        Dictionary berisi layout components
    """
    # Header dengan icon
    header = create_header(
        title="Konfigurasi Backbone Model",
        description="Pilih backbone dan konfigurasi optimasi untuk deteksi mata uang",
        icon="🧠"
    )
    
    # Model configuration section - one-liner responsive container
    model_section = create_responsive_container([
        widgets.HTML("<h4>🔧 Konfigurasi Model</h4>"),
        form_components['backbone_dropdown'],
        form_components['model_type_dropdown']
    ], padding="10px", margin="5px 0")
    
    # Optimization features section
    optimization_section = create_responsive_container([
        widgets.HTML("<h4>⚡ Fitur Optimasi</h4>"),
        form_components['use_attention_checkbox'],
        form_components['use_residual_checkbox'], 
        form_components['use_ciou_checkbox']
    ], padding="10px", margin="5px 0")
    
    # Control buttons section
    controls_section = create_responsive_container([
        form_components['save_reset_container']
    ], container_type="hbox", justify_content="flex-end", margin="10px 0")
    
    # Main container dengan responsive layout
    main_container = create_responsive_container([
        header,
        form_components['status_panel'],
        model_section,
        widgets.HTML("<hr style='margin: 10px 0; border-style: dashed;'>"),
        optimization_section,
        widgets.HTML("<hr style='margin: 10px 0; border-style: dashed;'>"),
        controls_section,
    ], padding="15px",width="100%")
    
    # Merge semua components
    layout_components = {
        'main_container': main_container,
        'header': header,
        'model_section': model_section,
        'optimization_section': optimization_section,
        'controls_section': controls_section,
        **form_components  # Include semua form components
    }
    
    return layout_components