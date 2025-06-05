"""
File: smartcash/ui/backbone/components/ui_layout.py
Deskripsi: Layout responsive untuk backbone configuration menggunakan shared layout utilities
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column

def create_backbone_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive layout untuk backbone configuration dengan two-column design"""
    
    # Header dengan icon dan description
    header = create_header(
        title="Konfigurasi Backbone Model",
        description="Pilih arsitektur backbone dan optimasi untuk deteksi denominasi mata uang Rupiah",
        icon="🧠"
    )
    
    # Left column: Model Selection
    model_selection = create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 15px 0;'>🔧 Seleksi Model</h4>"),
        form_components['backbone_dropdown'],
        form_components['model_type_dropdown']
    ], padding="15px", margin="5px")
    
    # Right column: Feature Optimization
    feature_optimization = create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 15px 0;'>⚡ Optimasi Fitur</h4>"),
        form_components['use_attention_checkbox'],
        form_components['use_residual_checkbox'],
        form_components['use_ciou_checkbox']
    ], padding="15px", margin="5px")
    
    # Two-column layout untuk model selection dan feature optimization
    main_content = create_responsive_two_column(
        model_selection,
        feature_optimization,
        left_width="48%",
        right_width="48%",
        gap="4%"
    )
    
    # Controls section untuk save/reset buttons
    controls_section = create_responsive_container([
        form_components['save_reset_container']
    ], container_type="hbox", justify_content="flex-end", margin="15px 0")
    
    # Main container dengan semua elements
    main_container = create_responsive_container([
        header,
        form_components['status_panel'],
        widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
        main_content,
        widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
        controls_section
    ], padding="20px", width="100%")
    
    # Return all layout components
    return {
        'main_container': main_container,
        'ui': main_container,  # Alias untuk compatibility
        'header': header,
        'model_selection': model_selection,
        'feature_optimization': feature_optimization,
        'main_content': main_content,
        'controls_section': controls_section,
        **form_components  # Include all form components
    }