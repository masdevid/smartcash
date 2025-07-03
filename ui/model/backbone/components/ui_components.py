"""
File: smartcash/ui/model/backbone/components/ui_components.py
Deskripsi: UI components untuk backbone model configuration dengan shared container components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import model-specific components
from smartcash.ui.model.backbone.components.model_form import create_model_form
from smartcash.ui.model.backbone.components.config_summary import create_config_summary

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components import create_save_reset_buttons
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

def create_backbone_child_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create child components untuk backbone configuration dengan shared container components
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of child components
    """
    config = config or {}
    child_components = {}
    
    # === 1. HEADER CONTAINER ===
    
    # Create header container
    header_container = create_header_container(
        title="Model Configuration",
        subtitle="Konfigurasi backbone model YOLOv5 dengan EfficientNet-B4",
        icon="🤖"
    )
    child_components['header_container'] = header_container.container
    
    # === 2. FORM CONTAINER WITH TWO-COLUMN LAYOUT ===
    
    # Create form container to hold the two-column layout
    form_container = create_form_container()
    
    # Model form (left column)
    model_form = create_model_form(config)
    child_components['model_form'] = model_form
    
    # Config summary (right column)
    config_summary = create_config_summary(config)
    child_components['config_summary'] = config_summary
    
    # Create two-column layout with optimized spacing
    two_column_layout = widgets.HBox([
        widgets.Box(
            [model_form],
            layout=widgets.Layout(width='65%', padding='0 5px 0 0')
        ),
        widgets.Box(
            [config_summary],
            layout=widgets.Layout(width='35%', padding='0 0 0 5px')
        )
    ], layout=widgets.Layout(
        display='flex',
        gap='10px',
        width='100%',
        align_items='flex-start',
        margin='0',
        padding='0'
    ))
    
    # Place two-column layout in the form container
    form_container['form_container'].children = (two_column_layout,)
    child_components['form_container'] = form_container['container']
    
    # Store the two-column layout for reference
    child_components['two_column_layout'] = two_column_layout
    
    # === 2. CONFIG BUTTONS SECTION ===
    
    # Create save/reset buttons with standard approach
    save_reset_components = create_save_reset_buttons(
        save_label="💾 Simpan",
        reset_label="🔄 Reset", 
        with_sync_info=True
    )
    child_components['save_reset_buttons'] = save_reset_components
    
    # Create config buttons container with proper styling
    config_buttons_container = widgets.Box(
        [save_reset_components['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='8px 0')
    )
    child_components['config_buttons_container'] = config_buttons_container
    
    # === 3. ACTION CONTAINER ===
    
    # Create action container with standard button configuration
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "build",
                "text": "📝️ Bangun Model",
                "style": "primary",
                "order": 1,
                "tooltip": "Membangun model dengan konfigurasi yang dipilih"
            },
            {
                "button_id": "validate",
                "text": "📊 Validasi Konfigurasi",
                "style": "info",
                "order": 2,
                "tooltip": "Memvalidasi konfigurasi model sebelum membangun"
            },
            {
                "button_id": "info",
                "text": "🔍 Info Model",
                "style": "warning",
                "order": 3,
                "tooltip": "Menampilkan informasi detail tentang model"
            }
        ],
        title="🚀 Model Operations",
        alignment="left",
        with_confirmation=True
    )
    child_components['action_container'] = action_container.container
    
    # === 4. FOOTER CONTAINER ===
    
    # Create progress tracker
    progress_tracker = ProgressTracker(
        title="Model Building",
        levels=[ProgressLevel.OVERALL, ProgressLevel.CURRENT],
        auto_hide=True
    )
    child_components['progress_tracker'] = progress_tracker
    
    # Create footer container with log accordion and info box
    footer_container = create_footer_container(
        log_output=widgets.Output(),
        info_box=widgets.HTML(
            """
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>Tips Model Backbone:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Pilih backbone yang sesuai dengan kebutuhan deteksi</li>
                    <li>Layer mode yang tepat dapat meningkatkan performa model</li>
                    <li>Feature optimization membantu pada perangkat dengan memori terbatas</li>
                </ul>
            </div>
            """
        )
    )
    child_components['footer_container'] = footer_container.container
    
    # === 5. EXTRACT INDIVIDUAL COMPONENTS ===
    
    # Extract form widgets dari model_form for direct access
    if hasattr(model_form, 'backbone_dropdown'):
        child_components['backbone_dropdown'] = model_form.backbone_dropdown
    if hasattr(model_form, 'detection_layers_select'):
        child_components['detection_layers_select'] = model_form.detection_layers_select
    if hasattr(model_form, 'layer_mode_dropdown'):
        child_components['layer_mode_dropdown'] = model_form.layer_mode_dropdown
    if hasattr(model_form, 'feature_optimization_checkbox'):
        child_components['feature_optimization_checkbox'] = model_form.feature_optimization_checkbox
    if hasattr(model_form, 'mixed_precision_checkbox'):
        child_components['mixed_precision_checkbox'] = model_form.mixed_precision_checkbox
    
    # Extract buttons with standard approach
    child_components['build_btn'] = action_container.get_button('build')
    child_components['validate_btn'] = action_container.get_button('validate')
    child_components['info_btn'] = action_container.get_button('info')
    child_components['save_button'] = save_reset_components.get('save_button')
    child_components['reset_button'] = save_reset_components.get('reset_button')
    
    # === 6. ASSEMBLE MAIN CONTAINER ===
    
    # Create main container with all components
    main_container = create_main_container(
        header_container=child_components['header_container'],
        form_container=child_components['form_container'],
        footer_container=child_components['footer_container'],
        additional_components=[
            config_buttons_container,
            child_components['action_container'],
            progress_tracker.container
        ]
    )
    child_components['main_container'] = main_container
    
    return child_components

def get_layout_sections(child_components: Dict[str, Any]) -> list:
    """Get ordered layout sections untuk main container
    
    Args:
        child_components: Dictionary of child components
        
    Returns:
        List of widgets in display order
    """
    # With the new shared container approach, we only need to return the main container
    if 'main_container' in child_components:
        return [child_components['main_container']]
    
    # Fallback to legacy approach if main_container is not available
    sections = []
    
    # Add form container if available
    if 'form_container' in child_components:
        sections.append(child_components['form_container'])
    
    # Add config buttons container if available
    if 'config_buttons_container' in child_components:
        sections.append(child_components['config_buttons_container'])
    
    # Add action container if available
    if 'action_container' in child_components:
        sections.append(child_components['action_container'])
    
    # Add footer container if available
    if 'footer_container' in child_components:
        sections.append(child_components['footer_container'])
    
    return sections