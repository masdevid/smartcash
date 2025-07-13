"""
File: smartcash/ui/model/backbone/components/backbone_ui.py
Description: Backbone models UI following SmartCash standardized template.

This module provides the user interface for configuring backbone models
with config summary panel moved to summary_container as requested.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Backbone Configuration)
3. Action Container (Validate/Build/Load/Summary Buttons)
4. Summary Container (Configuration Summary Panel)
5. Operation Container (Progress + Logs)
6. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG, DEFAULT_CONFIG
from ..configs.backbone_defaults import get_available_backbones, get_detection_layers_config

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


@handle_ui_errors(error_component_title="Backbone Models UI Error")
def create_backbone_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the backbone models UI following SmartCash standards.
    
    This function creates a complete UI for backbone model configuration
    with the following sections:
    - Backbone selection and configuration options
    - Model parameters and optimization settings
    - Configuration summary panel (moved to summary_container)
    - Status and progress tracking
    - Detailed operation logging
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references
        
    Example:
        >>> ui = create_backbone_ui()
        >>> display(ui['main_container'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or DEFAULT_CONFIG.copy()
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to configure backbone model",
        status_type="info"
    )
    ui_components['header_container'] = header_container.container
    ui_components['containers']['header'] = header_container
    
    # === 2. Create Form Container ===
    form_fields = _create_backbone_form_fields(current_config)
    form_container = create_form_container(
        title="Backbone Configuration",
        fields=form_fields,
        layout_type=LayoutType.COLUMN,
        collapsible=False
    )
    ui_components['form_container'] = form_container['container']
    ui_components['containers']['form'] = form_container
    
    # === 3. Create Action Container ===
    action_buttons = []
    for button_id, button_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'id': button_id,
            'text': button_config['text'],
            'style': button_config['style'],
            'tooltip': button_config['tooltip'],
            'disabled': False
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        show_save_reset=True
    )
    ui_components['action_container'] = action_container
    ui_components['containers']['action'] = action_container
    
    # === 4. Create Summary Container (Config Summary Panel) ===
    summary_content = _generate_config_summary_content(current_config)
    summary_container = create_summary_container(
        title="Configuration Summary",
        theme="info"
    )
    # Set the content after creation
    if hasattr(summary_container, 'update_content'):
        summary_container.update_content(summary_content)
    ui_components['summary_container'] = summary_container
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        title="Backbone Operations",
        show_progress=True,
        show_logs=True,
        collapsible=True
    )
    ui_components['operation_container'] = operation_container  # Store full container object
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Create Footer Container ===
    footer_tips = [
        "💡 EfficientNet-B4 is recommended for higher accuracy",
        "⚡ CSPDarknet provides faster inference with lower memory usage", 
        "🔧 Feature optimization improves model performance for currency detection",
        "📊 Use Model Summary to analyze memory usage and parameters"
    ]
    
    footer_container = create_footer_container(
        tips=footer_tips,
        module_info={
            'name': UI_CONFIG['title'],
            'version': UI_CONFIG['version'],
            'description': UI_CONFIG['description']
        }
    )
    ui_components['footer_container'] = footer_container.container
    ui_components['containers']['footer'] = footer_container
    
    # === 7. Create Main Container ===
    container_layout = [
        ui_components['header_container'],
        ui_components['form_container'],
        ui_components['action_container'],
        ui_components['summary_container'],  # Config summary panel here
        ui_components['operation_container']['container'],  # Use the widget part for layout
        ui_components['footer_container']
    ]
    
    main_container = create_main_container(
        containers=container_layout,
        title=f"{UI_CONFIG['title']} - Configuration Interface"
    )
    ui_components['main_container'] = main_container
    
    # Add reference to main UI for display
    ui_components['ui'] = main_container
    
    return ui_components


def _create_backbone_form_fields(config: Dict[str, Any]) -> list:
    """Create form fields for backbone configuration."""
    backbone_config = config.get('backbone', {})
    available_backbones = get_available_backbones()
    
    # Get current values
    current_backbone = backbone_config.get('model_type', 'efficientnet_b4')
    current_pretrained = backbone_config.get('pretrained', True)
    current_feature_opt = backbone_config.get('feature_optimization', True)
    current_mixed_precision = backbone_config.get('mixed_precision', True)
    current_input_size = backbone_config.get('input_size', 640)
    current_num_classes = backbone_config.get('num_classes', 7)
    
    form_fields = [
        # Backbone Selection
        {
            'type': 'dropdown',
            'key': 'backbone_type',
            'label': 'Backbone Architecture',
            'options': [
                ('EfficientNet-B4 (Recommended)', 'efficientnet_b4'),
                ('CSPDarknet (YOLOv5 Default)', 'cspdarknet')
            ],
            'value': current_backbone,
            'description': 'Select the backbone architecture for feature extraction'
        },
        
        # Pretrained Models
        {
            'type': 'checkbox',
            'key': 'pretrained',
            'label': 'Use Pretrained Weights',
            'value': current_pretrained,
            'description': 'Load pretrained weights for better initialization'
        },
        
        # Feature Optimization
        {
            'type': 'checkbox',
            'key': 'feature_optimization',
            'label': 'Enable Feature Optimization',
            'value': current_feature_opt,
            'description': 'Apply feature optimization techniques for currency detection'
        },
        
        # Mixed Precision
        {
            'type': 'checkbox',
            'key': 'mixed_precision',
            'label': 'Mixed Precision Training',
            'value': current_mixed_precision,
            'description': 'Use FP16 mixed precision for faster training and inference'
        },
        
        # Input Size
        {
            'type': 'int_slider',
            'key': 'input_size',
            'label': 'Input Image Size',
            'value': current_input_size,
            'min': 320,
            'max': 1280,
            'step': 32,
            'description': 'Input image size (must be multiple of 32)'
        },
        
        # Number of Classes
        {
            'type': 'int_text',
            'key': 'num_classes',
            'label': 'Number of Classes',
            'value': current_num_classes,
            'description': 'Total number of detection classes (including background)'
        },
        
        # Detection Layers (Read-only info)
        {
            'type': 'html',
            'key': 'detection_info',
            'label': 'Detection Configuration',
            'value': '<div style="padding: 10px; background: #f5f5f5; border-radius: 4px;">'
                    '<strong>Detection Layers:</strong> Banknote Detection<br>'
                    '<strong>Layer Mode:</strong> Single Layer<br>'
                    '<strong>Primary Classes:</strong> Currency banknotes'
                    '</div>',
            'description': 'Current detection layer configuration'
        },
        
        # Advanced Options (Collapsible)
        {
            'type': 'accordion',
            'key': 'advanced_options',
            'label': 'Advanced Configuration',
            'description': 'Advanced backbone configuration options',
            'children': [
                {
                    'type': 'checkbox',
                    'key': 'early_training_enabled',
                    'label': 'Enable Early Training Pipeline',
                    'value': backbone_config.get('early_training', {}).get('enabled', True),
                    'description': 'Enable early training pipeline integration'
                },
                {
                    'type': 'checkbox', 
                    'key': 'validation_from_pretrained',
                    'label': 'Validate from Pretrained Model',
                    'value': backbone_config.get('early_training', {}).get('validation_from_pretrained', True),
                    'description': 'Perform validation using existing pretrained models'
                }
            ]
        }
    ]
    
    return form_fields


def _generate_config_summary_content(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate configuration summary content for the summary container."""
    backbone_config = config.get('backbone', {})
    model_config = config.get('model', {})
    available_backbones = get_available_backbones()
    
    # Get backbone information
    backbone_type = backbone_config.get('model_type', 'efficientnet_b4')
    backbone_info = available_backbones.get(backbone_type, {})
    
    # Generate summary sections
    summary_sections = {
        'Model Architecture': {
            'Backbone': backbone_info.get('display_name', backbone_type),
            'Description': backbone_info.get('description', 'N/A'),
            'Recommended': '✅ Yes' if backbone_info.get('recommended') else '⚠️ No',
            'Pretrained': '✅ Enabled' if backbone_config.get('pretrained') else '❌ Disabled'
        },
        'Configuration Settings': {
            'Input Size': f"{backbone_config.get('input_size', 640)}px",
            'Number of Classes': backbone_config.get('num_classes', 7),
            'Feature Optimization': '✅ Enabled' if backbone_config.get('feature_optimization') else '❌ Disabled',
            'Mixed Precision': '✅ Enabled' if backbone_config.get('mixed_precision') else '❌ Disabled'
        },
        'Performance Characteristics': {
            'Memory Usage': backbone_info.get('memory_usage', 'N/A'),
            'Inference Speed': backbone_info.get('inference_speed', 'N/A'),
            'Expected Accuracy': backbone_info.get('accuracy', 'N/A'),
            'Output Channels': str(backbone_info.get('output_channels', []))
        },
        'Training Pipeline': {
            'Early Training': '✅ Enabled' if backbone_config.get('early_training', {}).get('enabled') else '❌ Disabled',
            'Pretrained Validation': '✅ Enabled' if backbone_config.get('early_training', {}).get('validation_from_pretrained') else '❌ Disabled',
            'Detection Layers': ', '.join(backbone_config.get('detection_layers', ['banknote'])),
            'Layer Mode': backbone_config.get('layer_mode', 'single').title()
        }
    }
    
    return {
        'title': 'Backbone Configuration Summary',
        'sections': summary_sections,
        'style': 'info'  # Use info style for the summary
    }


def update_config_summary(summary_container: Any, config: Dict[str, Any]) -> None:
    """
    Update the configuration summary in the summary container.
    
    Args:
        summary_container: The summary container widget
        config: Updated configuration dictionary
    """
    try:
        if summary_container and hasattr(summary_container, 'update_content'):
            new_content = _generate_config_summary_content(config)
            summary_container.update_content(new_content)
    except Exception as e:
        # Silently handle update errors to avoid UI disruption
        pass


# ==================== HELPER FUNCTIONS ====================

def get_backbone_form_values(form_container: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract form values from backbone form container.
    
    Args:
        form_container: Form container dictionary
        
    Returns:
        Dictionary of form values
    """
    try:
        if form_container and 'get_form_values' in form_container:
            return form_container['get_form_values']()
        return {}
    except Exception:
        return {}


def update_backbone_form(form_container: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update backbone form with configuration values.
    
    Args:
        form_container: Form container dictionary
        config: Configuration dictionary
    """
    try:
        if form_container and 'update_from_config' in form_container:
            form_container['update_from_config'](config)
    except Exception:
        # Silently handle update errors
        pass