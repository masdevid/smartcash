"""
File: smartcash/ui/dataset/visualization/components/visualization_ui.py
Description: Dataset visualization UI following SmartCash standardized template.

This module provides the user interface for visualizing dataset statistics,
class distributions, and generating various charts for dataset analysis.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Visualization Options)
3. Action Container (Analyze/Refresh/Export/Compare Buttons)
4. Summary Container (Statistics Overview)
5. Operation Container (Progress + Logs)
6. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any, List
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.decorators import handle_ui_errors

# Module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


def create_data_card(title: str, content: widgets.Widget, width: str = "100%") -> widgets.VBox:
    """Create a styled card container for data visualization.
    
    Args:
        title: Title of the card
        content: Widget to be placed inside the card
        width: Width of the card (default: "100%")
        
    Returns:
        A VBox widget containing the card
    """
    card_header = widgets.HTML(
        value=f'<div style="padding: 10px; background-color: #f8f9fa; border-bottom: 1px solid #dee2e6; border-radius: 5px 5px 0 0; font-weight: bold;">{title}</div>',
        layout=widgets.Layout(width='100%')
    )
    
    card_content = widgets.VBox(
        [content],
        layout=widgets.Layout(
            padding='10px',
            border='1px solid #dee2e6',
            border_top='none',
            border_radius='0 0 5px 5px',
            width='100%',
            overflow='auto'
        )
    )
    
    return widgets.VBox(
        [card_header, card_content],
        layout=widgets.Layout(
            width=width,
            margin='0 0 15px 0',
            box_shadow='0 2px 4px rgba(0,0,0,0.1)'
        )
    )


@handle_ui_errors(error_component_title="Visualization UI Error")
def create_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the dataset visualization UI following SmartCash standards.
    
    This function creates a complete UI for visualizing dataset statistics
    with the following sections:
    - Chart type and data split selection
    - Analysis and export options
    - Statistical summaries
    - Interactive charts and visualizations
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references with 'ui_components' key
        
    Example:
        >>> ui = create_visualization_ui()
        >>> display(ui['ui'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or {}
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to analyze dataset",
        status_type="info"
    )
    # Store both the container object and its widget
    ui_components['containers']['header'] = {
        'container': header_container.container,
        'widget': header_container
    }
    
    # === 2. Create Form Container ===
    # Create form widgets with two-column layout
    form_widgets = _create_module_form_widgets(current_config)
    
    # Create form container with the widgets
    form_container = create_form_container(
        form_rows=form_widgets['form_rows'],
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px",
        layout_kwargs={
            'width': '100%',
            'max_width': '100%',
            'margin': '0',
            'padding': '0',
            'justify_content': 'flex-start',
            'align_items': 'flex-start'
        }
    )
    
    # Store references
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Create Action Container ===
    # Create action buttons from BUTTON_CONFIG
    action_buttons = []
    for button_id, btn_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'name': button_id,
            'label': btn_config['text'],
            'button_style': btn_config['style'],
            'tooltip': btn_config['tooltip'],
            'icon': 'chart-bar' if button_id == 'analyze' else 'refresh' if button_id == 'refresh' else 'download' if button_id == 'export' else 'compare'
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        title="📊 Visualization Actions",
        alignment="left"
    )
    
    # Store references
    ui_components['containers']['actions'] = action_container
    if hasattr(action_container, 'get'):
        for btn in action_buttons:
            ui_components['widgets'][f'{btn["name"]}_button'] = action_container.get(btn['name'])
    
    # === 4. Create Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    summary_container = create_summary_container(
        title="📋 Dataset Statistics",
        theme="info",
        icon="📊"
    )
    summary_container.set_content(summary_content)
    
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        show_progress=True,
        show_logs=True,
        log_module_name=UI_CONFIG['module_name'],
        log_height="200px",
        log_entry_style='compact',  # Ensure consistent hover behavior
        collapsible=True,
        collapsed=False
    )
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Create Footer Container ===
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        show_tips=True,
        show_version=True
    )
    # Store both the container object and its widget
    ui_components['containers']['footer'] = {
        'container': footer_container.container,
        'widget': footer_container
    }
    
    # === 7. Create Main Container ===
    # Prepare components for main container
    components = [
        # Header container (object with .container attribute)
        {'type': 'header', 'component': header_container.container, 'order': 0},
        # Form container (dictionary with 'container' key)
        {'type': 'form', 'component': form_container['container'], 'order': 1},
        # Action container (dictionary with 'container' key)
        {'type': 'action', 'component': action_container['container'], 'order': 2},
        # Summary container (object with .container attribute)
        {'type': 'summary', 'component': summary_container.container, 'order': 3},
        # Operation container (dictionary with 'container' key)
        {'type': 'operation', 'component': operation_container['container'], 'order': 4},
        # Footer container (object with .container attribute)
        {'type': 'footer', 'component': footer_container.container, 'order': 5}
    ]
    
    # Create main container with all components
    main_container = create_main_container(
        components=components,
        **kwargs
    )
    
    # Store main UI references
    ui_components['ui'] = main_container
    ui_components['main_container'] = main_container
    
    # Add all containers to the ui_components for easy access
    ui_components['containers']['main'] = main_container
    
    # Create the result dictionary with all components
    result = {
        'ui_components': ui_components,
        'ui': main_container,
        'main_container': main_container,
        'containers': ui_components['containers'],
        'widgets': ui_components['widgets']
    }
    
    # Add all components to the root for backward compatibility
    result.update(ui_components['containers'])
    result.update(ui_components['widgets'])
    
    # Add direct references to all containers for easier access
    for container_name, container in ui_components['containers'].items():
        result[container_name] = container
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets for visualization options with a two-column layout.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    from ..constants import CHART_TYPE_OPTIONS, DATA_SPLIT_OPTIONS, EXPORT_FORMAT_OPTIONS
    
    # Common layout for form elements
    dropdown_layout = widgets.Layout(
        width='90%',
        margin='5px 0',
        padding='5px 0'
    )
    
    checkbox_layout = widgets.Layout(
        width='100%',
        margin='8px 0',
        padding='5px 0'
    )
    
    # Chart type selection
    chart_type_dropdown = widgets.Dropdown(
        options=CHART_TYPE_OPTIONS,
        value=config.get('chart_type', 'bar'),
        description='Chart Type:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Data split selection
    data_split_dropdown = widgets.Dropdown(
        options=DATA_SPLIT_OPTIONS,
        value=config.get('data_split', 'all'),
        description='Data Split:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Export format selection
    export_format_dropdown = widgets.Dropdown(
        options=EXPORT_FORMAT_OPTIONS,
        value=config.get('export_format', 'png'),
        description='Export Format:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Checkbox options
    show_grid_checkbox = widgets.Checkbox(
        value=config.get('show_grid', True),
        description='Show Grid',
        layout=checkbox_layout
    )
    
    show_legend_checkbox = widgets.Checkbox(
        value=config.get('show_legend', True),
        description='Show Legend',
        layout=checkbox_layout
    )
    
    auto_refresh_checkbox = widgets.Checkbox(
        value=config.get('auto_refresh', False),
        description='Auto Refresh',
        layout=checkbox_layout
    )
    
    # Refresh interval
    refresh_interval_int = widgets.IntText(
        value=config.get('refresh_interval', 60),
        description='Refresh (s):',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='80%', margin='8px 0')
    )
    
    # Create form sections with two-column layout
    chart_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>📊 Chart Configuration</h4>"),
        chart_type_dropdown,
        data_split_dropdown,
        export_format_dropdown
    ], layout=widgets.Layout(width='48%', margin='0 1% 10px 0'))
    
    display_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>⚙️ Display</h4>"),
        show_grid_checkbox,
        show_legend_checkbox
    ], layout=widgets.Layout(width='48%', margin='0 0 10px 1%'))
    
    refresh_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>🔄 Refresh</h4>"),
        widgets.HBox([
            auto_refresh_checkbox,
            refresh_interval_int
        ], layout=widgets.Layout(width='100%', justify_content='space-between'))
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Combine all sections in a two-column layout
    form_content = widgets.VBox([
        widgets.HBox([chart_section, display_section], 
                    layout=widgets.Layout(justify_content='space-between')),
        refresh_section
    ])
    
    return {
        'form_rows': [[form_content]],  # Single row containing our custom layout
        'widgets': {
            'chart_type_dropdown': chart_type_dropdown,
            'data_split_dropdown': data_split_dropdown,
            'export_format_dropdown': export_format_dropdown,
            'show_grid_checkbox': show_grid_checkbox,
            'show_legend_checkbox': show_legend_checkbox,
            'auto_refresh_checkbox': auto_refresh_checkbox,
            'refresh_interval_int': refresh_interval_int
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> str:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HTML string containing the summary content
    """
    return """
    <div style="padding: 10px;">
        <h5>📊 Dataset Overview</h5>
        <p>Dataset statistics and visualizations will be displayed here after analysis.</p>
        <ul>
            <li>Total samples: <span id="total-samples">-</span></li>
            <li>Class distribution: <span id="class-distribution">-</span></li>
            <li>Data splits: <span id="data-splits">-</span></li>
            <li>Augmentation status: <span id="augmentation-status">-</span></li>
        </ul>
    </div>
    """


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    return widgets.HTML(
        value="""
        <div style="padding: 12px; background: #e3f2fd; border-radius: 4px; margin: 8px 0;">
            <h4 style="margin-top: 0; color: #0d47a1;">📊 Visualization Guide</h4>
            <p>This module helps you analyze and visualize your dataset statistics.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Select chart type and data split</li>
                <li>Configure display options</li>
                <li>Click 'Analyze Dataset' to generate visualizations</li>
                <li>Use 'Export Charts' to save results</li>
            </ol>
        </div>
        """
    )