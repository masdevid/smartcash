"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Action buttons modern dengan design responsif, flexible, dan konfigurasi mudah
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import ipywidgets as widgets

class ActionButtonConfig:
    """Konfigurasi button dengan validasi dan defaults"""
    
    def __init__(self, label: Union[str, tuple], icon: str = "", style: str = "primary", 
                 tooltip: str = "", width: str = "auto", variant: str = "contained"):
        # Handle case where label might be a tuple
        if isinstance(label, (tuple, list)) and len(label) > 0:
            self.label = str(label[0]) if label else "Button"
        else:
            self.label = str(label) if label is not None else "Button"
            
        self.icon = self._validate_icon(icon)
        self.style = self._validate_style(style)
        self.tooltip = tooltip or f"Klik untuk {self.label.lower()}"
        self.width = width
        self.variant = variant
    
    def _validate_icon(self, icon: str) -> str:
        """Validasi icon dengan fallback ke icon yang tersedia"""
        valid_icons = ['play', 'pause', 'stop', 'download', 'upload', 'save', 'search', 
                      'check', 'times', 'trash', 'edit', 'copy', 'refresh', 'settings']
        return icon if icon in valid_icons else ""
    
    def _validate_style(self, style: str) -> str:
        """Validasi style dengan fallback ke primary"""
        valid_styles = ['primary', 'success', 'info', 'warning', 'danger', '']
        return style if style in valid_styles else 'primary'

def create_action_buttons(
    primary_button: Union[str, Dict[str, Any], ActionButtonConfig] = None,
    secondary_buttons: List[Union[str, Dict[str, Any], ActionButtonConfig]] = None,
    layout_style: str = "flex",  # flex, grid, stack
    responsive_breakpoint: str = "medium",  # small, medium, large
    container_style: Dict[str, str] = None,
    button_spacing: str = "8px",
    auto_width: bool = True,
    max_buttons_per_row: int = 4
) -> Dict[str, Any]:
    """
    🎨 Create modern, responsive action buttons dengan konfigurasi flexible
    
    Args:
        primary_button: Button utama dengan format:
            - str: "🚀 Process" 
            - dict: {"label": "🚀 Process", "icon": "play", "style": "primary"}
            - tuple: ("🚀 Process", "play", "primary")
            - ActionButtonConfig: ActionButtonConfig("🚀 Process", "play", "primary")
        secondary_buttons: List button sekunder dengan format sama seperti primary_button
        layout_style: Style layout container
            - "flex": horizontal layout dengan wrap (default)
            - "grid": grid layout responsif
            - "stack": vertical stack layout
        responsive_breakpoint: Breakpoint untuk responsif ("small", "medium", "large")
        container_style: Custom CSS styling untuk container
        button_spacing: Space antar button (default: "8px")
        auto_width: Auto-calculate optimal button width (default: True)
        max_buttons_per_row: Maksimal button per baris untuk grid layout (default: 4)
        
    Returns:
        Dict berisi:
            - 'container': Widget container utama
            - 'buttons': List semua button widgets
            - 'primary': Primary button widget (jika ada)
            - 'secondary_0', 'secondary_1', etc: Secondary button widgets
            - '{label_key}': Button dengan key dari sanitized label
            - 'count': Jumlah total buttons
            - Methods: add_button(), remove_button(), update_layout()
    
    Examples:
        # Simple usage
        >>> buttons = create_action_buttons("🚀 Process", ["🔍 Check", "🧹 Clean"])
        >>> display(buttons['container'])
        
        # Advanced configuration  
        >>> buttons = create_action_buttons(
        ...     primary_button={"label": "🚀 Start", "icon": "play", "style": "primary"},
        ...     secondary_buttons=[
        ...         ("🔍 Validate", "check", "success"),
        ...         {"label": "🧹 Cleanup", "icon": "trash", "style": "warning"}
        ...     ],
        ...     layout_style="grid",
        ...     max_buttons_per_row=3
        ... )
        
        # Access individual buttons
        >>> start_btn = buttons['start']  # Dari sanitized label
        >>> validate_btn = buttons['secondary_0']  # Dari index
        >>> cleanup_btn = buttons['cleanup']  # Dari sanitized label
        
        # Dynamic management
        >>> buttons['container'].add_button("📊 Analyze")
        >>> buttons['container'].remove_button("cleanup")
        
        # Button state management
        >>> disable_all_buttons(buttons)
        >>> enable_all_buttons(buttons)
        
        # Preset configurations
        >>> preprocessing_btns = create_preprocessing_buttons(cleanup_enabled=True)
        >>> training_btns = create_training_buttons()
    """
    
    # === BUTTON CREATION ===
    
    buttons = []
    button_refs = {}
    
    # Primary button
    if primary_button:
        btn_config = _parse_button_config(primary_button, is_primary=True)
        primary_btn = _create_single_button(btn_config)
        buttons.append(primary_btn)
        button_refs['primary'] = primary_btn
        button_refs[_sanitize_key(btn_config.label)] = primary_btn
    
    # Secondary buttons
    if secondary_buttons:
        for i, btn_data in enumerate(secondary_buttons):
            btn_config = _parse_button_config(btn_data, is_primary=False)
            secondary_btn = _create_single_button(btn_config)
            buttons.append(secondary_btn)
            button_refs[f'secondary_{i}'] = secondary_btn
            button_refs[_sanitize_key(btn_config.label)] = secondary_btn
    
    # === RESPONSIVE LAYOUT ===
    
    container_layout = _create_responsive_layout(
        layout_style=layout_style,
        responsive_breakpoint=responsive_breakpoint,
        button_spacing=button_spacing,
        max_buttons_per_row=max_buttons_per_row,
        button_count=len(buttons)
    )
    
    # Auto-width calculation
    if auto_width and buttons:
        button_width = _calculate_optimal_width(len(buttons), max_buttons_per_row)
        for btn in buttons:
            btn.layout.width = button_width
    
    # === CONTAINER ASSEMBLY ===
    
    # Merge custom styles
    if container_style:
        container_layout.update(container_style)
    
    # Create container berdasarkan layout style
    if layout_style == "stack":
        container = widgets.VBox(buttons, layout=widgets.Layout(**container_layout))
    else:
        container = widgets.HBox(buttons, layout=widgets.Layout(**container_layout))
    
    # === RESULT ASSEMBLY ===
    
    result = {
        'container': container,
        'buttons': buttons,
        'count': len(buttons),
        'layout_style': layout_style,
        **button_refs  # Spread individual button references
    }
    
    # Add utility methods to container
    container.add_button = lambda btn_config: _add_button_to_container(container, btn_config, result)
    container.remove_button = lambda label: _remove_button_from_container(container, label, result)
    container.update_layout = lambda new_style: _update_container_layout(container, new_style)
    
    return result

def _parse_button_config(btn_data: Union[str, Dict[str, Any], ActionButtonConfig], 
                        is_primary: bool = False) -> ActionButtonConfig:
    """Parse button data menjadi ActionButtonConfig"""
    
    if isinstance(btn_data, ActionButtonConfig):
        return btn_data
    
    if isinstance(btn_data, str):
        # Simple string label
        style = "primary" if is_primary else "info"
        return ActionButtonConfig(label=btn_data, style=style)
    
    if isinstance(btn_data, (tuple, list)) and len(btn_data) >= 2:
        # Tuple format: (label, icon, style)
        label, icon = btn_data[0], btn_data[1]
        style = btn_data[2] if len(btn_data) > 2 else ("primary" if is_primary else "info")
        return ActionButtonConfig(label=label, icon=icon, style=style)
    
    if isinstance(btn_data, dict):
        # Dictionary format
        return ActionButtonConfig(
            label=btn_data.get('label', 'Button'),
            icon=btn_data.get('icon', ''),
            style=btn_data.get('style', 'primary' if is_primary else 'info'),
            tooltip=btn_data.get('tooltip', ''),
            width=btn_data.get('width', 'auto'),
            variant=btn_data.get('variant', 'contained')
        )
    
    # Fallback
    return ActionButtonConfig(label="Button", style="primary" if is_primary else "info")

def _create_single_button(config: ActionButtonConfig) -> widgets.Button:
    """Create individual button dengan modern styling"""
    
    # Base button properties
    button_props = {
        'description': config.label,
        'button_style': config.style,
        'tooltip': config.tooltip,
        'layout': widgets.Layout(
            width=config.width,
            height='36px',
            margin='0',
            border_radius='6px',
            font_weight='500'
        )
    }
    
    # Add icon if provided
    if config.icon:
        button_props['icon'] = config.icon
    
    # Create button
    button = widgets.Button(**button_props)
    
    # Store original config for reference
    setattr(button, '_config', config)
    setattr(button, '_original_style', config.style)
    setattr(button, '_original_description', config.label)
    
    # Add modern CSS classes (for future styling)
    button.add_class(f'btn-{config.variant}')
    button.add_class(f'btn-{config.style}')
    
    return button

def _create_responsive_layout(layout_style: str, responsive_breakpoint: str, 
                            button_spacing: str, max_buttons_per_row: int,
                            button_count: int) -> Dict[str, str]:
    """Create responsive layout configuration"""
    
    # Base layout
    base_layout = {
        'width': '100%',
        'margin': '8px 0',
        'align_items': 'center',
        'gap': button_spacing
    }
    
    # Layout-specific configurations
    if layout_style == "flex":
        base_layout.update({
            'display': 'flex',
            'flex_flow': 'row wrap',
            'justify_content': 'flex-start'
        })
    elif layout_style == "grid":
        cols = min(button_count, max_buttons_per_row)
        base_layout.update({
            'display': 'grid',
            'grid_template_columns': f'repeat({cols}, 1fr)',
            'grid_gap': button_spacing
        })
    elif layout_style == "stack":
        base_layout.update({
            'display': 'flex',
            'flex_direction': 'column',
            'align_items': 'stretch'
        })
    
    # Responsive breakpoints
    breakpoint_configs = {
        'small': {'max_width': '600px'},
        'medium': {'max_width': '900px'},
        'large': {'max_width': '1280px'}
    }
    
    if responsive_breakpoint in breakpoint_configs:
        base_layout.update(breakpoint_configs[responsive_breakpoint])
    
    return base_layout

def _calculate_optimal_width(button_count: int, max_per_row: int) -> str:
    """Calculate optimal button width berdasarkan jumlah button"""
    
    if button_count <= 1:
        return '200px'
    elif button_count <= 2:
        return '180px'
    elif button_count <= 3:
        return '160px'
    elif button_count <= max_per_row:
        return '140px'
    else:
        return '120px'

def _sanitize_key(label: str) -> str:
    """Sanitize label untuk dijadikan key yang valid"""
    return label.lower().replace(' ', '_').replace('-', '_')

def _add_button_to_container(container: widgets.Widget, btn_config: Union[str, Dict[str, Any]], 
                           result: Dict[str, Any]) -> None:
    """Add button ke container yang sudah ada"""
    
    config = _parse_button_config(btn_config)
    new_button = _create_single_button(config)
    
    # Add to container
    container.children = list(container.children) + [new_button]
    
    # Update result references
    result['buttons'].append(new_button)
    result['count'] = len(result['buttons'])
    result[_sanitize_key(config.label)] = new_button

def _remove_button_from_container(container: widgets.Widget, label: str, 
                                result: Dict[str, Any]) -> bool:
    """Remove button dari container berdasarkan label"""
    
    sanitized_key = _sanitize_key(label)
    
    if sanitized_key in result:
        button_to_remove = result[sanitized_key]
        
        # Remove from container
        new_children = [child for child in container.children if child != button_to_remove]
        container.children = new_children
        
        # Update result references
        result['buttons'] = [btn for btn in result['buttons'] if btn != button_to_remove]
        result['count'] = len(result['buttons'])
        del result[sanitized_key]
        
        return True
    
    return False

def _update_container_layout(container: widgets.Widget, new_style: str) -> None:
    """Update container layout style"""
    
    if new_style == "stack":
        container.layout.flex_direction = 'column'
        container.layout.align_items = 'stretch'
    else:
        container.layout.flex_direction = 'row'
        container.layout.align_items = 'center'

# === PRESET CONFIGURATIONS ===

def create_preprocessing_buttons(cleanup_enabled: bool = True) -> Dict[str, Any]:
    """🔧 Preset untuk preprocessing operations"""
    
    buttons = [
        ActionButtonConfig("🚀 Mulai Preprocessing", "play", "primary"),
        ActionButtonConfig("🔍 Check Dataset", "search", "info")
    ]
    
    if cleanup_enabled:
        buttons.append(ActionButtonConfig("🧹 Cleanup", "trash", "warning"))
    
    return create_action_buttons(
        primary_button=buttons[0],
        secondary_buttons=buttons[1:],
        layout_style="flex",
        max_buttons_per_row=3
    )

def create_dataset_buttons() -> Dict[str, Any]:
    """📊 Preset untuk dataset operations"""
    
    return create_action_buttons(
        primary_button=ActionButtonConfig("📥 Download Dataset", "download", "primary"),
        secondary_buttons=[
            ActionButtonConfig("🔍 Validate", "check", "success"),
            ActionButtonConfig("📊 Analyze", "search", "info"),
            ActionButtonConfig("🧹 Clean", "trash", "warning")
        ],
        layout_style="flex",
        max_buttons_per_row=4
    )

def create_training_buttons() -> Dict[str, Any]:
    """🚀 Preset untuk training operations"""
    
    return create_action_buttons(
        primary_button=ActionButtonConfig("🚀 Start Training", "play", "primary"),
        secondary_buttons=[
            ActionButtonConfig("⏸️ Pause", "pause", "warning"),
            ActionButtonConfig("⏹️ Stop", "stop", "danger"),
            ActionButtonConfig("📊 Monitor", "search", "info")
        ],
        layout_style="flex",
        max_buttons_per_row=4
    )

# === UTILITIES ===

def get_button_by_label(action_buttons: Dict[str, Any], label: str) -> Optional[widgets.Button]:
    """Get button berdasarkan label"""
    sanitized_key = _sanitize_key(label)
    return action_buttons.get(sanitized_key)

def disable_all_buttons(action_buttons: Dict[str, Any]) -> None:
    """Disable semua buttons dalam action_buttons"""
    for button in action_buttons.get('buttons', []):
        if hasattr(button, 'disabled'):
            button.disabled = True

def enable_all_buttons(action_buttons: Dict[str, Any]) -> None:
    """Enable semua buttons dalam action_buttons"""
    for button in action_buttons.get('buttons', []):
        if hasattr(button, 'disabled'):
            button.disabled = False

def update_button_style(button: widgets.Button, new_style: str, new_label: str = None) -> None:
    """Update button style dan label"""
    if hasattr(button, 'button_style'):
        button.button_style = new_style
    
    if new_label and hasattr(button, 'description'):
        button.description = new_label