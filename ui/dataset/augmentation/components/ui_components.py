"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Fixed UI components dengan augmentation types full width dan layout yang diperbaiki
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI dengan augmentation types full width dan layout yang diperbaiki"""
    
    try:
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
        
        # Header
        header = create_header(
            f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
            "Augmentasi dataset dengan backend integration dan comprehensive check"
        )
        
        # Status panel
        status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")
        
        # Widget groups dengan border styling
        basic_options = _create_basic_options_group()
        advanced_options = _create_advanced_options_group()
        augmentation_types = _create_augmentation_types_group()  # FIXED: Full width
        
        # Progress tracker
        progress_tracker = create_dual_progress_tracker("Augmentation", auto_hide=False)
        
        # Config buttons
        config_buttons = create_save_reset_buttons(
            save_label="Simpan", 
            reset_label="Reset"
        )
        
        # Action buttons dengan enhanced labels
        action_buttons = create_action_buttons(
            primary_label="🎯 Run Augmentation Pipeline", 
            primary_icon="play",
            secondary_buttons=[("🔍 Check Dataset", "search", "info")],
            cleanup_enabled=True, 
            button_width="200px"
        )
        
        # Output areas
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', margin='8px 0', max_height='200px'
        ))
        
        log_components = create_log_accordion('augmentation', '200px')
        
        # FIXED: Layout dengan augmentation types full width
        # Row 1: Basic dan Advanced options side by side
        options_row = widgets.HBox([
            basic_options['container'], 
            advanced_options['container']
        ], layout=widgets.Layout(
            width='100%', 
            display="flex", 
            justify_content='space-between', 
            gap='15px'
        ))
        
        # Row 2: Augmentation types full width
        types_row = widgets.VBox([
            augmentation_types['container']
        ], layout=widgets.Layout(width='100%', overflow='hidden'))
        
        # Action section
        action_section = widgets.VBox([
            _create_section_header("▶️ Aksi Augmentasi", "#667eea"),
            action_buttons['container'],
            confirmation_area
        ])
        
        # Config section
        config_section = widgets.VBox([
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ])
        
        # FIXED: Main UI assembly dengan layout yang diperbaiki
        ui = widgets.VBox([
            header,
            status_panel,
            options_row,        # Basic + Advanced side by side
            types_row,          # Augmentation types full width
            config_section,
            action_section,
            progress_tracker.container,
            log_components['log_accordion']
        ], layout=widgets.Layout(width='100%', overflow='hidden'))
        
        # Component mapping
        return {
            'ui': ui,
            'header': header,
            'status_panel': status_panel,
            'confirmation_area': confirmation_area,
            
            # Form widgets
            **basic_options['widgets'],
            **advanced_options['widgets'], 
            **augmentation_types['widgets'],
            
            # Buttons
            'augment_button': action_buttons['download_button'],
            'check_button': action_buttons['check_button'],
            'cleanup_button': action_buttons.get('cleanup_button'),
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            
            # Progress tracker
            'progress_tracker': progress_tracker,
            'show_container': lambda op: progress_tracker.show(),
            'update_overall': lambda pct, msg: progress_tracker.update_overall(pct, msg),
            'update_step': lambda pct, msg: progress_tracker.update_step(pct, msg),
            'complete_operation': lambda msg: progress_tracker.complete(msg),
            'error_operation': lambda msg: progress_tracker.error(msg),
            'reset_all': lambda: progress_tracker.reset(),
            
            # Log outputs
            'log_output': log_components['log_output'],
            'status': log_components['log_output'],
            
            # Metadata
            'module_name': 'augmentation',
            'logger_namespace': 'smartcash.ui.dataset.augmentation',
            'config': config or {}
        }
        
    except Exception as e:
        return _create_fallback_ui(str(e))

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
    <h4 style="color: #333; margin: 15px 0 10px 0; border-bottom: 2px solid {color}; 
               font-size: 16px; padding-bottom: 6px;">
        {title}
    </h4>
    """)

def _create_basic_options_group() -> Dict[str, Any]:
    """Basic options dengan border styling - 48% width"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        result = create_basic_options_widget()
        result['container'] = _add_border_styling(result['container'], "📋 Opsi Dasar", "#4caf50", "48%")
        return result
    except ImportError:
        return _create_fallback_basic_options()

def _create_advanced_options_group() -> Dict[str, Any]:
    """Advanced options dengan border styling - 48% width"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        result = create_advanced_options_widget()
        result['container'] = _add_border_styling(result['container'], "⚙️ Opsi Lanjutan", "#9c27b0", "48%")
        return result
    except ImportError:
        return _create_fallback_advanced_options()

def _create_augmentation_types_group() -> Dict[str, Any]:
    """FIXED: Augmentation types dengan full width styling"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        result = create_augmentation_types_widget()
        # FIXED: Full width container
        result['container'] = _add_border_styling(result['container'], "🔄 Jenis Augmentasi & Target Split", "#2196f3", "100%")
        return result
    except ImportError:
        return _create_fallback_types_options()

def _add_border_styling(content_widget, title: str, border_color: str, width: str = "48%") -> widgets.VBox:
    """Add border styling ke widget group dengan custom width"""
    bg_color = f"{border_color}15"  # 15% opacity
    
    header_html = f"""
    <div style="padding: 8px 12px; margin-bottom: 8px;
                background: linear-gradient(145deg, {bg_color} 0%, rgba(255,255,255,0.9) 100%);
                border-radius: 8px; border-left: 4px solid {border_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: #333; margin: 0; font-size: 14px; font-weight: 600;">
            {title}
        </h5>
    </div>
    """
    
    return widgets.VBox([
        widgets.HTML(header_html),
        content_widget
    ], layout=widgets.Layout(
        width=width,                    # FIXED: Dynamic width
        margin='5px',
        padding='10px',
        border=f'1px solid {border_color}40',
        border_radius='8px',
        background_color='rgba(255,255,255,0.8)'
    ))

# Fallback implementations (unchanged)
def _create_fallback_basic_options() -> Dict[str, Any]:
    """Fallback basic options"""
    widgets_dict = {
        'num_variations': widgets.IntSlider(value=3, min=1, max=10, description='Variasi:', 
                                          style={'description_width': '80px'}, layout=widgets.Layout(width='95%')),
        'target_count': widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Target:',
                                        style={'description_width': '80px'}, layout=widgets.Layout(width='95%')),
        'output_prefix': widgets.Text(value='aug', description='Prefix:', style={'description_width': '80px'}, 
                                    layout=widgets.Layout(width='95%')),
        'balance_classes': widgets.Checkbox(value=True, description='Balance Classes', layout=widgets.Layout(margin='5px 0'))
    }
    container = widgets.VBox(list(widgets_dict.values()), layout=widgets.Layout(padding='10px'))
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_advanced_options() -> Dict[str, Any]:
    """Fallback advanced options"""
    pos_widgets = {
        'fliplr': widgets.FloatSlider(value=0.5, min=0.0, max=1.0, description='Flip:', style={'description_width': '60px'}),
        'degrees': widgets.IntSlider(value=10, min=0, max=30, description='Rotasi:', style={'description_width': '60px'}),
        'translate': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Trans:', style={'description_width': '60px'}),
        'scale': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Scale:', style={'description_width': '60px'})
    }
    
    light_widgets = {
        'hsv_h': widgets.FloatSlider(value=0.015, min=0.0, max=0.05, description='HSV H:', style={'description_width': '60px'}),
        'hsv_s': widgets.FloatSlider(value=0.7, min=0.0, max=1.0, description='HSV S:', style={'description_width': '60px'}),
        'brightness': widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Bright:', style={'description_width': '60px'}),
        'contrast': widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Contrast:', style={'description_width': '60px'})
    }
    
    pos_tab = widgets.VBox(list(pos_widgets.values()))
    light_tab = widgets.VBox(list(light_widgets.values()))
    tabs = widgets.Tab([pos_tab, light_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    all_widgets = {**pos_widgets, **light_widgets}
    return {'container': tabs, 'widgets': all_widgets}

def _create_fallback_types_options() -> Dict[str, Any]:
    """Fallback augmentation types dengan full width"""
    widgets_dict = {
        'augmentation_types': widgets.SelectMultiple(
            options=[('Combined', 'combined'), ('Position', 'position'), ('Lighting', 'lighting')],
            value=['combined'], description='Types:',
            layout=widgets.Layout(width='100%', height='80px'),  # FIXED: Full width
            style={'description_width': '100px'}
        ),
        'target_split': widgets.Dropdown(
            options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
            value='train', description='Split:',
            layout=widgets.Layout(width='100%'),                # FIXED: Full width
            style={'description_width': '100px'}
        )
    }
    
    container = widgets.VBox([
        widgets_dict['augmentation_types'],
        widgets_dict['target_split']
    ], layout=widgets.Layout(width='100%', padding='10px'))  # FIXED: Full width
    
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """Minimal fallback UI"""
    error_widget = widgets.HTML(f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #dc3545; 
                border-radius: 8px; color: #721c24; margin: 10px 0;">
        <h4>⚠️ Augmentation UI Error</h4>
        <p><strong>Error:</strong> {error_message}</p>
        <p>💡 Restart cell atau check imports</p>
    </div>
    """)
    
    fallback_button = widgets.Button(description="🔄 Retry", button_style='primary')
    log_output = widgets.Output()
    ui = widgets.VBox([error_widget, fallback_button, log_output])
    
    return {
        'ui': ui, 'augment_button': fallback_button, 'check_button': fallback_button,
        'cleanup_button': fallback_button, 'save_button': fallback_button, 'reset_button': fallback_button,
        'log_output': log_output, 'status': log_output, 'confirmation_area': log_output,
        'progress_tracker': None, 'error': error_message
    }