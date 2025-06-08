"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Fixed UI layout dengan border styling dan balanced layout
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI dengan proper styling dan progress tracker compatibility"""
    
    try:
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_triple_progress_tracker
        
        # Header
        header = create_header(
            f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
            "Augmentasi dataset dengan progress tracking dan service integration"
        )
        
        # Status panel
        status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")
        
        # Widget groups
        basic_options = _create_basic_options_safe()
        advanced_options = _create_advanced_options_safe()
        augmentation_types = _create_augmentation_types_safe()
        
        # Progress tracker - adapt to existing factory
        progress_result = create_triple_progress_tracker(auto_hide=True)
        
        # Config buttons
        config_buttons = create_save_reset_buttons(
            save_label="💾 Simpan Config", 
            reset_label="🔄 Reset Config"
        )
        
        # Action buttons
        action_buttons = create_action_buttons(
            primary_label="🎯 Run Augmentation", 
            primary_icon="play",
            secondary_buttons=[("🔍 Check Dataset", "search", "info")],
            cleanup_enabled=True, 
            button_width="180px"
        )
        
        # Output areas
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', margin='8px 0', max_height='200px'
        ))
        
        log_components = create_log_accordion('augmentation', '200px')
        
        # Styled sections dengan border dan balanced layout
        settings_section = _create_settings_section([
            _create_card_section("📋 Opsi Dasar", basic_options['container'], "#e3f2fd"),
            _create_card_section("⚙️ Opsi Lanjutan", advanced_options['container'], "#f3e5f5")
        ])
        
        types_section = _create_card_section(
            "🔄 Jenis Augmentasi & Target Split", 
            augmentation_types['container'], 
            "#e8f5e8",
            full_width=True
        )
        
        # Action section dengan styling
        action_section = widgets.VBox([
            widgets.HTML(f"""
            <div style="padding: 12px; margin: 15px 0 10px 0; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 8px; border-left: 4px solid #5a67d8;">
                <h4 style="color: white; margin: 0; font-size: 16px;">
                    ▶️ Aksi Augmentasi
                </h4>
            </div>
            """),
            action_buttons['container'],
            confirmation_area
        ])
        
        # Config section dengan styling
        config_section = widgets.VBox([
            widgets.HTML(f"""
            <div style="padding: 8px 12px; margin: 10px 0 5px 0; 
                        background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%);
                        border-radius: 6px; border-left: 4px solid #f6ad55;">
                <h5 style="color: #744210; margin: 0; font-size: 14px;">
                    💾 Manajemen Konfigurasi
                </h5>
            </div>
            """),
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ])
        
        # Main UI assembly
        ui = widgets.VBox([
            header,
            status_panel,
            settings_section,
            types_section,
            config_section,
            action_section,
            progress_result['container'],
            log_components['log_accordion']
        ], layout=widgets.Layout(width='100%'))
        
        # FIXED: Component mapping dengan backward compatibility
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
            
            # FIXED: Progress tracker - use dict keys directly from factory
            'progress_tracker': progress_result.get('tracker'),
            'progress_container': progress_result.get('container'),
            'show_container': progress_result.get('show_container'),
            'update_overall': progress_result.get('update_overall'),
            'update_step': progress_result.get('update_step'),
            'update_current': progress_result.get('update_current'),
            'complete_operation': progress_result.get('complete_operation'),
            'error_operation': progress_result.get('error_operation'),
            'reset_all': progress_result.get('reset_all'),
            
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

def _create_settings_section(cards_list):
    """Create balanced 2-column layout untuk settings"""
    if len(cards_list) >= 2:
        return widgets.HBox([
            widgets.Box([cards_list[0]], layout=widgets.Layout(
                width='49%', margin='0 1% 0 0'
            )),
            widgets.Box([cards_list[1]], layout=widgets.Layout(
                width='49%'
            ))
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
    else:
        return widgets.VBox(cards_list)

def _create_card_section(title: str, content_widget, bg_color: str, full_width: bool = False):
    """Create styled card section dengan border dan gradient"""
    border_colors = {
        "#e3f2fd": "#2196f3",  # Blue
        "#f3e5f5": "#9c27b0",  # Purple  
        "#e8f5e8": "#4caf50",  # Green
        "#fff3e0": "#ff9800"   # Orange
    }
    
    border_color = border_colors.get(bg_color, "#2196f3")
    
    card_html = f"""
    <div style="padding: 10px 12px 6px 12px; margin-bottom: 8px;
                background: linear-gradient(145deg, {bg_color} 0%, rgba(255,255,255,0.9) 100%);
                border-radius: 8px; border-left: 4px solid {border_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: #333; margin: 0; font-size: 14px; font-weight: 600;">
            {title}
        </h5>
    </div>
    """
    
    width = '100%' if full_width else '100%'
    margin = '10px 0' if full_width else '0'
    
    return widgets.VBox([
        widgets.HTML(card_html),
        content_widget
    ], layout=widgets.Layout(
        width=width, 
        margin=margin,
        padding='0',
        border=f'1px solid {border_color}20',
        border_radius='8px',
        background_color='rgba(255,255,255,0.7)'
    ))

def _create_basic_options_safe() -> Dict[str, Any]:
    """Basic options dengan compact styling"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        return create_basic_options_widget()
    except ImportError:
        widgets_dict = {
            'num_variations': widgets.IntSlider(
                value=3, min=1, max=10, description='Variasi:', 
                style={'description_width': '80px'},
                layout=widgets.Layout(width='95%')
            ),
            'target_count': widgets.IntSlider(
                value=500, min=100, max=2000, step=100, description='Target:',
                style={'description_width': '80px'},
                layout=widgets.Layout(width='95%')
            ),
            'output_prefix': widgets.Text(
                value='aug', description='Prefix:',
                style={'description_width': '80px'},
                layout=widgets.Layout(width='95%')
            ),
            'balance_classes': widgets.Checkbox(
                value=True, description='Balance Classes',
                layout=widgets.Layout(margin='5px 0')
            )
        }
        container = widgets.VBox(list(widgets_dict.values()), 
                                layout=widgets.Layout(padding='10px'))
        return {'container': container, 'widgets': widgets_dict}

def _create_advanced_options_safe() -> Dict[str, Any]:
    """Advanced options dengan tab layout"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        return create_advanced_options_widget()
    except ImportError:
        # Position tab
        pos_widgets = {
            'fliplr': widgets.FloatSlider(value=0.5, min=0.0, max=1.0, description='Flip:', style={'description_width': '60px'}),
            'degrees': widgets.IntSlider(value=10, min=0, max=30, description='Rotasi:', style={'description_width': '60px'}),
            'translate': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Trans:', style={'description_width': '60px'}),
            'scale': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Scale:', style={'description_width': '60px'})
        }
        
        # Lighting tab
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

def _create_augmentation_types_safe() -> Dict[str, Any]:
    """Augmentation types dengan responsive layout"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        return create_augmentation_types_widget()
    except ImportError:
        widgets_dict = {
            'augmentation_types': widgets.SelectMultiple(
                options=[('Combined', 'combined'), ('Position', 'position'), ('Lighting', 'lighting')],
                value=['combined'], description='Types:',
                layout=widgets.Layout(width='60%', height='80px'),
                style={'description_width': '60px'}
            ),
            'target_split': widgets.Dropdown(
                options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
                value='train', description='Split:',
                layout=widgets.Layout(width='35%'),
                style={'description_width': '50px'}
            )
        }
        
        container = widgets.HBox([
            widgets_dict['augmentation_types'],
            widgets_dict['target_split']
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
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
        'ui': ui,
        'augment_button': fallback_button,
        'check_button': fallback_button,
        'cleanup_button': fallback_button,
        'save_button': fallback_button,
        'reset_button': fallback_button,
        'log_output': log_output,
        'status': log_output,
        'confirmation_area': log_output,
        'error': error_message
    }