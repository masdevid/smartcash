"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Updated UI components dengan live preview integration dan cleanup target
"""

from IPython.display import display, HTML
import ipywidgets as widgets
from typing import Dict, Any

# Import internal components
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    styled_container, flex_layout
)

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI dengan live preview integration dan cleanup target update"""
    
    try:
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker

        # Header dan status panel
        header = create_header(
            f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
            "Pipeline augmentasi dengan live preview dan backend integration"
        )
        status_panel = create_status_panel("✅ Pipeline augmentasi siap", "success")
        
        # Widget groups dengan live preview integration
        basic_options = _create_basic_options_group()
        advanced_options = _create_advanced_options_group()
        augmentation_types = _create_augmentation_types_group()
        live_preview = _create_live_preview_group()  # CHANGED: Menggantikan normalization
        
        # Progress tracker dan buttons
        progress_tracker = create_dual_progress_tracker("Augmentation Pipeline", auto_hide=True)
        config_buttons = create_save_reset_buttons(
            save_label="Simpan", reset_label="Reset",
            with_sync_info=True, sync_message="Konfigurasi disinkronkan dengan backend"
        )
        action_buttons = create_action_buttons(
            primary_label="🚀 Jalankan Augmentasi", primary_icon="play",
            secondary_buttons=[("🔍 Cek Data", "search", "info")],
            cleanup_enabled=True, 
            button_width="220px"
        )
        
        # Confirmation area untuk dialog integration
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', 
            min_height='50px',
            max_height='800px', 
            margin='10px 0',
            padding='5px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            overflow='auto',
            background_color='#fafafa'
        ))
        
        # Log accordion
        log_components = create_log_accordion('augmentation', '250px')
        
        # 2x2 Grid dengan live preview
        row1 = widgets.HBox([
            styled_container(basic_options['container'], "📋 Opsi Dasar", 'basic', '47%'),
            styled_container(advanced_options['container'], "⚙️ Parameter Lanjutan", 'advanced', '47%')
        ], layout=widgets.Layout(
            width='100%', max_width='100%', display='flex',
            flex_flow='row wrap', justify_content='space-between',
            align_items='stretch', gap='6px', margin='8px 0',
            overflow='hidden', box_sizing='border-box'
        ))
        
        row2 = widgets.HBox([
            styled_container(augmentation_types['container'], "🔄 Jenis Augmentasi", 'types', '47%'),
            styled_container(live_preview['container'], "🎬 Live Preview", 'normalization', '47%')  # CHANGED
        ], layout=widgets.Layout(
            width='100%', max_width='100%', display='flex',
            flex_flow='row wrap', justify_content='space-between',
            align_items='stretch', gap='6px', margin='8px 0',
            overflow='hidden', box_sizing='border-box'
        ))
        
        # Action section dengan confirmation area
        action_section = widgets.VBox([
            _create_section_header("🚀 Pipeline Operations", "#667eea"),
            action_buttons['container'],
            widgets.HTML("<div style='margin: 5px 0;'><strong>📋 Status & Konfirmasi:</strong></div>"),
            confirmation_area
        ], layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#f9f9f9'
        ))
        
        # Config section
        config_section = widgets.VBox([
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ])
        
        # Main UI assembly dengan urutan yang benar
        ui = widgets.VBox([
            header, 
            status_panel, 
            row1, 
            row2, 
            config_section,
            action_section,
            progress_tracker.container,
            log_components['log_accordion']
        ], layout=widgets.Layout(
            width='100%', max_width='100%', display='flex',
            flex_flow='column', align_items='stretch'
        ))
        
        # Component mapping dengan live preview integration
        return {
            'ui': ui, 
            'header': header, 
            'status_panel': status_panel,
            'confirmation_area': confirmation_area,
            **basic_options['widgets'], 
            **advanced_options['widgets'],
            **augmentation_types['widgets'], 
            **live_preview['widgets'],  # CHANGED: Live preview widgets
            'augment_button': action_buttons['download_button'],
            'check_button': action_buttons['check_button'],
            'cleanup_button': action_buttons.get('cleanup_button'),
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            'progress_tracker': progress_tracker,
            'log_output': log_components['log_output'],
            'status': log_components['log_output'],
            'backend_ready': True, 
            'service_integration': True,
            'module_name': 'augmentation',
            'logger_namespace': 'smartcash.ui.dataset.augmentation',
            'augmentation_initialized': True,
            'config': config or {}
        }
        
    except Exception as e:
        return _create_fallback_ui(str(e))

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
    <h4 style="color: #333; margin: 10px 0 8px 0; border-bottom: 2px solid {color}; 
               font-size: 14px; padding-bottom: 4px;">
        {title}
    </h4>
    """)

def _create_basic_options_group() -> Dict[str, Any]:
    """Basic options group dengan cleanup target integration"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        return create_basic_options_widget()
    except ImportError:
        return _create_fallback_basic_options()

def _create_advanced_options_group() -> Dict[str, Any]:
    """Advanced options group dengan HSV parameters"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        return create_advanced_options_widget()
    except ImportError:
        return _create_fallback_advanced_options()

def _create_augmentation_types_group() -> Dict[str, Any]:
    """Augmentation types group"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        return create_augmentation_types_widget()
    except ImportError:
        return _create_fallback_types_options()

def _create_live_preview_group() -> Dict[str, Any]:
    """Live preview group - NEW"""
    try:
        from smartcash.ui.dataset.augmentation.components.live_preview_widget import create_live_preview_widget
        return create_live_preview_widget()
    except ImportError:
        return _create_fallback_live_preview()

# Fallback implementations
def _create_fallback_basic_options() -> Dict[str, Any]:
    """Fallback basic options dengan cleanup target"""
    from smartcash.ui.dataset.augmentation.utils.style_utils import style_widget, flex_layout
    
    widgets_dict = {
        'num_variations': style_widget(widgets.IntSlider(
            value=2, min=1, max=10, description='Variasi:', readout=True
        ), '80px'),
        'target_count': style_widget(widgets.IntSlider(
            value=500, min=100, max=2000, step=100, description='Target:', readout=True
        ), '80px'),
        'intensity': style_widget(widgets.FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.1, description='Intensitas:', readout=True
        ), '80px'),
        'target_split': style_widget(widgets.Dropdown(
            options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
            value='train', description='Target Split:'
        ), '80px'),
        'cleanup_target': style_widget(widgets.Dropdown(
            options=[('Both', 'both'), ('Augmented', 'augmented'), ('Samples', 'samples')],
            value='both', description='Cleanup:'
        ), '80px'),
        'balance_classes': widgets.Checkbox(
            value=True, description='Balance Classes',
            layout=widgets.Layout(margin='5px 0')
        )
    }
    
    container = widgets.VBox(list(widgets_dict.values()))
    flex_layout(container)
    
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_advanced_options() -> Dict[str, Any]:
    """Fallback advanced options dengan HSV parameters"""
    from smartcash.ui.dataset.augmentation.utils.style_utils import style_widget, flex_layout
    
    # Position widgets
    pos_widgets = {
        'fliplr': style_widget(widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, description='Flip:', readout=True
        ), '60px'),
        'degrees': style_widget(widgets.IntSlider(
            value=12, min=0, max=30, description='Rotasi:', readout=True
        ), '60px'),
        'translate': style_widget(widgets.FloatSlider(
            value=0.08, min=0.0, max=0.25, description='Trans:', readout=True
        ), '60px'),
        'scale': style_widget(widgets.FloatSlider(
            value=0.04, min=0.0, max=0.25, description='Scale:', readout=True
        ), '60px')
    }
    
    # Lighting widgets dengan HSV
    light_widgets = {
        'brightness': style_widget(widgets.FloatSlider(
            value=0.2, min=0.0, max=0.4, description='Bright:', readout=True
        ), '60px'),
        'contrast': style_widget(widgets.FloatSlider(
            value=0.15, min=0.0, max=0.4, description='Contrast:', readout=True
        ), '60px'),
        'hsv_h': style_widget(widgets.IntSlider(
            value=10, min=0, max=30, description='HSV H:', readout=True
        ), '60px'),
        'hsv_s': style_widget(widgets.IntSlider(
            value=15, min=0, max=50, description='HSV S:', readout=True
        ), '60px')
    }
    
    # Create tabs
    pos_tab = widgets.VBox(list(pos_widgets.values()))
    light_tab = widgets.VBox(list(light_widgets.values()))
    flex_layout(pos_tab)
    flex_layout(light_tab)
    
    tabs = widgets.Tab([pos_tab, light_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    all_widgets = {**pos_widgets, **light_widgets}
    return {'container': tabs, 'widgets': all_widgets}

def _create_fallback_types_options() -> Dict[str, Any]:
    """Fallback augmentation types"""
    widgets_dict = {
        'augmentation_types': widgets.SelectMultiple(
            options=[
                ('Combined', 'combined'), 
                ('Position', 'position'), 
                ('Lighting', 'lighting')
            ],
            value=['combined'], description='Types:',
            layout=widgets.Layout(width='100%', height='80px'),
            style={'description_width': '100px'}
        )
    }
    
    container = widgets.VBox([widgets_dict['augmentation_types']])
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_live_preview() -> Dict[str, Any]:
    """Fallback live preview"""
    preview_image = widgets.Image(
        value=b'', format='jpg',
        layout=widgets.Layout(width='200px', height='200px', border='1px solid #ddd')
    )
    
    generate_button = widgets.Button(
        description='🎯 Generate Preview',
        button_style='info',
        layout=widgets.Layout(width='180px', height='32px')
    )
    
    preview_status = widgets.HTML(
        value="<div style='text-align: center; color: #666; font-size: 12px;'>Preview: /data/aug_preview.jpg</div>"
    )
    
    container = widgets.VBox([
        widgets.HTML("<h6>🎬 Live Preview</h6>"),
        preview_image,
        preview_status,
        generate_button
    ])
    
    widgets_dict = {
        'preview_image': preview_image,
        'generate_button': generate_button,
        'preview_status': preview_status
    }
    
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
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', min_height='50px', border='1px solid #ddd'
    ))
    
    ui = widgets.VBox([error_widget, fallback_button, confirmation_area, log_output], 
                     layout=widgets.Layout(display='flex', flex_flow='column', align_items='stretch'))
    
    # Return minimal required components
    return {
        'ui': ui, 
        'augment_button': fallback_button, 
        'check_button': fallback_button,
        'cleanup_button': fallback_button, 
        'save_button': fallback_button, 
        'reset_button': fallback_button, 
        'log_output': log_output, 
        'status': log_output,
        'confirmation_area': confirmation_area,
        'progress_tracker': None, 
        'error': error_message,
        'intensity': widgets.FloatSlider(value=0.7, min=0.1, max=1.0),
        'target_split': widgets.Dropdown(options=[('train', 'train')], value='train'),
        'cleanup_target': widgets.Dropdown(options=[('both', 'both')], value='both'),
        'augmentation_initialized': True,
        'preview_image': widgets.Image(value=b''),
        'generate_button': fallback_button,
        'preview_status': widgets.HTML(value="Preview unavailable")
    }