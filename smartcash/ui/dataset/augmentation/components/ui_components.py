"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: UI components dengan 2x2 grid layout yang responsive
"""

from IPython.display import display
import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI dengan 2x2 grid layout"""
    
    try:
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
        
        # Header dengan professional description
        header = create_header(
            f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
            "Pipeline augmentasi dataset untuk peningkatan performa model deteksi denominasi mata uang"
        )
        
        # Status panel
        status_panel = create_status_panel("✅ Pipeline augmentasi siap", "success")
        
        # Widget groups
        basic_options = _create_basic_options_group()
        advanced_options = _create_advanced_options_group()
        augmentation_types = _create_augmentation_types_group()
        normalization_options = _create_normalization_options_group()
        
        # Progress tracker
        progress_tracker = create_dual_progress_tracker("Augmentation Pipeline", auto_hide=True)
        
        # Config buttons
        config_buttons = create_save_reset_buttons(
            save_label="Simpan", 
            reset_label="Reset",
            with_sync_info=True,
            sync_message="Konfigurasi disinkronkan dengan backend"
        )
        
        # Action buttons
        action_buttons = create_action_buttons(
            primary_label="🚀 Run Augmentation Pipeline", 
            primary_icon="play",
            secondary_buttons=[("🔍 Comprehensive Check", "search", "info")],
            cleanup_enabled=True, 
            button_width="220px"
        )
        
        # Output areas
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', margin='8px 0', max_height='300px', overflow='auto'
        ))
        
        log_components = create_log_accordion('augmentation', '250px')
        
        # FLEXBOX 2x2 GRID LAYOUT
        # Row 1: Opsi Dasar | Parameter Lanjutan
        row1 = widgets.HBox([
            basic_options['container'],     
            advanced_options['container']   
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-between',
            align_items='stretch',
            gap='15px',
            margin='8px 0'
        ))
        
        # Row 2: Jenis Augmentasi | Normalisasi Augmentasi  
        row2 = widgets.HBox([
            augmentation_types['container'],    
            normalization_options['container'] 
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-between',
            align_items='stretch',
            gap='15px',
            margin='8px 0'
        ))
        
        # Action section dengan flexbox
        action_section = widgets.VBox([
            _create_section_header("🚀 Pipeline Operations", "#667eea"),
            action_buttons['container'],
            confirmation_area
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        ))
        
        # Config section dengan flexbox
        config_section = widgets.VBox([
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(
                    display='flex', 
                    justify_content='flex-end', 
                    width='100%'
                ))
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column'
        ))
        
        # Main UI assembly dengan flexbox container
        ui = widgets.VBox([
            header,
            status_panel,
            row1,              # Opsi Dasar | Parameter Lanjutan
            row2,              # Jenis Augmentasi | Normalisasi Augmentasi
            config_section,
            action_section,
            progress_tracker.container,
            log_components['log_accordion']
        ], layout=widgets.Layout(
            width='100%', 
            max_width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        ))
        
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
            **normalization_options['widgets'],
            
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
            'update_current': lambda pct, msg: progress_tracker.update_current(pct, msg),
            'complete_operation': lambda msg: progress_tracker.complete(msg),
            'error_operation': lambda msg: progress_tracker.error(msg),
            'reset_all': lambda: progress_tracker.reset(),
            
            # Log outputs
            'log_output': log_components['log_output'],
            'status': log_components['log_output'],
            
            # Backend integration
            'backend_ready': True,
            'service_integration': True,
            
            # Metadata
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
    <h4 style="color: #333; margin: 15px 0 10px 0; border-bottom: 2px solid {color}; 
               font-size: 16px; padding-bottom: 6px;">
        {title}
    </h4>
    """)

def _create_basic_options_group() -> Dict[str, Any]:
    """Basic options group - 48% width untuk grid"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        result = create_basic_options_widget()
        result['container'] = _add_border_styling(result['container'], "📋 Opsi Dasar", "#4caf50", "48%")
        return result
    except ImportError:
        return _create_fallback_basic_options()

def _create_advanced_options_group() -> Dict[str, Any]:
    """Advanced options group - 48% width untuk grid"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        result = create_advanced_options_widget()
        result['container'] = _add_border_styling(result['container'], "⚙️ Parameter Lanjutan", "#9c27b0", "48%")
        return result
    except ImportError:
        return _create_fallback_advanced_options()

def _create_augmentation_types_group() -> Dict[str, Any]:
    """Augmentation types group - 48% width untuk grid"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        result = create_augmentation_types_widget()
        result['container'] = _add_border_styling(result['container'], "🔄 Jenis Augmentasi", "#2196f3", "48%")
        return result
    except ImportError:
        return _create_fallback_types_options()

def _create_normalization_options_group() -> Dict[str, Any]:
    """Normalization options group - 48% width untuk grid"""
    norm_method = widgets.Dropdown(
        options=[
            ('MinMax [0,1] - YOLO Recommended', 'minmax'),
            ('Standard (Z-score)', 'standard'),
            ('ImageNet Normalization', 'imagenet'),
            ('No Normalization', 'none')
        ],
        value='minmax',
        description='Method:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%')
    )
    
    denormalize = widgets.Checkbox(
        value=False,
        description='Denormalize (save as uint8)',
        indent=False,
        layout=widgets.Layout(margin='8px 0')
    )
    
    norm_info = widgets.HTML(f"""
    <div style="padding: 8px; background-color: #ff980315; 
                border-radius: 4px; margin: 5px 0; font-size: 11px;
                border: 1px solid #ff980340;">
        <strong style="color: #f57c00;">ℹ️ Normalization Info:</strong><br>
        • <strong style="color: #f57c00;">MinMax:</strong> Optimal untuk YOLO training<br>
        • <strong style="color: #f57c00;">Denormalize:</strong> Save sebagai uint8 [0,255]<br>
        • <strong style="color: #f57c00;">Target Size:</strong> 640x640 fixed untuk YOLO
    </div>
    """)
    
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #f57c00; margin: 5px 0;'>📊 Normalisasi Augmentasi</h6>"),
        norm_method,
        denormalize,
        norm_info
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    result = {
        'container': _add_border_styling(container, "📊 Normalisasi Augmentasi", "#ff9800", "48%"),
        'widgets': {
            'norm_method': norm_method,
            'denormalize': denormalize
        }
    }
    
    return result

def _add_border_styling(content_widget, title: str, border_color: str, width: str = "48%") -> widgets.VBox:
    """Add border styling dengan flexbox untuk 2x2 grid"""
    bg_color = f"{border_color}15"
    
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
        width=width,
        margin='5px',
        padding='10px',
        border=f'1px solid {border_color}40',
        border_radius='8px',
        background_color='rgba(255,255,255,0.8)',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        flex='1 1 48%'  # Flexbox dengan equal distribution
    ))

# Fallback implementations dengan flexbox dan target_split support
def _create_fallback_basic_options() -> Dict[str, Any]:
    """Fallback basic options dengan target_split dan flexbox"""
    widgets_dict = {
        'num_variations': widgets.IntSlider(value=2, min=1, max=10, description='Variasi:', 
                                          style={'description_width': '80px'}, layout=widgets.Layout(width='95%')),
        'target_count': widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Target:',
                                        style={'description_width': '80px'}, layout=widgets.Layout(width='95%')),
        'intensity': widgets.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.1, description='Intensitas:',
                                       style={'description_width': '80px'}, layout=widgets.Layout(width='95%')),
        'target_split': widgets.Dropdown(
            options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
            value='train', description='Target Split:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '80px'}
        ),
        'output_prefix': widgets.Text(value='aug', description='Prefix:', style={'description_width': '80px'}, 
                                    layout=widgets.Layout(width='95%')),
        'balance_classes': widgets.Checkbox(value=True, description='Balance Classes', layout=widgets.Layout(margin='5px 0'))
    }
    container = widgets.VBox(list(widgets_dict.values()), layout=widgets.Layout(
        padding='10px',
        display='flex',
        flex_flow='column',
        align_items='stretch'
    ))
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_advanced_options() -> Dict[str, Any]:
    """Fallback advanced options dengan flexbox"""
    pos_widgets = {
        'fliplr': widgets.FloatSlider(value=0.5, min=0.0, max=1.0, description='Flip:', style={'description_width': '60px'}),
        'degrees': widgets.IntSlider(value=12, min=0, max=30, description='Rotasi:', style={'description_width': '60px'}),
        'translate': widgets.FloatSlider(value=0.08, min=0.0, max=0.25, description='Trans:', style={'description_width': '60px'}),
        'scale': widgets.FloatSlider(value=0.04, min=0.0, max=0.25, description='Scale:', style={'description_width': '60px'})
    }
    
    light_widgets = {
        'brightness': widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Bright:', style={'description_width': '60px'}),
        'contrast': widgets.FloatSlider(value=0.15, min=0.0, max=0.4, description='Contrast:', style={'description_width': '60px'})
    }
    
    pos_tab = widgets.VBox(list(pos_widgets.values()), layout=widgets.Layout(
        display='flex', flex_flow='column', align_items='stretch'
    ))
    light_tab = widgets.VBox(list(light_widgets.values()), layout=widgets.Layout(
        display='flex', flex_flow='column', align_items='stretch'
    ))
    tabs = widgets.Tab([pos_tab, light_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    all_widgets = {**pos_widgets, **light_widgets}
    return {'container': tabs, 'widgets': all_widgets}

def _create_fallback_types_options() -> Dict[str, Any]:
    """Fallback augmentation types tanpa target_split"""
    widgets_dict = {
        'augmentation_types': widgets.SelectMultiple(
            options=[('Combined', 'combined'), ('Position', 'position'), ('Lighting', 'lighting')],
            value=['combined'], description='Types:',
            layout=widgets.Layout(width='100%', height='80px'),
            style={'description_width': '100px'}
        )
    }
    
    container = widgets.VBox([
        widgets_dict['augmentation_types']
    ], layout=widgets.Layout(
        width='100%', 
        padding='10px',
        display='flex',
        flex_flow='column',
        align_items='stretch'
    ))
    
    return {'container': container, 'widgets': widgets_dict}

def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """Minimal fallback UI dengan flexbox"""
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
    ui = widgets.VBox([error_widget, fallback_button, log_output], layout=widgets.Layout(
        display='flex', flex_flow='column', align_items='stretch'
    ))
    
    # Return minimal required components dengan intensity dan target_split support
    return {
        'ui': ui, 'augment_button': fallback_button, 'check_button': fallback_button,
        'cleanup_button': fallback_button, 'save_button': fallback_button, 'reset_button': fallback_button,
        'log_output': log_output, 'status': log_output, 'confirmation_area': log_output,
        'progress_tracker': None, 'error': error_message,
        'norm_method': widgets.Dropdown(options=[('minmax', 'minmax')], value='minmax'),
        'denormalize': widgets.Checkbox(value=False),
        'intensity': widgets.FloatSlider(value=0.7, min=0.1, max=1.0),
        'target_split': widgets.Dropdown(options=[('train', 'train')], value='train'),  # Added target_split
        'augmentation_initialized': True
    }