"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Fixed main UI assembly dengan proper widget extraction dan error handling
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main UI assembly dengan proper error handling dan widget extraction
    
    Args:
        config: Konfigurasi untuk initialize UI values
        
    Returns:
        Dictionary berisi semua UI components dengan proper mapping
    """
    try:
        # Import reused components dengan error handling
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_triple_progress_tracker
        
        # Header dengan module context
        header = create_header(
            f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
            "Augmentasi dataset dengan progress tracking dan service integration"
        )
        
        # Status panel initial
        status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")
        
        # Widget groups dengan safe creation
        basic_options = _create_basic_options_safe()
        advanced_options = _create_advanced_options_safe()
        augmentation_types = _create_augmentation_types_safe()
        
        # Progress tracker dengan safe config
        progress_tracker = create_triple_progress_tracker(
            operation="Dataset Augmentation",
            steps=["prepare", "augment", "normalize", "verify"],
            step_weights={"prepare": 10, "augment": 50, "normalize": 30, "verify": 10},
            auto_hide=True
        )
        
        # Control buttons
        config_buttons = create_save_reset_buttons(
            save_label="Simpan Config", 
            reset_label="Reset Config",
            save_tooltip="Simpan konfigurasi augmentation",
            reset_tooltip="Reset ke konfigurasi default"
        )
        
        # Action buttons dengan proper mapping
        action_buttons = create_action_buttons(
            primary_label="🎯 Run Augmentation", 
            primary_icon="play",
            secondary_buttons=[("🔍 Check Dataset", "search", "info")],
            cleanup_enabled=True, 
            button_width="180px"
        )
        
        # Output areas
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', margin='8px 0', max_height='200px', overflow='auto'
        ))
        
        # Log components
        log_components = create_log_accordion('augmentation', '200px')
        
        # Layout assembly dengan responsive design
        settings_container = widgets.VBox([
            _create_two_column_layout(basic_options['container'], advanced_options['container']),
            augmentation_types['container'],
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ], layout=widgets.Layout(width='100%'))
        
        # Main UI assembly
        ui = widgets.VBox([
            header,
            status_panel,
            settings_container,
            widgets.HTML("<hr style='margin: 15px 0; border: 1px solid #e0e0e0;'>"),
            action_buttons['container'],
            confirmation_area,
            progress_tracker['container'],
            log_components['log_accordion']
        ], layout=widgets.Layout(width='100%'))
        
        # CRITICAL: Comprehensive component mapping untuk handlers
        ui_components = {
            'ui': ui,
            'header': header,
            'status_panel': status_panel,
            'confirmation_area': confirmation_area,
            
            # CRITICAL: Proper widget mappings dari form components
            'num_variations': basic_options['widgets']['num_variations'],
            'target_count': basic_options['widgets']['target_count'],
            'output_prefix': basic_options['widgets']['output_prefix'],
            'balance_classes': basic_options['widgets']['balance_classes'],
            
            # Advanced options
            'fliplr': advanced_options['widgets']['fliplr'],
            'degrees': advanced_options['widgets']['degrees'],
            'translate': advanced_options['widgets']['translate'],
            'scale': advanced_options['widgets']['scale'],
            'hsv_h': advanced_options['widgets']['hsv_h'],
            'hsv_s': advanced_options['widgets']['hsv_s'],
            'brightness': advanced_options['widgets']['brightness'],
            'contrast': advanced_options['widgets']['contrast'],
            
            # Augmentation types
            'augmentation_types': augmentation_types['widgets']['augmentation_types'],
            'target_split': augmentation_types['widgets']['target_split'],
            
            # CRITICAL: Button mappings dengan proper action names
            'augment_button': action_buttons['download_button'],  # Primary action
            'check_button': action_buttons['check_button'],       # Secondary action
            'cleanup_button': action_buttons.get('cleanup_button'), # Cleanup action
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            
            # Progress tracking dengan new API compatibility
            'progress_tracker': progress_tracker['tracker'],
            'progress_container': progress_tracker['container'],
            'update_overall': progress_tracker.get('update_overall'),
            'update_step': progress_tracker.get('update_step'),
            'update_current': progress_tracker.get('update_current'),
            'complete_operation': progress_tracker.get('complete_operation'),
            'error_operation': progress_tracker.get('error_operation'),
            'reset_all': progress_tracker.get('reset_all'),
            
            # Log outputs
            'log_output': log_components['log_output'],
            'status': log_components['log_output'],  # Compatibility alias
            
            # Metadata untuk initializer
            'module_name': 'augmentation',
            'logger_namespace': 'smartcash.ui.dataset.augmentation',
            'config': config or {}
        }
        
        return ui_components
        
    except Exception as e:
        # Fallback UI minimal jika terjadi error
        return _create_fallback_ui(str(e))

def _create_basic_options_safe() -> Dict[str, Any]:
    """Create basic options widget dengan safe defaults"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        return create_basic_options_widget()
    except ImportError:
        return _create_basic_options_fallback()

def _create_advanced_options_safe() -> Dict[str, Any]:
    """Create advanced options widget dengan safe defaults"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        return create_advanced_options_widget()
    except ImportError:
        return _create_advanced_options_fallback()

def _create_augmentation_types_safe() -> Dict[str, Any]:
    """Create augmentation types widget dengan safe defaults"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        return create_augmentation_types_widget()
    except ImportError:
        return _create_augmentation_types_fallback()

def _create_basic_options_fallback() -> Dict[str, Any]:
    """Fallback basic options jika import gagal"""
    num_variations = widgets.IntSlider(value=3, min=1, max=10, description='Jumlah Variasi:')
    target_count = widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Target Count:')
    output_prefix = widgets.Text(value='aug', description='Output Prefix:')
    balance_classes = widgets.Checkbox(value=True, description='Balance Classes')
    
    container = widgets.VBox([
        widgets.HTML("<h6>📋 Opsi Dasar</h6>"),
        num_variations, target_count, output_prefix, balance_classes
    ])
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        }
    }

def _create_advanced_options_fallback() -> Dict[str, Any]:
    """Fallback advanced options jika import gagal"""
    fliplr = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, description='Flip:')
    degrees = widgets.IntSlider(value=10, min=0, max=30, description='Rotasi:')
    translate = widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Translasi:')
    scale = widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Skala:')
    hsv_h = widgets.FloatSlider(value=0.015, min=0.0, max=0.05, description='HSV H:')
    hsv_s = widgets.FloatSlider(value=0.7, min=0.0, max=1.0, description='HSV S:')
    brightness = widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Brightness:')
    contrast = widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Contrast:')
    
    container = widgets.VBox([
        widgets.HTML("<h6>⚙️ Opsi Lanjutan</h6>"),
        fliplr, degrees, translate, scale, hsv_h, hsv_s, brightness, contrast
    ])
    
    return {
        'container': container,
        'widgets': {
            'fliplr': fliplr, 'degrees': degrees, 'translate': translate, 'scale': scale,
            'hsv_h': hsv_h, 'hsv_s': hsv_s, 'brightness': brightness, 'contrast': contrast
        }
    }

def _create_augmentation_types_fallback() -> Dict[str, Any]:
    """Fallback augmentation types jika import gagal"""
    augmentation_types = widgets.SelectMultiple(
        options=[('Combined', 'combined'), ('Position', 'position'), ('Lighting', 'lighting')],
        value=['combined'],
        description='Types:'
    )
    target_split = widgets.Dropdown(
        options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
        value='train',
        description='Split:'
    )
    
    container = widgets.VBox([
        widgets.HTML("<h6>🔄 Jenis Augmentasi</h6>"),
        augmentation_types, target_split
    ])
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types,
            'target_split': target_split
        }
    }

def _create_two_column_layout(left_widget, right_widget):
    """Create two column layout dengan responsive design"""
    return widgets.HBox([
        widgets.Box([left_widget], layout=widgets.Layout(width='48%', margin='0 1% 0 0')),
        widgets.Box([right_widget], layout=widgets.Layout(width='48%'))
    ], layout=widgets.Layout(width='100%'))

def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """Create fallback UI jika terjadi error critical"""
    error_widget = widgets.HTML(f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #dc3545; 
                border-radius: 5px; color: #721c24; margin: 10px 0;">
        <h4>⚠️ Augmentation UI Error</h4>
        <p><strong>Error:</strong> {error_message}</p>
        <p>💡 <strong>Fix:</strong> Restart cell atau check imports</p>
    </div>
    """)
    
    # Minimal working widgets
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
        'error': error_message,
        'fallback_mode': True
    }