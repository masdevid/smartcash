"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Fixed progress tracker subscription error dan proper component mapping
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI assembly dengan fixed progress tracker dan proper mapping"""
    
    try:
        # Import components
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
        
        # FIXED: Progress tracker dengan proper dictionary access
        progress_result = create_triple_progress_tracker(
            operation="Dataset Augmentation",
            steps=["prepare", "augment", "normalize", "verify"],
            step_weights={"prepare": 10, "augment": 50, "normalize": 30, "verify": 10},
            auto_hide=True
        )
        
        # Ensure progress_result is a dict
        if not isinstance(progress_result, dict):
            progress_result = {'container': progress_result, 'tracker': progress_result}
        
        # Config buttons
        config_buttons = create_save_reset_buttons(
            save_label="Simpan Config", 
            reset_label="Reset Config"
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
        
        # Layout assembly
        settings_container = widgets.VBox([
            _create_two_column_layout(basic_options['container'], advanced_options['container']),
            augmentation_types['container'],
            widgets.Box([config_buttons['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ])
        
        # Main UI
        ui = widgets.VBox([
            header,
            status_panel,
            settings_container,
            widgets.HTML("<hr style='margin: 15px 0;'>"),
            action_buttons['container'],
            confirmation_area,
            progress_result['container'],  # FIXED: Use dict access
            log_components['log_accordion']
        ])
        
        # FIXED: Component mapping dengan proper progress tracker access
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
            
            # FIXED: Progress tracker dengan proper dict access
            'progress_tracker': progress_result.get('tracker'),
            'progress_container': progress_result.get('container'),
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

def _create_basic_options_safe() -> Dict[str, Any]:
    """Basic options dengan inline fallback"""
    try:
        from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
        return create_basic_options_widget()
    except ImportError:
        # Inline fallback
        widgets_dict = {
            'num_variations': widgets.IntSlider(value=3, min=1, max=10, description='Variasi:'),
            'target_count': widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Target:'),
            'output_prefix': widgets.Text(value='aug', description='Prefix:'),
            'balance_classes': widgets.Checkbox(value=True, description='Balance Classes')
        }
        container = widgets.VBox([widgets.HTML("<h6>📋 Opsi Dasar</h6>")] + list(widgets_dict.values()))
        return {'container': container, 'widgets': widgets_dict}

def _create_advanced_options_safe() -> Dict[str, Any]:
    """Advanced options dengan inline fallback"""
    try:
        from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
        return create_advanced_options_widget()
    except ImportError:
        # Inline fallback
        widgets_dict = {
            'fliplr': widgets.FloatSlider(value=0.5, min=0.0, max=1.0, description='Flip:'),
            'degrees': widgets.IntSlider(value=10, min=0, max=30, description='Rotasi:'),
            'translate': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Translasi:'),
            'scale': widgets.FloatSlider(value=0.1, min=0.0, max=0.25, description='Skala:'),
            'hsv_h': widgets.FloatSlider(value=0.015, min=0.0, max=0.05, description='HSV H:'),
            'hsv_s': widgets.FloatSlider(value=0.7, min=0.0, max=1.0, description='HSV S:'),
            'brightness': widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Brightness:'),
            'contrast': widgets.FloatSlider(value=0.2, min=0.0, max=0.4, description='Contrast:')
        }
        container = widgets.VBox([widgets.HTML("<h6>⚙️ Opsi Lanjutan</h6>")] + list(widgets_dict.values()))
        return {'container': container, 'widgets': widgets_dict}

def _create_augmentation_types_safe() -> Dict[str, Any]:
    """Augmentation types dengan inline fallback"""
    try:
        from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
        return create_augmentation_types_widget()
    except ImportError:
        # Inline fallback
        widgets_dict = {
            'augmentation_types': widgets.SelectMultiple(
                options=[('Combined', 'combined'), ('Position', 'position'), ('Lighting', 'lighting')],
                value=['combined'], description='Types:'
            ),
            'target_split': widgets.Dropdown(
                options=[('Train', 'train'), ('Valid', 'valid'), ('Test', 'test')],
                value='train', description='Split:'
            )
        }
        container = widgets.VBox([widgets.HTML("<h6>🔄 Jenis Augmentasi</h6>")] + list(widgets_dict.values()))
        return {'container': container, 'widgets': widgets_dict}

def _create_two_column_layout(left_widget, right_widget):
    """Two column responsive layout"""
    return widgets.HBox([
        widgets.Box([left_widget], layout=widgets.Layout(width='48%', margin='0 1% 0 0')),
        widgets.Box([right_widget], layout=widgets.Layout(width='48%'))
    ])

def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """Minimal fallback UI"""
    error_widget = widgets.HTML(f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #dc3545; 
                border-radius: 5px; color: #721c24;">
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