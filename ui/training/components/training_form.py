"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Training form components fokus pada persiapan dan eksekusi training
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS


def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form components fokus pada training preparation dan execution"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.status_panel import create_status_panel
    
    training_config = config.get('training', {})
    
    # Training control buttons
    control_buttons_container = create_training_control_buttons()
    
    # Progress tracking components
    progress_components = create_progress_tracking_container()
    
    # Log accordion untuk training logs
    log_components = create_log_accordion('training', height='250px')
    
    # Status panel untuk training feedback
    status_panel = create_status_panel("🧠 EfficientNet-B4 siap untuk training", "info")
    
    # Model readiness display
    model_readiness_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #28a745', padding='12px', 
        margin='10px 0', max_height='200px', overflow_y='auto',
        background_color='#f8fff8', border_radius='6px'
    ))
    
    # Training configuration summary
    training_config_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #007bff', padding='12px', 
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#f8f9ff', border_radius='6px'
    ))
    
    # GPU status and cleanup results
    gpu_status_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #ffc107', padding='12px', 
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#fffef8', border_radius='6px'
    ))
    
    # Real-time metrics display
    metrics_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='280px', height='auto', 
        margin='10px 0', border='1px solid #17a2b8', 
        padding='12px', overflow_y='auto', border_radius='6px'
    ))
    
    # Training metrics chart
    chart_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='420px', margin='10px 0', 
        border='1px solid #6f42c1', padding='12px', 
        overflow_y='auto', border_radius='6px'
    ))
    
    return {
        # Training control buttons
        'start_button': control_buttons_container['start_button'],
        'stop_button': control_buttons_container['stop_button'],
        'reset_button': control_buttons_container['reset_button'],
        'validate_button': control_buttons_container['validate_button'],
        'cleanup_button': control_buttons_container['cleanup_button'],
        'button_container': control_buttons_container['container'],
        
        # Progress and status tracking
        'progress_container': progress_components['container'],
        'progress_bar': progress_components.get('tracker'),
        'status_panel': status_panel,
        
        # Information displays
        'model_readiness_display': model_readiness_display,
        'training_config_display': training_config_display,
        'gpu_status_display': gpu_status_display,
        
        # Training outputs
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'metrics_output': metrics_output,
        'chart_output': chart_output,
        
        # Config data
        'config': training_config
    }


def create_training_control_buttons() -> Dict[str, Any]:
    """Create training control buttons dengan specific functions"""
    
    # Main training button
    start_button = widgets.Button(
        description="🚀 Mulai Training",
        button_style='success',
        tooltip='Mulai training EfficientNet-B4 dengan konfigurasi saat ini',
        layout=widgets.Layout(width='160px', height='40px', margin='2px')
    )
    
    # Stop training button
    stop_button = widgets.Button(
        description="⏹️ Stop Training",
        button_style='danger',
        tooltip='Hentikan training dan simpan checkpoint',
        disabled=True,
        layout=widgets.Layout(width='140px', height='40px', margin='2px')
    )
    
    # Reset metrics button
    reset_button = widgets.Button(
        description="🔄 Reset Metrics",
        button_style='warning',
        tooltip='Reset training metrics dan chart',
        layout=widgets.Layout(width='140px', height='40px', margin='2px')
    )
    
    # Model validation button
    validate_button = widgets.Button(
        description="🔍 Cek Model",
        button_style='info',
        tooltip='Validasi model readiness dan pre-trained weights',
        layout=widgets.Layout(width='130px', height='40px', margin='2px')
    )
    
    # GPU cleanup button
    cleanup_button = widgets.Button(
        description="🧹 Cleanup GPU",
        button_style='secondary',
        tooltip='Bersihkan GPU memory dan reset model state',
        layout=widgets.Layout(width='130px', height='40px', margin='2px')
    )
    
    # Buttons container dengan responsive layout
    buttons_row_1 = widgets.HBox([
        start_button,
        stop_button,
        reset_button
    ], layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        margin='5px 0'
    ))
    
    buttons_row_2 = widgets.HBox([
        validate_button,
        cleanup_button
    ], layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        margin='5px 0'
    ))
    
    container = widgets.VBox([
        buttons_row_1,
        buttons_row_2
    ], layout=widgets.Layout(
        width='100%',
        align_items='center',
        border='1px solid #dee2e6',
        border_radius='8px',
        padding='15px',
        margin='10px 0',
        background_color='#f8f9fa'
    ))
    
    return {
        'start_button': start_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'validate_button': validate_button,
        'cleanup_button': cleanup_button,
        'container': container
    }


def create_training_info_section() -> widgets.VBox:
    """Create training information section"""
    
    # Training overview
    overview_html = widgets.HTML("""
    <div style="padding: 10px; background: #e3f2fd; border-radius: 6px; margin: 5px 0;">
        <h4 style="margin: 0 0 10px 0; color: #1976d2;">🧠 EfficientNet-B4 Training Overview</h4>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li><b>Architecture:</b> YOLOv5 + EfficientNet-B4 backbone</li>
            <li><b>Optimizations:</b> FeatureAdapter, ResidualAdapter, CIoU Loss</li>
            <li><b>Target:</b> Indonesian Rupiah currency detection</li>
            <li><b>Pre-trained:</b> ImageNet weights + YOLOv5 weights</li>
        </ul>
    </div>
    """)
    
    # Training preparation checklist
    checklist_html = widgets.HTML("""
    <div style="padding: 10px; background: #f3e5f5; border-radius: 6px; margin: 5px 0;">
        <h4 style="margin: 0 0 10px 0; color: #7b1fa2;">✅ Pre-Training Checklist</h4>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>🔍 Model architecture validation</li>
            <li>📦 Pre-trained weights availability</li>
            <li>🖥️ GPU memory status</li>
            <li>⚙️ Training configuration</li>
            <li>📊 Detection layers setup</li>
        </ul>
    </div>
    """)
    
    return widgets.VBox([
        overview_html,
        checklist_html
    ], layout=widgets.Layout(margin='10px 0'))


def get_training_tips() -> widgets.HTML:
    """Get training tips dan best practices"""
    return widgets.HTML("""
    <div style="padding: 12px; background: #fff3e0; border-radius: 6px; border-left: 4px solid #ff9800;">
        <h4 style="margin: 0 0 8px 0; color: #ef6c00;">💡 Training Tips</h4>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li><b>GPU Memory:</b> Monitor usage, cleanup jika diperlukan</li>
            <li><b>Learning Rate:</b> 0.001 optimal untuk EfficientNet-B4</li>
            <li><b>Batch Size:</b> Adjust berdasarkan GPU memory</li>
            <li><b>Epochs:</b> 100 epochs umumnya cukup untuk convergence</li>
            <li><b>Checkpoints:</b> Otomatis tersimpan setiap 10 epochs</li>
        </ul>
    </div>
    """)


# One-liner utilities untuk button management
enable_training_mode = lambda ui_components: setattr(ui_components.get('start_button'), 'disabled', False) or setattr(ui_components.get('stop_button'), 'disabled', True)
enable_stopping_mode = lambda ui_components: setattr(ui_components.get('start_button'), 'disabled', True) or setattr(ui_components.get('stop_button'), 'disabled', False)
reset_button_states = lambda ui_components: [setattr(ui_components.get(btn), 'disabled', False) for btn in ['start_button', 'reset_button', 'validate_button', 'cleanup_button']] or setattr(ui_components.get('stop_button'), 'disabled', True)