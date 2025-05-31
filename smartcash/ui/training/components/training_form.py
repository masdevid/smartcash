"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Fixed training form components dengan proper error handling dan progress integration
"""

import ipywidgets as widgets
from typing import Dict, Any


def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form components dengan proper fallback handling"""
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.status_panel import create_status_panel
        
        # Training control buttons
        control_buttons = create_training_control_buttons()
        
        # Progress tracking dengan proper container
        progress_components = create_progress_tracking_container()
        
        # Log components dengan existing log_accordion
        log_components = create_log_accordion('training', height='250px')
        
        # Status panel untuk training feedback
        status_panel = create_status_panel("🧠 EfficientNet-B4 siap untuk training", "info")
        
        # Critical display components dengan safe layout
        display_components = create_display_components()
        
        # Info display untuk configuration summary
        info_display = create_info_display(config)
        
        return {
            # Control buttons
            **control_buttons,
            
            # Progress tracking
            'progress_container': progress_components.get('container'),
            'progress_tracker': progress_components.get('tracker'),
            'status_panel': status_panel,
            
            # Display components
            **display_components,
            
            # Log components
            'log_output': log_components.get('log_output'),
            'log_accordion': log_components.get('log_accordion'),
            
            # Info display
            'info_display': info_display,
            
            # Config reference
            'config': config.get('training', {})
        }
        
    except Exception as e:
        # Simple fallback untuk prevent complete failure
        return create_fallback_training_form(str(e))


def create_training_control_buttons() -> Dict[str, Any]:
    """Create training control buttons using existing action_buttons"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    
    # Primary button (Start Training)
    primary_buttons = create_action_buttons(
        primary_label="🚀 Mulai Training",
        primary_icon="",
        primary_style='success',
        secondary_buttons=[],
        cleanup_enabled=False,
        button_width='160px'
    )
    
    # Secondary control buttons
    stop_button = widgets.Button(
        description="⏹️ Stop Training",
        button_style='danger',
        tooltip='Hentikan training dan simpan checkpoint',
        disabled=True,
        layout=widgets.Layout(width='140px', height='35px', margin='2px')
    )
    
    reset_button = widgets.Button(
        description="🔄 Reset Metrics", 
        button_style='warning',
        tooltip='Reset training metrics dan chart',
        layout=widgets.Layout(width='140px', height='35px', margin='2px')
    )
    
    validate_button = widgets.Button(
        description="🔍 Cek Model",
        button_style='info', 
        tooltip='Validasi model readiness',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    cleanup_button = widgets.Button(
        description="🧹 Cleanup GPU",
        button_style='',
        tooltip='Bersihkan GPU memory',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    # Container layout
    buttons_row_1 = widgets.HBox([primary_buttons['download_button'], stop_button, reset_button])
    buttons_row_2 = widgets.HBox([validate_button, cleanup_button])
    
    button_container = widgets.VBox([
        buttons_row_1, buttons_row_2
    ], layout=widgets.Layout(
        width='100%', align_items='center', border='1px solid #dee2e6',
        border_radius='8px', padding='15px', margin='10px 0',
        background_color='#f8f9fa'
    ))
    
    return {
        'start_button': primary_buttons['download_button'],  # Map to start_button
        'stop_button': stop_button,
        'reset_button': reset_button, 
        'validate_button': validate_button,
        'cleanup_button': cleanup_button,
        'button_container': button_container
    }


def create_display_components() -> Dict[str, Any]:
    """Create display components dengan proper styling"""
    
    # Model readiness display
    model_readiness_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #28a745', padding='12px',
        margin='10px 0', max_height='200px', overflow_y='auto',
        background_color='#f8fff8', border_radius='6px'
    ))
    
    # Training config display
    training_config_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #007bff', padding='12px',
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#f8f9ff', border_radius='6px'
    ))
    
    # GPU status display
    gpu_status_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #ffc107', padding='12px',
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#fffef8', border_radius='6px'
    ))
    
    # Metrics display
    metrics_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='280px', margin='10px 0',
        border='1px solid #17a2b8', padding='12px',
        overflow_y='auto', border_radius='6px'
    ))
    
    # Chart display
    chart_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='420px', margin='10px 0',
        border='1px solid #6f42c1', padding='12px',
        overflow_y='auto', border_radius='6px'
    ))
    
    return {
        'model_readiness_display': model_readiness_display,
        'training_config_display': training_config_display,
        'gpu_status_display': gpu_status_display,
        'metrics_output': metrics_output,
        'chart_output': chart_output
    }


def create_info_display(config: Dict[str, Any]) -> widgets.HTML:
    """Create comprehensive info display untuk configuration summary dengan detailed monitoring"""
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    model_optimization = config.get('model_optimization', {})
    paths_config = config.get('paths', {})
    data_config = config.get('data', {})
    augmentation_config = config.get('augmentation', {})
    
    # Current timestamp untuk tracking updates
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    info_html = f"""
    <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        <h4 style="color: #495057; margin: 0 0 15px 0;">🧠 Training Configuration Monitor</h4>
        <p style="font-size: 11px; color: #6c757d; margin: 0 0 15px 0;">Last Updated: {timestamp}</p>
        
        <!-- Model Configuration -->
        <div style="margin-bottom: 15px; padding: 10px; background: #e3f2fd; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #1976d2;">🏗️ Model Architecture</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>Model Type:</b> <span style="color: #1976d2;">{training_config.get('model_type', 'efficient_optimized')}</span></li>
                <li><b>Backbone:</b> <span style="color: #1976d2;">{training_config.get('backbone', model_config.get('backbone', 'efficientnet_b4'))}</span></li>
                <li><b>Detection Layers:</b> <span style="color: #1976d2;">{', '.join(training_config.get('detection_layers', ['banknote']))}</span></li>
                <li><b>Total Classes:</b> <span style="color: #1976d2;">{training_config.get('num_classes', model_config.get('num_classes', 7))}</span></li>
                <li><b>Image Size:</b> <span style="color: #1976d2;">{training_config.get('image_size', model_config.get('input_size', [640, 640]))}</span></li>
            </ul>
        </div>
        
        <!-- Training Parameters -->
        <div style="margin-bottom: 15px; padding: 10px; background: #e8f5e8; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #2e7d32;">⚙️ Training Parameters</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>Epochs:</b> <span style="color: #2e7d32;">{training_config.get('epochs', 100)}</span></li>
                <li><b>Batch Size:</b> <span style="color: #2e7d32;">{training_config.get('batch_size', 16)}</span></li>
                <li><b>Learning Rate:</b> <span style="color: #2e7d32;">{training_config.get('learning_rate', 0.001)}</span></li>
                <li><b>Optimizer:</b> <span style="color: #2e7d32;">{training_config.get('optimizer', 'Adam')}</span></li>
                <li><b>Weight Decay:</b> <span style="color: #2e7d32;">{training_config.get('weight_decay', 0.0005)}</span></li>
                <li><b>Use Mixed Precision:</b> <span style="color: #2e7d32;">{training_config.get('use_mixed_precision', True)}</span></li>
            </ul>
        </div>
        
        <!-- Model Optimizations -->
        <div style="margin-bottom: 15px; padding: 10px; background: #fff3e0; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #ef6c00;">🚀 Model Optimizations</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>FeatureAdapter:</b> <span style="color: {'#2e7d32' if model_optimization.get('use_attention', True) else '#d32f2f'};">{'✅ Enabled' if model_optimization.get('use_attention', True) else '❌ Disabled'}</span></li>
                <li><b>ResidualAdapter:</b> <span style="color: {'#2e7d32' if model_optimization.get('use_residual', True) else '#d32f2f'};">{'✅ Enabled' if model_optimization.get('use_residual', True) else '❌ Disabled'}</span></li>
                <li><b>CIoU Loss:</b> <span style="color: {'#2e7d32' if model_optimization.get('use_ciou', True) else '#d32f2f'};">{'✅ Enabled' if model_optimization.get('use_ciou', True) else '❌ Disabled'}</span></li>
                <li><b>Transfer Learning:</b> <span style="color: {'#2e7d32' if model_config.get('transfer_learning', True) else '#d32f2f'};">{'✅ Enabled' if model_config.get('transfer_learning', True) else '❌ Disabled'}</span></li>
                <li><b>Pretrained Weights:</b> <span style="color: {'#2e7d32' if model_config.get('pretrained', True) else '#d32f2f'};">{'✅ Used' if model_config.get('pretrained', True) else '❌ Not Used'}</span></li>
            </ul>
        </div>
        
        <!-- Data Configuration -->
        <div style="margin-bottom: 15px; padding: 10px; background: #f3e5f5; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #7b1fa2;">📊 Data Configuration</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>Data Source:</b> <span style="color: #7b1fa2;">{data_config.get('source', 'roboflow')}</span></li>
                <li><b>Train Split:</b> <span style="color: #7b1fa2;">{data_config.get('split_ratios', {}).get('train', 0.7) * 100:.0f}%</span></li>
                <li><b>Valid Split:</b> <span style="color: #7b1fa2;">{data_config.get('split_ratios', {}).get('valid', 0.15) * 100:.0f}%</span></li>
                <li><b>Test Split:</b> <span style="color: #7b1fa2;">{data_config.get('split_ratios', {}).get('test', 0.15) * 100:.0f}%</span></li>
                <li><b>Stratified Split:</b> <span style="color: {'#2e7d32' if data_config.get('stratified_split', True) else '#d32f2f'};">{'✅ Yes' if data_config.get('stratified_split', True) else '❌ No'}</span></li>
            </ul>
        </div>
        
        <!-- Augmentation Configuration -->
        <div style="margin-bottom: 15px; padding: 10px; background: #fce4ec; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #c2185b;">🔄 Augmentation Settings</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>Augmentation:</b> <span style="color: {'#2e7d32' if augmentation_config.get('enabled', True) else '#d32f2f'};">{'✅ Enabled' if augmentation_config.get('enabled', True) else '❌ Disabled'}</span></li>
                <li><b>Variations:</b> <span style="color: #c2185b;">{augmentation_config.get('num_variations', 2)}</span></li>
                <li><b>Types:</b> <span style="color: #c2185b;">{', '.join(augmentation_config.get('types', ['position', 'lighting']))}</span></li>
                <li><b>Process Bboxes:</b> <span style="color: {'#2e7d32' if augmentation_config.get('process_bboxes', True) else '#d32f2f'};">{'✅ Yes' if augmentation_config.get('process_bboxes', True) else '❌ No'}</span></li>
            </ul>
        </div>
        
        <!-- Paths Configuration -->
        <div style="margin-bottom: 10px; padding: 10px; background: #e0f2f1; border-radius: 6px;">
            <h5 style="margin: 0 0 8px 0; color: #00695c;">📁 Paths & Storage</h5>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                <li><b>Data YAML:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('data_yaml', 'data/currency_dataset.yaml')}</span></li>
                <li><b>Checkpoint Dir:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('checkpoint_dir', 'runs/train/checkpoints')}</span></li>
                <li><b>Tensorboard Dir:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('tensorboard_dir', 'runs/tensorboard')}</span></li>
            </ul>
        </div>
        
        <!-- Config Source Info -->
        <div style="padding: 8px; background: #fff8e1; border-radius: 4px; border-left: 4px solid #ff8f00;">
            <small style="color: #e65100; font-weight: 500;">
                💡 Configuration akan otomatis update saat config modules lain diubah
            </small>
        </div>
    </div>
    """
    
    return widgets.HTML(value=info_html)


def create_fallback_training_form(error_msg: str) -> Dict[str, Any]:
    """Simple fallback training form untuk error cases"""
    
    # Basic buttons dengan minimal functionality
    start_button = widgets.Button(description="🚀 Mulai Training", button_style='success')
    stop_button = widgets.Button(description="⏹️ Stop", button_style='danger', disabled=True)
    reset_button = widgets.Button(description="🔄 Reset", button_style='warning')
    
    # Basic container
    button_container = widgets.HBox([start_button, stop_button, reset_button])
    
    # Error status
    status_panel = widgets.HTML(f"""
    <div style="padding: 10px; background: #ffeaea; border-radius: 6px; color: #d32f2f;">
        ❌ Error creating training form: {error_msg}
    </div>
    """)
    
    # Basic outputs
    log_output = widgets.Output()
    progress_container = widgets.VBox([])
    
    return {
        'start_button': start_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'button_container': button_container,
        'status_panel': status_panel,
        'log_output': log_output,
        'progress_container': progress_container,
        'info_display': widgets.HTML("Training form error"),
        'model_readiness_display': widgets.Output(),
        'training_config_display': widgets.Output(),
        'gpu_status_display': widgets.Output(),
        'metrics_output': widgets.Output(),
        'chart_output': widgets.Output(),
        'config': {}
    }


# One-liner utilities
update_info_display = lambda info_widget, config: setattr(info_widget, 'value', create_info_display(config).value)
enable_training_mode = lambda ui: [setattr(ui.get(btn), 'disabled', False) for btn in ['start_button', 'reset_button']] and setattr(ui.get('stop_button'), 'disabled', True)
enable_stopping_mode = lambda ui: setattr(ui.get('start_button'), 'disabled', True) or setattr(ui.get('stop_button'), 'disabled', False)