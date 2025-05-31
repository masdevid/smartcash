"""
File: smartcash/ui/training/components/fallback_component.py
Deskripsi: Komponen fallback untuk menangani error pada training UI
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_fallback_component(error_msg: str) -> Dict[str, Any]:
    """Create basic fallback component untuk error cases"""
    error_html = widgets.HTML(f"""
    <div style="padding: 10px; background: #ffeaea; border-radius: 6px; color: #d32f2f; margin: 10px 0;">
        ❌ <strong>Terjadi error:</strong> {error_msg}
    </div>
    """)
    
    fallback_container = widgets.VBox([
        widgets.HTML("<h3>⚠️ Error pada SmartCash Training UI</h3>"),
        error_html
    ])
    
    return {
        'error_message': error_html,
        'fallback_container': fallback_container
    }


def create_fallback_layout(components: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create fallback layout untuk error cases"""
    fallback = create_fallback_component(error_msg)
    
    header_section = widgets.HTML("""
    <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h2>💰 SmartCash Training Module</h2>
        <p>⚠️ <strong>Fallback Mode:</strong> Beberapa komponen tidak dapat dimuat</p>
    </div>
    """)
    
    error_section = widgets.VBox([
        fallback['fallback_container'],
        widgets.HTML("""
        <div style="padding: 10px; background: #e8f4fd; border-radius: 6px; margin: 10px 0;">
            <p>ℹ️ <strong>Saran:</strong> Coba refresh notebook atau periksa log error untuk diagnosis lebih lanjut.</p>
        </div>
        """)
    ])
    
    main_container = widgets.VBox([
        header_section,
        error_section
    ])
    
    return {
        'header_section': header_section,
        'error_section': error_section,
        'main_container': main_container,
        **fallback,
        **components
    }


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
    
    # Create fallback components
    fallback = create_fallback_component(error_msg)
    
    # Buat main container - penting untuk daftar komponen kritis
    main_container = widgets.VBox([
        widgets.HTML(f"""<div style="text-align: center; padding: 10px; background: #ffecb3; border-radius: 5px;">
            <h3>⚠️ SmartCash Training - Fallback Mode</h3>
            <p>{error_msg}</p>
        </div>"""),
        button_container,
        status_panel,
        log_output
    ])
    
    return {
        'main_container': main_container,  # Komponen kritis yang harus ada
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
        'config': {},
        **fallback
    }
