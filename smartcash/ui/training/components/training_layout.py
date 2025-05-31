"""\nFile: smartcash/ui/training/components/training_layout.py\nDeskripsi: Layout arrangement dengan tabs untuk konfigurasi dan accordion untuk metrics\n"""

import ipywidgets as widgets
from typing import Dict, Any


def create_training_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive training layout dengan safe container handling"""
    try:
        # Header section
        header = create_header_section()
        
        # Info section dengan config tabs
        info_section = create_info_section(form_components)
        
        # Control section dengan buttons
        control_section = create_control_section(form_components)
        
        # Progress section
        progress_section = create_progress_section(form_components)
        
        # Metrics section dengan accordion
        metrics_section = create_metrics_section(form_components)
        
        # Log section
        log_section = create_log_section(form_components)
        
        # Main container dengan safe layout
        main_container = widgets.VBox([
            header,
            info_section,
            create_divider(),
            control_section,
            create_divider(),
            progress_section,
            create_divider(),
            metrics_section,
            create_divider(),
            log_section
        ], layout=widgets.Layout(
            width='100%', padding='15px', overflow_y='auto'
        ))
        
        # Cek tampilan duplikat
        if hasattr(main_container, '_displayed') and main_container._displayed:
            return form_components
        
        main_container._displayed = True
        
        # Update form components
        form_components.update({
            'main_container': main_container,
            'ui': main_container,
            'header': header,
            'info_section': info_section,
            'control_section': control_section,
            'progress_section': progress_section,
            'metrics_section': metrics_section,
            'log_section': log_section
        })
        
        return form_components
        
    except Exception as e:
        return create_fallback_layout(form_components, str(e))


def create_header_section() -> widgets.HTML:
    """Create header section dengan description"""
    return widgets.HTML(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; font-size: 24px; font-weight: bold;">🚀 Model Training</h2>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 14px;">
            Latih model YOLOv5 dengan EfficientNet-B4 backbone untuk deteksi mata uang
        </p>
    </div>
    """)


def create_info_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create info section dengan config tabs"""
    return widgets.VBox([
        widgets.HTML("<h4>ℹ️ Informasi Konfigurasi</h4>"),
        form_components.get('config_tabs', form_components.get('info_display', widgets.HTML("Info tidak tersedia")))
    ], layout=widgets.Layout(margin='10px 0'))


def create_control_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create control section dengan buttons"""
    return widgets.VBox([
        widgets.HTML("<h4>⚙️ Kontrol Training</h4>"),
        form_components.get('button_container', widgets.HTML("Buttons tidak tersedia"))
    ], layout=widgets.Layout(margin='10px 0'))


def create_progress_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create progress section dengan status panel (default hidden)"""
    # Ambil progress container dan set default hidden
    progress_container = form_components.get('progress_container', widgets.HTML("Progress tidak tersedia"))
    
    # Set visibility ke hidden by default jika adalah widget yang valid
    if hasattr(progress_container, 'layout'):
        progress_container.layout.display = 'none'
    
    # Return layout
    return widgets.VBox([
        widgets.HTML("<h4>📊 Progress Training</h4>"),
        progress_container,
        form_components.get('status_panel', widgets.HTML("Status tidak tersedia"))
    ], layout=widgets.Layout(margin='10px 0'))


def create_metrics_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create metrics section dengan metrics accordion"""
    return widgets.VBox([
        form_components.get('metrics_accordion', widgets.VBox([
            form_components.get('chart_output', widgets.Output()),
            form_components.get('metrics_output', widgets.Output())
        ]))
    ], layout=widgets.Layout(margin='10px 0'))


def create_log_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create log section dengan training logs"""
    return widgets.VBox([
        widgets.HTML("<h4>📋 Training Logs</h4>"),
        form_components.get('log_accordion', form_components.get('log_output', widgets.Output()))
    ], layout=widgets.Layout(margin='10px 0'))


def create_divider() -> widgets.HTML:
    """Create visual divider"""
    return widgets.HTML("""
    <div style="height: 1px; background: linear-gradient(to right, transparent, #ddd, transparent); 
                margin: 15px 0;"></div>
    """)


def create_fallback_layout(form_components: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create fallback layout untuk error cases"""
    
    # Simple error display
    error_display = widgets.HTML(f"""
    <div style="padding: 20px; background: #ffeaea; border-radius: 8px; text-align: center;">
        <h3>❌ Layout Error</h3>
        <p>Error creating training layout: {error_msg}</p>
        <p>Using basic fallback layout...</p>
    </div>
    """)
    
    # Basic vertical layout dari form components
    available_components = [error_display]
    
    # Add available components safely
    safe_components = ['button_container', 'status_panel', 'log_output', 'chart_output']
    [available_components.append(form_components[comp]) for comp in safe_components if comp in form_components]
    
    # Simple container
    main_container = widgets.VBox(available_components, layout=widgets.Layout(
        width='100%', padding='15px'
    ))
    
    form_components.update({
        'main_container': main_container,
        'ui': main_container,
        'error': error_msg
    })
    
    return form_components


# One-liner utilities
get_layout_height = lambda: "800px"
create_responsive_width = lambda: "100%"
safe_get_component = lambda form_components, key, fallback=None: form_components.get(key, fallback or widgets.HTML(f"{key} tidak tersedia"))