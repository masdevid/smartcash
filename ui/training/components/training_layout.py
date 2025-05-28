"""
File: smartcash/ui/training/components/training_layout.py
Deskripsi: Layout arrangement untuk training UI menggunakan responsive design
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_divider, create_responsive_container
from smartcash.ui.utils.constants import ICONS, COLORS


def create_training_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive training layout dengan consolidated components"""
    
    # Header dengan description
    header = create_header(
        f"{ICONS.get('training', '🚀')} Model Training",
        "Latih model YOLOv5 dengan EfficientNet-B4 backbone untuk deteksi mata uang"
    )
    
    # Info section untuk menampilkan config summary
    info_section = create_responsive_container([
        widgets.HTML(f"<h4>{ICONS.get('info', 'ℹ️')} Informasi Konfigurasi</h4>"),
        form_components['info_display']
    ])
    
    # Control section dengan buttons
    control_section = create_responsive_container([
        widgets.HTML(f"<h4>{ICONS.get('action', '⚙️')} Kontrol Training</h4>"),
        form_components['button_container']
    ])
    
    # Progress section
    progress_section = create_responsive_container([
        widgets.HTML(f"<h4>{ICONS.get('progress', '📊')} Progress Training</h4>"),
        form_components['progress_container'],
        form_components['status_panel']
    ])
    
    # Metrics section dengan chart dan metrics
    metrics_section = create_responsive_container([
        widgets.HTML(f"<h4>{ICONS.get('chart', '📈')} Metrik & Visualisasi</h4>"),
        form_components['chart_output'],
        form_components['metrics_output']
    ])
    
    # Log section
    log_section = create_responsive_container([
        widgets.HTML(f"<h4>{ICONS.get('log', '📋')} Training Logs</h4>"),
        form_components['log_accordion']
    ])
    
    # Main container dengan responsive layout
    main_container = create_responsive_container([
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
    ], container_type="vbox", padding="15px")
    
    # Update form components dengan layout containers
    form_components.update({
        'main_container': main_container,
        'ui': main_container,  # Untuk compatibility
        'header': header,
        'info_section': info_section,
        'control_section': control_section,
        'progress_section': progress_section,
        'metrics_section': metrics_section,
        'log_section': log_section
    })
    
    return form_components