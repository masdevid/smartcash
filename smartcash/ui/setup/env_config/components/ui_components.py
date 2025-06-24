# File: smartcash/ui/setup/env_config/components/ui_components.py
# Deskripsi: UI components creation (refactored from existing)

import ipywidgets as widgets
from typing import Dict, Any

def create_env_config_ui() -> Dict[str, Any]:
    """Create complete environment config UI"""
    # Header
    header = widgets.HTML(
        value="<h2>🚀 SmartCash Environment Configuration</h2>",
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # Setup button
    setup_button = widgets.Button(
        description="🚀 Setup Environment",
        button_style='primary',
        layout=widgets.Layout(width='200px', margin='10px auto')
    )
    
    # Status panel
    status_panel = widgets.HTML(
        value="<p>🔍 Memeriksa status environment...</p>",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Progress components
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, layout=widgets.Layout(width='100%'))
    progress_text = widgets.HTML(value="Siap untuk setup")
    
    # Log accordion dengan auto-open
    log_output = widgets.HTML(value="", layout=widgets.Layout(height='200px', overflow='auto'))
    log_accordion = widgets.Accordion(children=[log_output])
    log_accordion.set_title(0, "📋 Setup Logs")
    log_accordion.selected_index = 0  # Auto-open
    
    # Summary panel
    summary_panel = widgets.HTML(
        value="<p>Loading system info...</p>",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Tips panel
    tips_html = """
    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
        <h4>💡 Tips & Requirements</h4>
        <ul>
            <li>Pastikan Google Drive memiliki ruang minimal 12GB</li>
            <li>Setup akan membuat folder struktur data lengkap</li>
            <li>Proses setup memerlukan waktu 1-2 menit</li>
        </ul>
    </div>
    """
    tips_panel = widgets.HTML(value=tips_html)
    
    # Layout assembly
    main_layout = widgets.VBox([
        header,
        widgets.HBox([setup_button], layout=widgets.Layout(justify_content='center')),
        status_panel,
        progress_bar,
        progress_text,
        log_accordion,
        summary_panel,
        tips_panel
    ])
    
    return {
        'ui': main_layout,
        'setup_button': setup_button,
        'status_panel': status_panel,
        'progress_bar': progress_bar,
        'progress_text': progress_text,
        'log_accordion': log_accordion,
        'log_output': log_output,
        'summary_panel': summary_panel
    }