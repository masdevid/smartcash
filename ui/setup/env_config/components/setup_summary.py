"""
File: smartcash/ui/setup/env_config/components/setup_summary.py
Deskripsi: Component untuk menampilkan summary hasil setup environment
"""

import ipywidgets as widgets
from typing import Dict

def create_setup_summary() -> widgets.HTML:
    """
    📋 Buat panel untuk setup summary
    
    Returns:
        Setup summary HTML widget
    """
    return widgets.HTML(
        value=_get_initial_summary_content(),
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #ddd',
            border_radius='4px',
            margin='10px 0'
        )
    )

def update_setup_summary(summary_widget: widgets.HTML, summary_data: Dict) -> None:
    """
    🔄 Update setup summary dengan data baru
    
    Args:
        summary_widget: Setup summary HTML widget
        summary_data: Data summary yang akan ditampilkan
    """
    try:
        content = _format_summary_content(summary_data)
        summary_widget.value = content
    except Exception as e:
        summary_widget.value = f"<p style='color: red;'>❌ Error updating summary: {str(e)}</p>"

def _get_initial_summary_content() -> str:
    """📝 Generate initial summary content"""
    return """
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <h4 style="color: #2196f3; margin-top: 0; margin-bottom: 15px;">📋 Setup Summary</h4>
        <p style="color: #888; font-style: italic;">Waiting for setup to complete...</p>
    </div>
    """

def _format_summary_content(data: Dict) -> str:
    """📊 Format summary data untuk drive mount, config sync, symlinks, dan folders"""
    drive_mounted = data.get('drive_mounted', False)
    mount_path = data.get('mount_path', 'N/A')
    configs_synced = data.get('configs_synced', 0)
    symlinks_created = data.get('symlinks_created', 0)
    folders_created = data.get('folders_created', 0)
    
    drive_status = "✅ Mounted" if drive_mounted else "❌ Not mounted"
    mount_info = f"at {mount_path}" if drive_mounted else ""
    
    return f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <h4 style="color: #2196f3; margin-top: 0; margin-bottom: 15px;">📋 Setup Summary</h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #2196f3;">
                <h5 style="margin: 0 0 8px 0; color: #333;">💾 Drive Status</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{drive_status}</strong> {mount_info}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #4caf50;">
                <h5 style="margin: 0 0 8px 0; color: #333;">⚙️ Configs Synced</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{configs_synced}</strong> config files</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #ff9800;">
                <h5 style="margin: 0 0 8px 0; color: #333;">🔗 Symlinks Created</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{symlinks_created}</strong> symlinks</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #9c27b0;">
                <h5 style="margin: 0 0 8px 0; color: #333;">📁 Folders Created</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{folders_created}</strong> directories</p>
            </div>
        </div>
    </div>
    """