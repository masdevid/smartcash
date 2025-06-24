"""
File: smartcash/ui/setup/env_config/components/env_info_panel.py
Deskripsi: Component untuk menampilkan informasi environment dan Colab
"""

import ipywidgets as widgets
from smartcash.ui.setup.env_config.utils.env_detector import detect_environment_info

def create_env_info_panel() -> widgets.HTML:
    """
    🖥️ Buat panel informasi environment dan Colab
    
    Returns:
        Environment info panel HTML widget
    """
    # Detect environment info
    env_info = detect_environment_info()
    
    return widgets.HTML(
        value=_format_env_info_content(env_info),
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            margin='10px 0',
            background='#f9f9f9'
        )
    )

def _format_env_info_content(env_info: dict) -> str:
    """📊 Format environment info menjadi HTML"""
    python_version = env_info.get('python_version', 'N/A')
    platform = env_info.get('platform', 'N/A')
    is_colab = env_info.get('is_colab', False)
    gpu_info = env_info.get('gpu_info', 'N/A')
    drive_mounted = env_info.get('drive_mounted', False)
    
    colab_status = "✅ Detected" if is_colab else "❌ Not detected"
    drive_status = "✅ Mounted" if drive_mounted else "❌ Not mounted"
    
    return f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h5 style="color: #333; margin-bottom: 10px; border-bottom: 2px solid #2196f3; padding-bottom: 5px;">
                    🐍 Python & Platform
                </h5>
                <p><strong>Python Version:</strong> {python_version}</p>
                <p><strong>Platform:</strong> {platform}</p>
                <p><strong>Environment:</strong> {_get_env_type(env_info)}</p>
                <p><strong>Google Colab:</strong> {colab_status}</p>
            </div>
            
            <div>
                <h5 style="color: #333; margin-bottom: 10px; border-bottom: 2px solid #4caf50; padding-bottom: 5px;">
                    ⚙️ System Resources
                </h5>
                <p><strong>CPU Cores:</strong> {env_info.get('cpu_cores', 'N/A')}</p>
                <p><strong>Total RAM:</strong> {env_info.get('total_ram', 'N/A')}</p>
                <p><strong>Storage:</strong> {env_info.get('storage_info', 'N/A')}</p>
                <p><strong>GPU:</strong> {gpu_info}</p>
            </div>
        </div>
    </div>
    """

def _get_env_type(env_info: dict) -> str:
    """🔍 Determine environment type"""
    if env_info.get('is_colab'):
        return "Google Colab"
    return "Local"

def _get_storage_status(env_info: dict) -> str:
    """💾 Get storage status"""
    if env_info.get('drive_mounted'):
        return "Google Drive"
    return "Local"