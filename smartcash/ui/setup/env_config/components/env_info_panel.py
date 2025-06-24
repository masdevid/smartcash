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
            border='1px solid #ddd',
            border_radius='4px',
            margin='10px 0'
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
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
            <div>
                <h5 style="color: #333; margin-bottom: 10px; border-bottom: 2px solid #2196f3; padding-bottom: 5px;">
                    🐍 Python & Platform
                </h5>
                <p><strong>Python Version:</strong> {python_version}</p>
                <p><strong>Platform:</strong> {platform}</p>
                <p><strong>GPU:</strong> {gpu_info}</p>
            </div>
            
            <div>
                <h5 style="color: #333; margin-bottom: 10px; border-bottom: 2px solid #4caf50; padding-bottom: 5px;">
                    🔧 Colab Status
                </h5>
                <p><strong>Google Colab:</strong> {colab_status}</p>
                <p><strong>Drive Mounted:</strong> {drive_status}</p>
                <p><strong>Runtime Type:</strong> {env_info.get('runtime_type', 'N/A')}</p>
            </div>
        </div>
        
        <div style="background: #f5f5f5; border-radius: 4px; padding: 15px; margin-top: 15px;">
            <h5 style="color: #666; margin-top: 0;">📋 Environment Summary</h5>
            <ul style="margin: 10px 0; padding-left: 20px; color: #555;">
                <li>Environment: <strong>{_get_env_type(env_info)}</strong></li>
                <li>GPU Acceleration: <strong>{_get_gpu_status(env_info)}</strong></li>
                <li>Storage Access: <strong>{_get_storage_status(env_info)}</strong></li>
            </ul>
        </div>
        
        {_get_recommendations_section(env_info)}
    </div>
    """

def _get_env_type(env_info: dict) -> str:
    """🔍 Determine environment type"""
    if env_info.get('is_colab'):
        return "Google Colab"
    return "Local Environment"

def _get_gpu_status(env_info: dict) -> str:
    """🎮 Get GPU status"""
    gpu_info = env_info.get('gpu_info', '')
    if 'CUDA' in gpu_info or 'Tesla' in gpu_info:
        return "Available"
    return "Not available"

def _get_storage_status(env_info: dict) -> str:
    """💾 Get storage status"""
    if env_info.get('drive_mounted'):
        return "Google Drive mounted"
    return "Local storage only"

def _get_recommendations_section(env_info: dict) -> str:
    """💡 Generate recommendations based on environment"""
    recommendations = []
    
    if not env_info.get('is_colab'):
        recommendations.append("Consider using Google Colab for better GPU access")
    
    if env_info.get('is_colab') and not env_info.get('drive_mounted'):
        recommendations.append("Mount Google Drive for persistent storage")
    
    if 'CUDA' not in env_info.get('gpu_info', ''):
        recommendations.append("Enable GPU runtime for faster training")
    
    if not recommendations:
        return """
        <div style="background: #e8f5e8; border-left: 4px solid #4caf50; padding: 10px; margin-top: 15px;">
            <p style="margin: 0; color: #2e7d32;"><strong>✅ Environment Optimal</strong></p>
            <p style="margin: 0; color: #2e7d32;">Your environment is configured optimally for training.</p>
        </div>
        """
    
    return """
    <div style="background: #e8f5e8; border-left: 4px solid #4caf50; padding: 10px; margin-top: 15px;">
        <p style="margin: 0; color: #2e7d32;"><strong>💡 Recommendations</strong></p>
        <ul style="margin: 10px 0; padding-left: 20px; color: #555;">
            {recommendations}
        </ul>
    </div>
    """.format(
        recommendations="".join([f"<li>{rec}</li>" for rec in recommendations])
    )