# File: smartcash/ui/setup/env_config/handlers/system_info_handler.py
# Deskripsi: Handler untuk system information collection dan display

import psutil
import platform
from typing import Dict, Any

class SystemInfoHandler:
    """🖥️ Handler untuk informasi sistem Colab"""
    
    def update_system_info_panel(self, ui_components: Dict[str, Any]):
        """Update panel summary dengan informasi sistem"""
        try:
            gpu_info = self._get_gpu_info()
            sys_info = self._get_system_info()
            
            summary_html = self._format_system_info(gpu_info, sys_info)
            
            if 'summary_panel' in ui_components:
                ui_components['summary_panel'].value = summary_html
                
        except Exception as e:
            self._set_fallback_summary(ui_components, str(e))
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Dapatkan info GPU dengan PyTorch detection"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                return {'available': True, 'name': gpu_name, 'memory_gb': gpu_memory}
            else:
                return {'available': False, 'name': 'None', 'memory_gb': 0}
        except ImportError:
            return {'available': False, 'name': 'PyTorch not available', 'memory_gb': 0}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Dapatkan info sistem lengkap"""
        try:
            return {
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'ram_gb': psutil.virtual_memory().total // (1024**3),
                'disk_free_gb': psutil.disk_usage('/').free // (1024**3),
                'python_version': platform.python_version(),
                'platform': platform.platform()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _format_system_info(self, gpu_info: Dict[str, Any], sys_info: Dict[str, Any]) -> str:
        """Format system info ke HTML"""
        gpu_status = f"🎮 GPU: {gpu_info['name']} ({gpu_info['memory_gb']}GB)" if gpu_info['available'] else "🎮 GPU: Tidak tersedia"
        cpu_info = f"🖥️ CPU: {sys_info.get('cpu_cores', 'N/A')} cores, {sys_info.get('cpu_threads', 'N/A')} threads"
        ram_info = f"🧠 RAM: {sys_info.get('ram_gb', 'N/A')} GB"
        disk_info = f"💾 Disk: {sys_info.get('disk_free_gb', 'N/A')} GB tersedia"
        python_info = f"🐍 Python: {sys_info.get('python_version', 'N/A')}"
        
        return f"""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; font-family: monospace;">
            <h4 style="margin-top: 0; color: #2c3e50;">📊 Informasi Sistem Colab</h4>
            <div style="line-height: 1.6;">
                {gpu_status}<br>
                {cpu_info}<br>
                {ram_info}<br>
                {disk_info}<br>
                {python_info}
            </div>
        </div>
        """
    
    def _set_fallback_summary(self, ui_components: Dict[str, Any], error_msg: str):
        """Set fallback summary jika gagal mengambil info"""
        fallback_html = f"""
        <div style="background: #fff3cd; padding: 10px; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #856404;">⚠️ Info Sistem</h4>
            <p>Tidak dapat mengambil detail sistem: {error_msg}</p>
        </div>
        """
        if 'summary_panel' in ui_components:
            ui_components['summary_panel'].value = fallback_html