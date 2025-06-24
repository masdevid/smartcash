"""
File: smartcash/ui/setup/env_config/handlers/system_info_handler.py
Deskripsi: Handler untuk system information display dengan pemisahan yang tepat antara environment dan system info
"""

from typing import Dict, Any
from smartcash.common.environment import get_environment_manager

class SystemInfoHandler:
    """🖥️ Handler untuk system information display"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_manager = get_environment_manager(logger=logger)
        
    def generate_environment_summary(self, env_status: Dict[str, Any]) -> str:
        """📋 Generate environment summary dengan status yang benar"""
        try:
            # Refresh environment manager untuk status terbaru
            self.env_manager.refresh_drive_status()
            
            # Status items untuk environment summary (kiri)
            python_status = "✅ Ready"
            drive_status = "✅ Connected" if self.env_manager.is_drive_mounted else "❌ Not Connected"
            configs_status = "✅ Complete" if env_status.get('configs_complete', False) else "❌ Incomplete" 
            directories_status = "✅ Ready" if env_status.get('directories_ready', False) else "❌ Not Ready"
            
            # Drive path info jika tersedia
            drive_path_info = ""
            if self.env_manager.is_drive_mounted and self.env_manager.drive_path:
                drive_path_info = f"<li>💾 Drive Path: <span style='color: #6c757d; font-family: monospace;'>{self.env_manager.drive_path}</span></li>"
            
            summary_html = f"""
            <div style='background: linear-gradient(135deg, #e8f5e8, #f0f8f0); padding: 12px; border-radius: 8px; border-left: 4px solid #28a745;'>
                <h4 style='margin: 0 0 8px 0; color: #155724;'>🌍 Environment Status</h4>
                <ul style='margin: 4px 0; padding-left: 20px; font-size: 13px;'>
                    <li>Python Environment: <span style='color: #28a745; font-weight: 500;'>{python_status}</span></li>
                    <li>Google Drive: <span style='color: {"#28a745" if self.env_manager.is_drive_mounted else "#dc3545"}; font-weight: 500;'>{drive_status}</span></li>
                    {drive_path_info}
                    <li>Configurations: <span style='color: {"#28a745" if env_status.get("configs_complete") else "#dc3545"}; font-weight: 500;'>{configs_status}</span></li>
                    <li>Directory Structure: <span style='color: {"#28a745" if env_status.get("directories_ready") else "#dc3545"}; font-weight: 500;'>{directories_status}</span></li>
                </ul>
            </div>
            """
            
            return summary_html
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Could not generate environment summary: {str(e)}")
            return self._get_fallback_environment_summary()
    
    def generate_colab_system_info(self) -> str:
        """🖥️ Generate system info khusus untuk Colab (kanan)"""
        try:
            system_info = self.env_manager.get_system_info()
            
            # System specs
            python_version = system_info.get('python_version', 'Unknown')
            cuda_available = system_info.get('cuda_available', False)
            cuda_device = system_info.get('cuda_device_name', 'Not Available')
            total_memory = system_info.get('total_memory_gb', 'Unknown')
            available_memory = system_info.get('available_memory_gb', 'Unknown')
            
            # GPU info
            gpu_info = ""
            if cuda_available:
                gpu_info = f"<li>🎮 GPU: <span style='color: #28a745; font-weight: 500;'>{cuda_device}</span></li>"
            else:
                gpu_info = "<li>🎮 GPU: <span style='color: #dc3545;'>Not Available</span></li>"
            
            # Memory info
            memory_info = ""
            if isinstance(total_memory, (int, float)) and isinstance(available_memory, (int, float)):
                memory_usage_pct = round((1 - available_memory/total_memory) * 100, 1)
                memory_color = "#28a745" if memory_usage_pct < 70 else "#ffc107" if memory_usage_pct < 85 else "#dc3545"
                memory_info = f"""
                <li>💾 Memory: <span style='color: {memory_color}; font-weight: 500;'>{available_memory:.1f}GB / {total_memory:.1f}GB available ({memory_usage_pct}% used)</span></li>
                """
            else:
                memory_info = "<li>💾 Memory: <span style='color: #6c757d;'>Information not available</span></li>"
            
            system_html = f"""
            <div style='background: linear-gradient(135deg, #e8f4f8, #f0f8ff); padding: 12px; border-radius: 8px; border-left: 4px solid #007bff;'>
                <h4 style='margin: 0 0 8px 0; color: #0c5460;'>🖥️ System Information</h4>
                <ul style='margin: 4px 0; padding-left: 20px; font-size: 13px;'>
                    <li>🐍 Python: <span style='color: #007bff; font-weight: 500;'>{python_version}</span></li>
                    {gpu_info}
                    {memory_info}
                    <li>☁️ Platform: <span style='color: #007bff; font-weight: 500;'>Google Colab</span></li>
                </ul>
            </div>
            """
            
            return system_html
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Could not generate system info: {str(e)}")
            return self._get_fallback_system_info()
    
    def _get_fallback_environment_summary(self) -> str:
        """Fallback environment summary jika terjadi error"""
        return """
        <div style='background: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107;'>
            <h4 style='margin: 0 0 8px 0; color: #856404;'>🌍 Environment Status</h4>
            <p style='margin: 4px 0; color: #856404; font-size: 13px;'>⚠️ Status information temporarily unavailable</p>
        </div>
        """
    
    def _get_fallback_system_info(self) -> str:
        """Fallback system info jika terjadi error"""
        return """
        <div style='background: #d1ecf1; padding: 12px; border-radius: 8px; border-left: 4px solid #bee5eb;'>
            <h4 style='margin: 0 0 8px 0; color: #0c5460;'>🖥️ System Information</h4>
            <p style='margin: 4px 0; color: #0c5460; font-size: 13px;'>⚠️ System information temporarily unavailable</p>
        </div>
        """