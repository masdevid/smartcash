"""
File: smartcash/ui/setup/env_config/handlers/system_info_handler.py
Deskripsi: Fixed handler untuk system information dengan dual column support
"""

import os
import sys
import platform
import psutil
from typing import Dict, Any

class SystemInfoHandler:
    """📊 Handler untuk system information dan Colab-specific info"""
    
    def generate_colab_system_info(self) -> str:
        """Generate HTML untuk informasi sistem Colab"""
        try:
            # Get system information
            system_info = self._get_system_info()
            
            # Get Colab-specific information
            colab_info = self._get_colab_info()
            
            # Generate HTML
            return self._format_colab_info_html(system_info, colab_info)
            
        except Exception as e:
            return f"<p style='color: #dc3545;'>❌ Error getting system info: {str(e)}</p>"
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'python_version': sys.version.split()[0],
                'platform': platform.system(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': self._format_bytes(memory.total),
                'memory_available': self._format_bytes(memory.available),
                'memory_percent': memory.percent,
                'disk_total': self._format_bytes(disk.total),
                'disk_free': self._format_bytes(disk.free),
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception:
            return {}
    
    def _get_colab_info(self) -> Dict[str, Any]:
        """Get Colab-specific information"""
        colab_info = {
            'is_colab': False,
            'gpu_available': False,
            'gpu_type': 'None',
            'tpu_available': False,
            'drive_mounted': False
        }
        
        try:
            # Check if running in Colab
            import google.colab
            colab_info['is_colab'] = True
            
            # Check GPU
            try:
                import torch
                if torch.cuda.is_available():
                    colab_info['gpu_available'] = True
                    colab_info['gpu_type'] = torch.cuda.get_device_name(0)
            except ImportError:
                pass
            
            # Check TPU
            try:
                import tensorflow as tf
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                colab_info['tpu_available'] = True
            except Exception:
                pass
            
            # Check Drive mount
            colab_info['drive_mounted'] = os.path.exists('/content/drive/MyDrive')
            
        except ImportError:
            # Not in Colab
            pass
        
        return colab_info
    
    def _format_colab_info_html(self, system_info: Dict[str, Any], 
                               colab_info: Dict[str, Any]) -> str:
        """Format system info sebagai HTML untuk panel kanan"""
        
        # Environment status
        env_status = "✅ Google Colab" if colab_info['is_colab'] else "❌ Local Environment"
        
        # GPU/TPU status
        gpu_status = "❌ No GPU"
        if colab_info['gpu_available']:
            gpu_status = f"✅ {colab_info['gpu_type']}"
        elif colab_info['tpu_available']:
            gpu_status = "✅ TPU Available"
        
        # Drive status
        drive_status = "✅ Mounted" if colab_info['drive_mounted'] else "❌ Not Mounted"
        
        # Memory status dengan color coding
        memory_percent = system_info.get('memory_percent', 0)
        memory_color = '#28a745' if memory_percent < 70 else '#ffc107' if memory_percent < 85 else '#dc3545'
        
        # Disk status dengan color coding
        disk_percent = system_info.get('disk_percent', 0)
        disk_color = '#28a745' if disk_percent < 70 else '#ffc107' if disk_percent < 85 else '#dc3545'
        
        html = f"""
        <div style="margin-bottom: 15px;">
            <h5 style="margin: 0 0 10px 0; color: #1976d2;">🌐 Runtime Environment</h5>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                <li>Environment: <strong>{env_status}</strong></li>
                <li>Python: <strong>{system_info.get('python_version', 'Unknown')}</strong></li>
                <li>CPU Cores: <strong>{system_info.get('cpu_count', 'Unknown')}</strong></li>
            </ul>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h5 style="margin: 0 0 10px 0; color: #1976d2;">🚀 Compute Resources</h5>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                <li>GPU/TPU: <strong>{gpu_status}</strong></li>
                <li>Memory: <strong style="color: {memory_color};">{system_info.get('memory_available', 'N/A')} / {system_info.get('memory_total', 'N/A')} ({memory_percent:.1f}%)</strong></li>
                <li>Storage: <strong style="color: {disk_color};">{system_info.get('disk_free', 'N/A')} free / {system_info.get('disk_total', 'N/A')} ({disk_percent:.1f}% used)</strong></li>
            </ul>
        </div>
        
        <div>
            <h5 style="margin: 0 0 10px 0; color: #1976d2;">💾 Google Drive</h5>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                <li>Status: <strong>{drive_status}</strong></li>
                <li>Path: <code style="font-size: 12px; background: #f1f3f4; padding: 2px 4px; border-radius: 3px;">/content/drive/MyDrive</code></li>
            </ul>
        </div>
        """
        
        return html
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes ke human readable format"""
        try:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_value < 1024.0:
                    return f"{bytes_value:.1f} {unit}"
                bytes_value /= 1024.0
            return f"{bytes_value:.1f} PB"
        except Exception:
            return "N/A"
    
    def update_system_info_panel(self, ui_components: Dict[str, Any]):
        """Update system info panel (legacy compatibility)"""
        # Legacy method - bisa digunakan untuk backward compatibility
        try:
            colab_info = self.generate_colab_system_info()
            if 'right_colab_panel' in ui_components:
                current_html = ui_components['right_colab_panel'].value
                # Update content part
                updated_html = current_html.replace(
                    '<p style="color: #1565c0;">Loading sistem Colab information...</p>',
                    colab_info
                )
                ui_components['right_colab_panel'].value = updated_html
        except Exception as e:
            # Silent fail untuk backward compatibility
            pass