"""
File: smartcash/ui/setup/env_config/helpers/system_info_helper.py
Deskripsi: System info helper dengan constants integration dan one-liner optimizations
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

from smartcash.ui.setup.env_config.constants import DRIVE_MOUNT_POINT
from smartcash.ui.setup.env_config.utils import is_colab_environment

class SystemInfoHelper:
    """System info helper dengan constants integration"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system info dengan one-liner pattern"""
        info = {}
        info.update(SystemInfoHelper._get_basic_environment_info())
        info.update(SystemInfoHelper._get_hardware_info())
        info.update(SystemInfoHelper._get_software_info())
        info.update(SystemInfoHelper._get_storage_info())
        info.update(SystemInfoHelper._get_connectivity_info())
        return info
    
    @staticmethod
    def _get_basic_environment_info() -> Dict[str, Any]:
        """Basic environment info dengan constants"""
        info = {
            'python_version': sys.version.split()[0],
            'python_executable': sys.executable,
            'platform': sys.platform,
            'current_directory': os.getcwd(),
            'is_colab': is_colab_environment(),
            'environment': 'Google Colab' if is_colab_environment() else 'Local/Jupyter'
        }
        return info
    
    @staticmethod
    def _get_hardware_info() -> Dict[str, Any]:
        """Hardware info dengan one-liner optimizations"""
        info = {'cpu_count': os.cpu_count() or 1}
        
        # Memory info dengan one-liner
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.update({
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'available_memory_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory.percent
            })
        except ImportError:
            info['memory_info'] = 'psutil not available'
        
        # GPU info dengan one-liner
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            info.update({
                'cuda_available': cuda_available,
                'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
                'cuda_device_name': torch.cuda.get_device_name(0) if cuda_available else 'N/A',
                'cuda_version': torch.version.cuda if cuda_available else 'Not Available'
            })
        except ImportError:
            info.update({'cuda_available': False, 'cuda_version': 'PyTorch not installed'})
        
        return info
    
    @staticmethod
    def _get_software_info() -> Dict[str, Any]:
        """Software packages info dengan one-liner pattern"""
        critical_packages = {
            'torch': 'PyTorch', 'torchvision': 'TorchVision', 'ultralytics': 'YOLO',
            'cv2': 'OpenCV', 'albumentations': 'Augmentation', 'roboflow': 'Dataset',
            'yaml': 'Config', 'tqdm': 'Progress', 'pandas': 'Data Processing',
            'numpy': 'NumPy', 'matplotlib': 'Plotting'
        }
        
        available_packages = {}
        missing_packages = []
        
        for package, display_name in critical_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                available_packages[display_name] = version
            except ImportError:
                missing_packages.append(display_name)
        
        return {
            'available_packages': available_packages,
            'missing_packages': missing_packages,
            'packages_status': (len(available_packages), len(critical_packages))
        }
    
    @staticmethod
    def _get_storage_info() -> Dict[str, Any]:
        """Storage info dengan constants integration"""
        info = {}
        
        try:
            current_path = Path.cwd()
            usage = shutil.disk_usage(current_path)
            
            info.update({
                'disk_total_gb': round(usage.total / (1024**3), 2),
                'disk_free_gb': round(usage.free / (1024**3), 2),
                'disk_used_gb': round(usage.used / (1024**3), 2),
                'disk_usage_percent': round((usage.used / usage.total) * 100, 1)
            })
            
            # Drive specific info dengan constants
            drive_path = Path(DRIVE_MOUNT_POINT)
            if drive_path.exists():
                try:
                    drive_usage = shutil.disk_usage(drive_path)
                    info.update({
                        'drive_free_gb': round(drive_usage.free / (1024**3), 2),
                        'drive_total_gb': round(drive_usage.total / (1024**3), 2)
                    })
                except Exception:
                    info['drive_status'] = 'Access limited'
            
        except Exception as e:
            info['disk_info_error'] = str(e)
        
        return info
    
    @staticmethod
    def _get_connectivity_info() -> Dict[str, Any]:
        """Connectivity info dengan constants"""
        info = {}
        drive_path = Path(DRIVE_MOUNT_POINT)
        
        # Drive mount status dengan constants
        if drive_path.exists():
            info.update({
                'drive_mounted': True,
                'drive_path': str(drive_path)
            })
            
            # Test write access
            try:
                test_file = drive_path / '.smartcash_test'
                test_file.write_text('test')
                test_file.unlink()
                info['drive_writable'] = True
            except Exception:
                info['drive_writable'] = False
        else:
            info['drive_mounted'] = False
        
        # Internet connectivity test
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=3)
            info['internet_available'] = True
        except Exception:
            info['internet_available'] = False
        
        return info
    
    @staticmethod
    def format_system_summary(info: Dict[str, Any]) -> List[str]:
        """Format system summary dengan one-liner optimizations"""
        summary_lines = []
        
        # Environment dengan one-liner
        env_emoji = "🔬" if info.get('is_colab') else "💻"
        summary_lines.append(f"🌍 <strong>Platform:</strong> {env_emoji} {info.get('environment', 'Unknown')}")
        
        # Python info
        summary_lines.append(f"🐍 <strong>Python:</strong> {info.get('python_version', 'N/A')}")
        
        # Hardware summary dengan one-liner pattern
        cpu_count = info.get('cpu_count', 'N/A')
        if 'available_memory_gb' in info:
            memory_gb = info['available_memory_gb']
            memory_color = "#4CAF50" if memory_gb > 8 else "#FF9800" if memory_gb > 4 else "#F44336"
            summary_lines.append(f"🧠 <span style='color:{memory_color}'><strong>Memory:</strong></span> {memory_gb:.1f}GB / {info.get('total_memory_gb', 0):.1f}GB")
        
        summary_lines.append(f"⚡ <strong>CPU Cores:</strong> {cpu_count}")
        
        # GPU info dengan one-liner
        if info.get('cuda_available'):
            gpu_name = info.get('cuda_device_name', 'Unknown GPU')[:35]
            summary_lines.append(f"🎮 <span style='color:#4CAF50'><strong>GPU:</strong></span> ✅ {gpu_name}")
        else:
            summary_lines.append("🎮 <span style='color:#F44336'><strong>GPU:</strong></span> ❌ CPU Only")
        
        # Storage info dengan one-liner
        if 'disk_free_gb' in info:
            free_gb = info['disk_free_gb']
            storage_color = "#4CAF50" if free_gb > 20 else "#FF9800" if free_gb > 10 else "#F44336"
            summary_lines.append(f"💾 <span style='color:{storage_color}'><strong>Storage:</strong></span> {free_gb:.1f}GB free")
        
        # Drive status dengan constants integration
        if info.get('is_colab'):
            if info.get('drive_mounted'):
                drive_status = "✅ Mounted"
                if info.get('drive_writable'):
                    drive_status += " & Writable"
                else:
                    drive_status += " (Read-Only)"
                summary_lines.append(f"📂 <span style='color:#4CAF50'><strong>Google Drive:</strong></span> {drive_status}")
            else:
                summary_lines.append("📂 <span style='color:#F44336'><strong>Google Drive:</strong></span> ❌ Not Mounted")
        
        # Packages summary dengan one-liner
        if 'packages_status' in info:
            available, total = info['packages_status']
            package_color = "#4CAF50" if available >= total * 0.8 else "#FF9800" if available >= total * 0.5 else "#F44336"
            summary_lines.append(f"📦 <span style='color:{package_color}'><strong>Packages:</strong></span> {available}/{total} available")
        
        # Connectivity dengan one-liner
        if info.get('internet_available'):
            summary_lines.append("🌐 <strong>Internet:</strong> ✅ Connected")
        else:
            summary_lines.append("🌐 <span style='color:#F44336'><strong>Internet:</strong></span> ❌ No Connection")
        
        return summary_lines
    
    @staticmethod
    def get_system_recommendations(info: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Get system recommendations dengan one-liner pattern"""
        recommendations = []
        
        # One-liner recommendations dengan condition mapping
        recommendation_conditions = [
            (info.get('available_memory_gb', 0) < 4, ("⚠️", "Memory rendah - pertimbangkan reduce batch size")),
            (not info.get('cuda_available'), ("💡", "GPU tidak tersedia - training akan lebih lambat")),
            (info.get('disk_free_gb', 0) < 5, ("🚨", "Storage hampir penuh - bersihkan file tidak perlu")),
            (info.get('is_colab') and not info.get('drive_mounted'), ("📂", "Mount Google Drive untuk data persistence")),
            (len(info.get('missing_packages', [])) > 3, ("📦", f"Install {len(info.get('missing_packages', []))} missing packages untuk fitur lengkap"))
        ]
        
        # Add recommendations dengan one-liner filter
        recommendations.extend([rec for condition, rec in recommendation_conditions if condition])
        
        return recommendations