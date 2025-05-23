"""
File: smartcash/ui/setup/env_config/helpers/system_info_helper.py
Deskripsi: Helper untuk mengumpulkan dan memformat informasi sistem yang informatif
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

class SystemInfoHelper:
    """Helper untuk mengumpulkan informasi sistem yang komprehensif"""
    
    @staticmethod
    def get_enhanced_system_info() -> Dict[str, Any]:
        """Dapatkan informasi sistem yang diperluas dengan detail yang berguna"""
        info = {}
        
        # Basic environment info
        info.update(SystemInfoHelper._get_basic_environment_info())
        
        # Hardware info
        info.update(SystemInfoHelper._get_hardware_info())
        
        # Software/Package info
        info.update(SystemInfoHelper._get_software_info())
        
        # Storage info
        info.update(SystemInfoHelper._get_storage_info())
        
        # Network/Connectivity info
        info.update(SystemInfoHelper._get_connectivity_info())
        
        return info
    
    @staticmethod
    def _get_basic_environment_info() -> Dict[str, Any]:
        """Informasi dasar environment"""
        info = {
            'python_version': sys.version.split()[0],
            'python_executable': sys.executable,
            'platform': sys.platform,
            'current_directory': os.getcwd()
        }
        
        # Detect environment type
        try:
            import google.colab
            info['environment'] = 'Google Colab'
            info['is_colab'] = True
        except ImportError:
            info['environment'] = 'Local/Jupyter'
            info['is_colab'] = False
        
        return info
    
    @staticmethod
    def _get_hardware_info() -> Dict[str, Any]:
        """Informasi hardware dan resource"""
        info = {}
        
        # CPU info
        info['cpu_count'] = os.cpu_count() or 1
        
        # Memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = round(memory.total / (1024**3), 2)
            info['available_memory_gb'] = round(memory.available / (1024**3), 2)
            info['memory_percent'] = memory.percent
        except ImportError:
            info['memory_info'] = 'psutil not available'
        
        # GPU info
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
            else:
                info['cuda_version'] = 'Not Available'
        except ImportError:
            info['cuda_available'] = False
            info['cuda_version'] = 'PyTorch not installed'
        
        return info
    
    @staticmethod
    def _get_software_info() -> Dict[str, Any]:
        """Informasi software dan packages yang terinstall"""
        info = {}
        
        # Python packages yang critical
        critical_packages = {
            'torch': 'PyTorch',
            'torchvision': 'TorchVision',
            'ultralytics': 'YOLO',
            'cv2': 'OpenCV',
            'albumentations': 'Augmentation',
            'roboflow': 'Dataset',
            'yaml': 'Config',
            'tqdm': 'Progress',
            'pandas': 'Data Processing',
            'numpy': 'NumPy',
            'matplotlib': 'Plotting'
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
        
        info['available_packages'] = available_packages
        info['missing_packages'] = missing_packages
        info['packages_status'] = len(available_packages), len(critical_packages)
        
        return info
    
    @staticmethod
    def _get_storage_info() -> Dict[str, Any]:
        """Informasi storage dan disk space"""
        info = {}
        
        try:
            # Current directory disk usage
            current_path = Path.cwd()
            usage = shutil.disk_usage(current_path)
            
            info['disk_total_gb'] = round(usage.total / (1024**3), 2)
            info['disk_free_gb'] = round(usage.free / (1024**3), 2)
            info['disk_used_gb'] = round(usage.used / (1024**3), 2)
            info['disk_usage_percent'] = round((usage.used / usage.total) * 100, 1)
            
            # Drive specific info jika Colab
            if Path('/content/drive/MyDrive').exists():
                try:
                    drive_usage = shutil.disk_usage('/content/drive/MyDrive')
                    info['drive_free_gb'] = round(drive_usage.free / (1024**3), 2)
                    info['drive_total_gb'] = round(drive_usage.total / (1024**3), 2)
                except Exception:
                    info['drive_status'] = 'Access limited'
            
        except Exception as e:
            info['disk_info_error'] = str(e)
        
        return info
    
    @staticmethod
    def _get_connectivity_info() -> Dict[str, Any]:
        """Informasi konektivitas dan akses"""
        info = {}
        
        # Google Drive mount status
        if Path('/content/drive/MyDrive').exists():
            info['drive_mounted'] = True
            info['drive_path'] = '/content/drive/MyDrive'
            
            # Test write access
            try:
                test_file = Path('/content/drive/MyDrive/.smartcash_test')
                test_file.write_text('test')
                test_file.unlink()
                info['drive_writable'] = True
            except Exception:
                info['drive_writable'] = False
        else:
            info['drive_mounted'] = False
        
        # Internet connectivity test (simple)
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=3)
            info['internet_available'] = True
        except Exception:
            info['internet_available'] = False
        
        return info
    
    @staticmethod
    def format_system_summary(info: Dict[str, Any]) -> List[str]:
        """Format system info menjadi summary lines yang informatif"""
        summary_lines = []
        
        # Environment
        env_emoji = "🔬" if info.get('is_colab') else "💻"
        summary_lines.append(f"🌍 <strong>Platform:</strong> {env_emoji} {info.get('environment', 'Unknown')}")
        
        # Python info
        summary_lines.append(f"🐍 <strong>Python:</strong> {info.get('python_version', 'N/A')}")
        
        # Hardware summary
        cpu_count = info.get('cpu_count', 'N/A')
        if 'available_memory_gb' in info:
            memory_gb = info['available_memory_gb']
            memory_color = "#4CAF50" if memory_gb > 8 else "#FF9800" if memory_gb > 4 else "#F44336"
            summary_lines.append(f"🧠 <span style='color:{memory_color}'><strong>Memory:</strong></span> {memory_gb:.1f}GB / {info.get('total_memory_gb', 0):.1f}GB")
        
        summary_lines.append(f"⚡ <strong>CPU Cores:</strong> {cpu_count}")
        
        # GPU info
        if info.get('cuda_available'):
            gpu_name = info.get('cuda_device_name', 'Unknown GPU')[:35]
            summary_lines.append(f"🎮 <span style='color:#4CAF50'><strong>GPU:</strong></span> ✅ {gpu_name}")
        else:
            summary_lines.append("🎮 <span style='color:#F44336'><strong>GPU:</strong></span> ❌ CPU Only")
        
        # Storage info
        if 'disk_free_gb' in info:
            free_gb = info['disk_free_gb']
            storage_color = "#4CAF50" if free_gb > 20 else "#FF9800" if free_gb > 10 else "#F44336"
            summary_lines.append(f"💾 <span style='color:{storage_color}'><strong>Storage:</strong></span> {free_gb:.1f}GB free")
        
        # Drive status
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
        
        # Packages summary
        if 'packages_status' in info:
            available, total = info['packages_status']
            package_color = "#4CAF50" if available >= total * 0.8 else "#FF9800" if available >= total * 0.5 else "#F44336"
            summary_lines.append(f"📦 <span style='color:{package_color}'><strong>Packages:</strong></span> {available}/{total} available")
        
        # Connectivity
        if info.get('internet_available'):
            summary_lines.append("🌐 <strong>Internet:</strong> ✅ Connected")
        else:
            summary_lines.append("🌐 <span style='color:#F44336'><strong>Internet:</strong></span> ❌ No Connection")
        
        return summary_lines
    
    @staticmethod
    def get_system_recommendations(info: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Dapatkan rekomendasi berdasarkan system info"""
        recommendations = []
        
        # Memory recommendations
        if info.get('available_memory_gb', 0) < 4:
            recommendations.append(("⚠️", "Memory rendah - pertimbangkan reduce batch size"))
        
        # GPU recommendations
        if not info.get('cuda_available'):
            recommendations.append(("💡", "GPU tidak tersedia - training akan lebih lambat"))
        
        # Storage recommendations
        if info.get('disk_free_gb', 0) < 5:
            recommendations.append(("🚨", "Storage hampir penuh - bersihkan file tidak perlu"))
        
        # Drive recommendations
        if info.get('is_colab') and not info.get('drive_mounted'):
            recommendations.append(("📂", "Mount Google Drive untuk data persistence"))
        
        # Package recommendations
        if info.get('missing_packages'):
            missing_count = len(info['missing_packages'])
            if missing_count > 3:
                recommendations.append(("📦", f"Install {missing_count} missing packages untuk fitur lengkap"))
        
        return recommendations