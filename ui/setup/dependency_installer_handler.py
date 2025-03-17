"""
File: smartcash/ui/setup/dependency_installer_handler.py
Deskripsi: Handler untuk instalasi dependencies SmartCash
"""

import sys
import re
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
from IPython.display import display, clear_output, HTML

def _get_yolov5_requirements() -> List[str]:
    """Dapatkan requirements YOLOv5."""
    try:
        req_file = Path('yolov5/requirements.txt')
        if not req_file.exists():
            return ["matplotlib", "numpy", "opencv-python", "torch", "torchvision", "tqdm"]
        
        with open(req_file, 'r') as f:
            return [re.match(r'^([a-zA-Z0-9_\-]+)', line.strip()).group(1) 
                    for line in f if line.strip() and not line.startswith('#')]
    except Exception:
        return ["matplotlib", "numpy", "opencv-python", "torch", "torchvision", "tqdm"]

def setup_dependency_installer_handlers(ui_components: Dict, config: Dict[Any, Any]):
    """
    Setup handler untuk instalasi dependencies.
    
    Args:
        ui_components: Komponen UI untuk instalasi
        config: Konfigurasi dependencies
    
    Returns:
        Dictionary UI components yang telah ditambahkan handler
    """
    # Definisi package dan requirement
    PACKAGE_GROUPS = {
        'yolov5_req': _get_yolov5_requirements,
        'torch_req': f"{sys.executable} -m pip install torch torchvision torchaudio",
        'albumentations_req': f"{sys.executable} -m pip install albumentations",
        'notebook_req': f"{sys.executable} -m pip install ipywidgets tqdm",
        'smartcash_req': f"{sys.executable} -m pip install pyyaml termcolor python-dotenv",
        'opencv_req': f"{sys.executable} -m pip install opencv-python",
        'matplotlib_req': f"{sys.executable} -m pip install matplotlib seaborn",
        'pandas_req': f"{sys.executable} -m pip install pandas",
        'seaborn_req': f"{sys.executable} -m pip install seaborn"
    }

    PACKAGE_CHECKS = [
        ('PyTorch', 'torch'), ('TorchVision', 'torchvision'), 
        ('OpenCV', 'cv2'), ('Albumentations', 'albumentations'), 
        ('NumPy', 'numpy'), ('Pandas', 'pandas'), 
        ('Matplotlib', 'matplotlib'), ('Seaborn', 'seaborn'), 
        ('ipywidgets', 'ipywidgets'), ('tqdm', 'tqdm'), 
        ('PyYAML', 'yaml'), ('termcolor', 'termcolor')
    ]

    def _run_pip_install(cmd: str, package_name: str) -> Tuple[bool, str]:
        """Eksekusi perintah pip install."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            status = result.returncode == 0
            msg = f"✅ {package_name} berhasil diinstall" if status else f"❌ Gagal install {package_name}"
            return status, msg
        except Exception as e:
            return False, f"❌ Gagal install {package_name}: {str(e)}"

    

    def _check_package_status(package_checks: List[Tuple[str, str]]) -> None:
        """Periksa status paket yang terinstall."""
        for display_name, import_name in package_checks:
            try:
                module = __import__(import_name)
                display(HTML(f"<div style='color:green'>✅ {display_name} (v{module.__version__})</div>"))
            except ImportError:
                display(HTML(f"<div style='color:orange'>⚠️ {display_name} tidak terinstall</div>"))

    def _on_install_packages(b):
        """Handler untuk tombol install packages."""
        with ui_components['status']:
            clear_output()
            start_time = time.time()
            
            # Update UI state
            ui_components['install_progress'].layout.visibility = 'visible'
            ui_components['install_progress'].value = 0

            # Proses instalasi package yang dipilih
            total_packages = 0
            installed_packages = 0

            for pkg_key, pkg_cmd in PACKAGE_GROUPS.items():
                if not ui_components[pkg_key].value:
                    continue

                # Dapatkan command untuk instalasi
                cmd = pkg_cmd() if callable(pkg_cmd) else pkg_cmd
                display(HTML(f"📦 Memulai instalasi: {pkg_key}"))
                
                # Jalankan instalasi
                success, msg = _run_pip_install(cmd, pkg_key)
                display(HTML(f"{'✅' if success else '❌'} {msg}"))
                
                total_packages += 1
                if success:
                    installed_packages += 1

            # Tambahan package custom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            for pkg in custom_packages:
                pkg = pkg.strip()
                if pkg:
                    cmd = f"{sys.executable} -m pip install {pkg}"
                    success, msg = _run_pip_install(cmd, pkg)
                    display(HTML(f"{'✅' if success else '❌'} {msg}"))
                    
                    total_packages += 1
                    if success:
                        installed_packages += 1

            # Update progress
            duration = time.time() - start_time
            display(HTML(
                f"🏁 Instalasi selesai: "
                f"{installed_packages}/{total_packages} package "
                f"dalam {duration:.2f} detik"
            ))
            
            ui_components['install_progress'].value = 100

    def _on_check_installations(b):
        """Handler untuk tombol cek instalasi."""
        with ui_components['status']:
            clear_output()
            _check_package_status(PACKAGE_CHECKS)

    def _on_check_all(b):
        """Handler untuk cek semua package."""
        for key in PACKAGE_GROUPS.keys():
            ui_components[key].value = True

    def _on_uncheck_all(b):
        """Handler untuk uncheck semua package."""
        for key in PACKAGE_GROUPS.keys():
            ui_components[key].value = False

    # Registrasi event handlers
    ui_components['install_button'].on_click(_on_install_packages)
    ui_components['check_button'].on_click(_on_check_installations)
    ui_components['check_all_button'].on_click(_on_check_all)
    ui_components['uncheck_all_button'].on_click(_on_uncheck_all)

    return ui_components