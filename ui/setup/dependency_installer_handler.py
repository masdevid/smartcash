"""
File: smartcash/ui/setup/dependency_installer_handler.py
Deskripsi: Handler untuk instalasi dependencies SmartCash
"""

import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple
from IPython.display import display, clear_output, HTML

def setup_dependency_installer_handlers(ui_components, config=None):
    """Setup handler untuk instalasi dependencies SmartCash."""
    
    # Definisi package dan requirement
    PACKAGE_GROUPS = {
        'yolov5_req': ('YOLOv5 Requirements', _get_yolov5_requirements),
        'torch_req': ('PyTorch', f"{sys.executable} -m pip install torch torchvision torchaudio"),
        'albumentations_req': ('Albumentations', f"{sys.executable} -m pip install albumentations"),
        'notebook_req': ('Notebook Packages', f"{sys.executable} -m pip install ipywidgets tqdm"),
        'smartcash_req': ('SmartCash Utils', f"{sys.executable} -m pip install pyyaml python-dotenv termcolor"),
        'opencv_req': ('OpenCV', f"{sys.executable} -m pip install opencv-python"),
        'matplotlib_req': ('Matplotlib', f"{sys.executable} -m pip install matplotlib seaborn"),
        'pandas_req': ('Pandas', f"{sys.executable} -m pip install pandas"),
        'seaborn_req': ('Seaborn', f"{sys.executable} -m pip install seaborn")
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
        """Eksekusi perintah pip install dengan update progress."""
        try:
            ui_components['install_progress'].description = f'Installing {package_name}...'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, f"✅ {package_name} berhasil diinstall"
        except Exception as e:
            return False, f"❌ Gagal install {package_name}: {str(e)}"

    def _get_yolov5_requirements() -> List[str]:
        """Dapatkan daftar requirements YOLOv5."""
        try:
            req_file = Path('yolov5/requirements.txt')
            if not req_file.exists():
                return ["matplotlib", "numpy", "opencv-python", "PyYAML", "torch", "torchvision", "tqdm"]
            
            with open(req_file, 'r') as f:
                return [re.match(r'^([a-zA-Z0-9_\-]+)', line.strip()).group(1) 
                        for line in f if line.strip() and not line.startswith('#')]
        except Exception:
            return ["matplotlib", "numpy", "opencv-python", "PyYAML", "torch", "torchvision", "tqdm"]

    def _check_installed_packages(package_checks: List[Tuple[str, str]]) -> List[str]:
        """Cek paket yang sudah terinstall."""
        installed_packages = []
        for display_name, import_name in package_checks:
            try:
                __import__(import_name)
                installed_packages.append(import_name)
                display(HTML(f"<div style='color:green'>✅ {display_name} terinstall</div>"))
            except ImportError:
                display(HTML(f"<div style='color:orange'>⚠️ {display_name} tidak terinstall</div>"))
        return installed_packages

    def _on_install_packages(b):
        """Handler untuk tombol install packages."""
        with ui_components['status']:
            clear_output()
            start_time = time.time()
            installed_count = 0
            total_packages = 0

            # Update UI state
            ui_components['install_progress'].layout.visibility = 'visible'
            ui_components['install_progress'].value = 0

            # Proses instalasi package yang dipilih
            for pkg_key, (pkg_name, pkg_cmd) in PACKAGE_GROUPS.items():
                if not ui_components[pkg_key].value:
                    continue

                total_packages += 1
                cmd = pkg_cmd() if callable(pkg_cmd) else pkg_cmd
                success, msg = _run_pip_install(cmd, pkg_name)
                
                display(HTML(f"<div style='color:{'green' if success else 'red'}'>{msg}</div>"))
                if success:
                    installed_count += 1

            # Tambahan package custom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            for pkg in custom_packages:
                pkg = pkg.strip()
                if pkg:
                    total_packages += 1
                    cmd = f"{sys.executable} -m pip install {pkg}"
                    success, msg = _run_pip_install(cmd, pkg)
                    display(HTML(f"<div style='color:{'green' if success else 'red'}'>{msg}</div>"))
                    if success:
                        installed_count += 1

            # Update progress
            ui_components['install_progress'].value = 100
            duration = time.time() - start_time
            display(HTML(f"✅ Instalasi selesai: {installed_count}/{total_packages} package dalam {duration:.2f} detik"))

    def _on_check_installations(b):
        """Handler untuk tombol cek instalasi."""
        with ui_components['status']:
            clear_output()
            _check_installed_packages(PACKAGE_CHECKS)

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