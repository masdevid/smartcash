"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Refactored
Deskripsi: Handler untuk instalasi dependencies SmartCash dengan pendekatan modular dan pemrosesan YOLOv5 requirements yang lebih baik
"""

import sys
import subprocess
import time
import re
from typing import List, Tuple, Dict
from IPython.display import display, clear_output, HTML
from pathlib import Path

def setup_dependency_installer_handlers(ui_components, config=None):
    """Setup handler untuk instalasi dependencies SmartCash."""

    # Mapping package groups
    PACKAGE_GROUPS = {
        'yolov5_req': {
            'name': 'YOLOv5 Requirements',
            'command': 'yolov5_requirements'  # Special handling untuk yolov5
        },
        'torch_req': {
            'name': 'PyTorch',
            'command': f"{sys.executable} -m pip install torch torchvision torchaudio"
        },
        'albumentations_req': {
            'name': 'Albumentations',
            'command': f"{sys.executable} -m pip install albumentations"
        },
        'notebook_req': {
            'name': 'Notebook Packages',
            'command': f"{sys.executable} -m pip install ipywidgets tqdm matplotlib"
        },
        'smartcash_req': {
            'name': 'SmartCash Requirements',
            'command': f"{sys.executable} -m pip install pyyaml termcolor python-dotenv roboflow"
        },
        'opencv_req': {
            'name': 'OpenCV',
            'command': f"{sys.executable} -m pip install opencv-python"
        },
        'matplotlib_req': {
            'name': 'Matplotlib',
            'command': f"{sys.executable} -m pip install matplotlib"
        },
        'pandas_req': {
            'name': 'Pandas',
            'command': f"{sys.executable} -m pip install pandas"
        },
        'seaborn_req': {
            'name': 'Seaborn',
            'command': f"{sys.executable} -m pip install seaborn"
        }
    }
    PACKAGE_CHECKS = [
        ('PyTorch', 'torch'),
        ('TorchVision', 'torchvision'),
        ('OpenCV', 'cv2'),
        ('Albumentations', 'albumentations'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('Matplotlib', 'matplotlib'),
        ('Seaborn', 'seaborn'),
        ('ipywidgets', 'ipywidgets'),
        ('tqdm', 'tqdm'),
        ('PyYAML', 'yaml'),
        ('termcolor', 'termcolor'),
        ('python-dotenv', 'dotenv'),
        ('roboflow', 'roboflow')
    ]

    def get_package_requirements(key) -> List[str]:
        """Mendapatkan daftar package berdasarkan key."""
        if key == 'yolov5_req':
            try:
                req_file = Path('yolov5/requirements.txt')
                if not req_file.exists():
                    return ["matplotlib", "numpy", "opencv-python", "Pillow", "PyYAML", "requests", "scipy", "torch", "torchvision", "tqdm"]

                requirements = []
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            match = re.match(r'^([a-zA-Z0-9_\-]+)', line)
                            if match:
                                requirements.append(match.group(1))
                return requirements
            except Exception:
                return ["matplotlib", "numpy", "opencv-python", "Pillow", "PyYAML", "requests", "scipy", "torch", "torchvision", "tqdm"]
        elif key == 'smartcash_req':
            return ["pyyaml", "termcolor", "python-dotenv", "roboflow", "ipywidgets", "tqdm"]
        return []

    def run_pip_install(cmd: str, package_name: str) -> Tuple[bool, str]:
        """Eksekusi perintah pip install."""
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    # Update progress description
                    ui_components['install_progress'].description = f'Installing {package_name}...'
                    ui_components['install_progress'].value += 1  # Pulsing animation

            return process.returncode == 0, f"✅ {package_name} berhasil diinstall"
        except Exception as e:
            return False, f"❌ Gagal install {package_name}: {str(e)}"

    def _check_installed(package_checks: List[Tuple[str, str]]) -> List[str]:
        """
        Check if packages are already installed and return a list of packages to skip.

        Args:
            package_checks: List of tuples containing (display_name, import_name) for packages to check.

        Returns:
            List of packages that are already installed.
        """
        installed_packages = []

        for display_name, import_name in package_checks:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                installed_packages.append(import_name)
                display(HTML(
                    f"<div style='padding: 5px; margin: 2px 0; border-left: 3px solid green;'>"
                    f"{display_name}: v{version} (sudah terinstall)"
                    "</div>"
                ))
            except ImportError:
                display(HTML(
                    f"<div style='padding: 5px; margin: 2px 0; border-left: 3px solid orange;'>"
                    f"{display_name}: Tidak terinstall"
                    "</div>"
                ))

        return installed_packages

    def on_install_click(b):
        """Handler untuk tombol install."""
        # Update UI state
        ui_components['install_button'].disabled = True
        ui_components['check_button'].disabled = True
        ui_components['check_all_button'].disabled = True
        ui_components['uncheck_all_button'].disabled = True
        ui_components['install_progress'].layout.visibility = 'visible'
        ui_components['install_progress'].value = 0

        with ui_components['status']:
            clear_output()

        # Check for already installed packages
        installed_packages = _check_installed(PACKAGE_CHECKS)

        # Collect selected packages, skipping already installed ones
        selected_packages = []
        for key, pkg_info in PACKAGE_GROUPS.items():
            if not ui_components[key].value:
                continue

            if key in ['yolov5_req', 'smartcash_req']:
                # Handle multi-package requirements
                for req in get_package_requirements(key):
                    if req not in installed_packages:
                        selected_packages.append((
                            f"{sys.executable} -m pip install {req}",
                            f"{pkg_info['name']}: {req}"
                        ))
            elif pkg_info['command'] not in installed_packages:
                selected_packages.append((pkg_info['command'], pkg_info['name']))

        # Add custom packages, skipping already installed ones
        for pkg in [p.strip() for p in ui_components['custom_packages'].value.strip().split('\n') if p.strip()]:
            if pkg not in installed_packages:
                selected_packages.append((
                    f"{sys.executable} -m pip install {pkg}",
                    f"Custom: {pkg}"
                ))

        # Cek apakah ada package yang dipilih
        if not selected_packages:
            display(HTML(
                "<div style='padding: 5px; margin: 2px 0; border-left: 3px solid orange;'>"
                "⚠️ Tidak ada package yang dipilih untuk diinstall"
                "</div>"
            ))
            ui_components['install_button'].disabled = False
            ui_components['check_button'].disabled = False
            ui_components['check_all_button'].disabled = False
            ui_components['uncheck_all_button'].disabled = False
            return

        # Install packages
        total = len(selected_packages)
        start_time = time.time()

        for i, (cmd, name) in enumerate(selected_packages):
            # Update progress
            progress = int((i / total) * 100)
            ui_components['install_progress'].value = progress
            ui_components['install_progress'].description = f'Installing {name}...'

            # Run installation
            success, msg = run_pip_install(cmd, name)
            display(HTML(
                f"<div style='padding: 5px; margin: 2px 0; border-left: 3px solid {'green' if success else 'red'};'>"
                f"{msg}"
                "</div>"
            ))

        # Complete
        total_time = time.time() - start_time
        ui_components['install_progress'].value = 100
        ui_components['install_progress'].description = 'Instalasi selesai'

        display(HTML(
            f"<div style='padding: 5px; margin: 2px 0; border-left: 3px solid green;'>"
            f"✅ Instalasi selesai dalam {total_time:.1f} detik"
            "</div>"
        ))

        # Reset UI state
        ui_components['install_button'].disabled = False
        ui_components['check_button'].disabled = False
        ui_components['check_all_button'].disabled = False
        ui_components['uncheck_all_button'].disabled = False

    def on_check_click(b):
        """Handler untuk tombol cek instalasi."""
        with ui_components['status']:
            clear_output()
            _check_installed(PACKAGE_CHECKS)

    def on_check_all(b):
        """Handler untuk tombol check all."""
        for key in PACKAGE_GROUPS.keys():
            ui_components[key].value = True

    def on_uncheck_all(b):
        """Handler untuk tombol uncheck all."""
        for key in PACKAGE_GROUPS.keys():
            ui_components[key].value = False

    # Register handlers
    ui_components['install_button'].on_click(on_install_click)
    ui_components['check_button'].on_click(on_check_click)
    ui_components['check_all_button'].on_click(on_check_all)
    ui_components['uncheck_all_button'].on_click(on_uncheck_all)

    return ui_components