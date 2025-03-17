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

def _get_project_requirements(project_name: str) -> List[str]:
    """
    Dapatkan requirements untuk project tertentu.
    
    Args:
        project_name: Nama project (e.g. 'smartcash', 'yolov5')
    
    Returns:
        List requirements terdeteksi
    """
    default_requirements = {
        'smartcash': [
            "pyyaml", "termcolor", "python-dotenv", 
            "roboflow", "ultralytics", "matplotlib", 
            "seaborn", "pandas"
        ],
        'yolov5': [
            "matplotlib", "numpy", "opencv-python", 
            "torch", "torchvision", "tqdm", "pillow", 
            "requests", "scipy"
        ]
    }
    
    # Lokasi potensial file requirements
    potential_paths = [
        Path(f'{project_name}/requirements.txt'),
        Path.cwd() / f'{project_name}/requirements.txt',
        Path.home() / f'{project_name}/requirements.txt'
    ]
    
    def parse_requirements(file_path):
        """Parse requirements dari file."""
        try:
            with open(file_path, 'r') as f:
                return [
                    re.match(r'^([a-zA-Z0-9_\-]+)', line.strip()).group(1)
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
        except Exception:
            return []
    
    # Coba temukan dan parsing requirements
    for path in potential_paths:
        if path.exists():
            parsed_reqs = parse_requirements(path)
            if parsed_reqs:
                return list(dict.fromkeys(parsed_reqs + default_requirements.get(project_name, [])))
    
    return default_requirements.get(project_name, [])

def _simulate_progress(progress_widget, message: str = 'Memproses...'):
    """
    Simulasi progress bar tanpa threading.
    
    Args:
        progress_widget: Widget progress untuk diupdate
        message: Pesan progress
    """
    progress_widget.layout.visibility = 'visible'
    
    for i in range(0, 101, 10):
        progress_widget.value = i
        progress_widget.description = f"{message} {i}%"
        time.sleep(0.2)  # Simulasi delay
    
    progress_widget.description = 'Selesai!'

def setup_dependency_installer_handlers(ui_components: Dict, config: Dict[Any, Any]):
    """Setup handler untuk instalasi dependencies SmartCash."""
    # Definisi package dan requirement
    PACKAGE_GROUPS = {
        'yolov5_req': lambda: f"{sys.executable} -m pip install {' '.join(_get_project_requirements('yolov5'))}",
        'torch_req': f"{sys.executable} -m pip install torch torchvision torchaudio",
        'albumentations_req': f"{sys.executable} -m pip install albumentations",
        'notebook_req': f"{sys.executable} -m pip install ipywidgets tqdm",
        'smartcash_req': lambda: f"{sys.executable} -m pip install {' '.join(_get_project_requirements('smartcash'))}",
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
                version = getattr(module, '__version__', 'Unknown')
                version_display = f" (v{version})" if version != 'Unknown' else ''
                display(HTML(f"<div style='color:green'>✅ {display_name}{version_display}</div>"))
            except ImportError:
                display(HTML(f"<div style='color:orange'>⚠️ {display_name} tidak terinstall</div>"))

    def _on_install_packages(b):
        """Handler untuk tombol install packages."""
        with ui_components['status']:
            clear_output()
            start_time = time.time()
            
            # Proses instalasi package yang dipilih
            total_packages = 0
            installed_packages = 0

            for pkg_key, pkg_cmd in PACKAGE_GROUPS.items():
                if not ui_components[pkg_key].value:
                    continue

                # Dapatkan command untuk instalasi
                cmd = pkg_cmd() if callable(pkg_cmd) else pkg_cmd
                display(HTML(f"📦 Memulai instalasi: {pkg_key}"))
                
                # Simulasi progress
                _simulate_progress(ui_components['install_progress'], f'Instalasi {pkg_key}')
                
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
                    # Simulasi progress
                    _simulate_progress(ui_components['install_progress'], f'Instalasi {pkg}')
                    
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