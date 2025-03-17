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
from tqdm.auto import tqdm

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

def setup_dependency_installer_handlers(ui_components: Dict, config: Dict[Any, Any]):
    """Setup handler untuk instalasi dependencies SmartCash."""
    # Definisi package dan requirement
    PACKAGE_GROUPS = {
        'yolov5_req': lambda: _get_project_requirements('yolov5'),
        'torch_req': ['torch', 'torchvision', 'torchaudio'],
        'albumentations_req': ['albumentations'],
        'notebook_req': ['ipywidgets', 'tqdm'],
        'smartcash_req': lambda: _get_project_requirements('smartcash'),
        'opencv_req': ['opencv-python'],
        'matplotlib_req': ['matplotlib', 'seaborn'],
        'pandas_req': ['pandas'],
        'seaborn_req': ['seaborn']
    }

    PACKAGE_CHECKS = [
        ('PyTorch', 'torch'), ('TorchVision', 'torchvision'), 
        ('OpenCV', 'cv2'), ('Albumentations', 'albumentations'), 
        ('NumPy', 'numpy'), ('Pandas', 'pandas'), 
        ('Matplotlib', 'matplotlib'), ('Seaborn', 'seaborn'), 
        ('ipywidgets', 'ipywidgets'), ('tqdm', 'tqdm'), 
        ('PyYAML', 'yaml'), ('termcolor', 'termcolor')
    ]

    def _run_pip_install(packages: List[str]) -> Tuple[bool, str]:
        """Eksekusi instalasi package."""
        try:
            # Gabungkan package ke dalam satu command
            cmd = f"{sys.executable} -m pip install {' '.join(packages)}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stderr if result.returncode != 0 else ''
        except Exception as e:
            return False, str(e)

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
            
            # Dapatkan daftar package yang akan diinstall
            packages_to_install = []
            for pkg_key, pkg_list in PACKAGE_GROUPS.items():
                if not ui_components[pkg_key].value:
                    continue
                
                # Resolve package list (untuk packages yang menggunakan lambda)
                resolved_packages = pkg_list() if callable(pkg_list) else pkg_list
                packages_to_install.extend(resolved_packages)
            
            # Tambahkan package custom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            packages_to_install.extend([pkg.strip() for pkg in custom_packages if pkg.strip()])
            
            # Hapus duplikat
            packages_to_install = list(dict.fromkeys(packages_to_install))
            
            # Siapkan progress bar
            progress_bar = tqdm(
                total=len(packages_to_install), 
                desc="Instalasi Packages", 
                bar_format="{l_bar}{bar}"
            )
            
            # Update progress bar di UI
            ui_components['install_progress'].max = len(packages_to_install)
            ui_components['install_progress'].layout.visibility = 'visible'
            
            # Proses instalasi
            installed_count = 0
            for pkg in packages_to_install:
                display(HTML(f"📦 Memulai instalasi: {pkg}"))
                
                # Jalankan instalasi
                success, error_msg = _run_pip_install([pkg])
                
                # Update progress
                if success:
                    installed_count += 1
                    display(HTML(f"✅ {pkg} berhasil diinstall"))
                else:
                    display(HTML(f"❌ Gagal install {pkg}: {error_msg}"))
                
                # Update progress bar
                progress_bar.update(1)
                ui_components['install_progress'].value = progress_bar.n
            
            # Tutup progress bar
            progress_bar.close()
            
            # Hitung durasi
            duration = time.time() - start_time
            display(HTML(
                f"🏁 Instalasi selesai: "
                f"{installed_count}/{len(packages_to_install)} package "
                f"dalam {duration:.2f} detik"
            ))

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