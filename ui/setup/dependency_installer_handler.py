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
from smartcash.ui.components.alerts import create_info_alert, create_status_indicator, create_info_box

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], config: Dict[Any, Any] = None):
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
                display(create_status_indicator('success', f"✅ {display_name}{version_display}"))
            except ImportError:
                display(create_status_indicator('warning', f"⚠️ {display_name} tidak terinstall"))

    def _update_status_panel(status_message, status_type='info'):
        """Update status panel."""
        if status_type == 'warning':
            icon = '⚠️'
        elif status_type == 'success':
            icon = '✅'
        elif status_type == 'error':
            icon = '❌'
        else:  # info
            icon = 'ℹ️'
            
        ui_components['info_box'].value = create_info_alert(message=status_message, alert_type=status_type, icon=icon)

    def _on_install_packages(b):
        """Handler untuk tombol install packages."""
        with ui_components['status']:
            clear_output()
            start_time = time.time()
            
            # Update status panel
            _update_status_panel("Memulai instalasi packages...")
            
            # Dapatkan daftar package yang akan diinstall
            packages_to_install = []
            for pkg_key, pkg_list in PACKAGE_GROUPS.items():
                if pkg_key in ui_components and ui_components[pkg_key].value:
                    # Resolve package list (untuk packages yang menggunakan lambda)
                    resolved_packages = pkg_list() if callable(pkg_list) else pkg_list
                    packages_to_install.extend(resolved_packages)
            
            # Tambahkan package custom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            packages_to_install.extend([pkg.strip() for pkg in custom_packages if pkg.strip()])
            
            # Hapus duplikat
            packages_to_install = list(dict.fromkeys(packages_to_install))
            
            if not packages_to_install:
                _update_status_panel("Tidak ada package yang dipilih", 'warning')
                display(create_info_alert(message="Tidak ada package yang dipilih", alert_type='warning'))
                return
            
            # Siapkan progress bar
            progress_bar = tqdm(
                total=len(packages_to_install), 
                desc="Instalasi Packages", 
                bar_format="{l_bar}{bar}"
            )
            
            # Update progress bar di UI
            ui_components['install_progress'].max = len(packages_to_install)
            ui_components['install_progress'].value = 0
            ui_components['install_progress'].layout.visibility = 'visible'
            
            # Proses instalasi
            installed_count = 0
            failed_packages = []
            
            for pkg in packages_to_install:
                display(create_info_alert(message=f"📦 Memulai instalasi: {pkg}", alert_type='info'))
                
                # Jalankan instalasi
                success, error_msg = _run_pip_install([pkg])
                
                # Update progress
                if success:
                    installed_count += 1
                    display(create_info_alert(message=f"✅ {pkg} berhasil diinstall", alert_type='success'))
                else:
                    failed_packages.append(pkg)
                    display(create_info_alert(message=f"❌ Gagal install {pkg}: {error_msg}", alert_type='error'))
                
                # Update progress bar
                progress_bar.update(1)
                ui_components['install_progress'].value = progress_bar.n
            
            # Tutup progress bar
            progress_bar.close()
            
            # Hitung durasi
            duration = time.time() - start_time
            
            # Update status panel berdasarkan hasil
            if failed_packages:
                _update_status_panel(f"Instalasi selesai dengan {len(failed_packages)} error", 'warning')
            else:
                _update_status_panel(f"Semua {installed_count} package berhasil diinstall", 'success')
            
            # Tampilkan ringkasan
            display(HTML(f"""
            <div style="padding:10px;margin:10px 0;background-color:#d1ecf1;color:#0c5460;border-radius:4px">
                <h3 style="margin:5px 0;color:inherit">📊 Ringkasan Instalasi</h3>
                <p style="margin:5px 0;color:inherit">🕒 Waktu instalasi: {duration:.2f} detik</p>
                <p style="margin:5px 0;color:inherit">📦 Total package: {len(packages_to_install)}</p>
                <p style="margin:5px 0;color:inherit">✅ Berhasil: {installed_count}</p>
                <p style="margin:5px 0;color:inherit">❌ Gagal: {len(failed_packages)}</p>
            </div>
            """))
            
            # Tampilkan failed packages jika ada
            if failed_packages:
                failed_list = "<br>".join([f"❌ {pkg}" for pkg in failed_packages])
                display(HTML(f"""
                <div style="padding:10px;margin:10px 0;background-color:#f8d7da;color:#721c24;border-radius:4px">
                    <h3 style="margin:5px 0;color:inherit">❌ Package Gagal Diinstall</h3>
                    <div style="color:inherit">{failed_list}</div>
                </div>
                """))

    def _on_check_installations(b):
        """Handler untuk tombol cek instalasi."""
        with ui_components['status']:
            clear_output()
            display(create_info_box(message="🔍 Memeriksa Status Instalasi"))
            _check_package_status(PACKAGE_CHECKS)

    def _on_check_all(b):
        """Handler untuk cek semua package."""
        for key in PACKAGE_GROUPS.keys():
            if key in ui_components:
                ui_components[key].value = True
        
        _update_status_panel("Semua package dipilih", 'success')

    def _on_uncheck_all(b):
        """Handler untuk uncheck semua package."""
        for key in PACKAGE_GROUPS.keys():
            if key in ui_components:
                ui_components[key].value = False
        
        _update_status_panel("Semua package tidak dipilih", 'warning')

    # Registrasi event handlers
    ui_components['install_button'].on_click(_on_install_packages)
    ui_components['check_button'].on_click(_on_check_installations)
    ui_components['check_all_button'].on_click(_on_check_all)
    ui_components['uncheck_all_button'].on_click(_on_uncheck_all)

    return ui_components