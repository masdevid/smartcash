"""
File: smartcash/ui/setup/dependency_installer_handler.py
Deskripsi: Handler untuk instalasi dependencies SmartCash dengan integrasi UI utils, handlers, dan helpers
"""

import sys
import re
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable
from IPython.display import display, clear_output
from tqdm.auto import tqdm

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], config: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Setup handler untuk instalasi dependencies SmartCash dengan integrasi UI utils."""
    
    # Import utils dan handlers untuk konsistensi dan mengurangi duplikasi
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    from smartcash.ui.utils.metric_utils import create_metric_display
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    from smartcash.ui.utils.metric_utils import create_metric_display
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    from smartcash.ui.utils.fallback_utils import update_status_panel
    from smartcash.ui.utils.fallback_utils import update_status_panel
    
    # Setup observer handlers
    ui_components = setup_observer_handlers(ui_components, "dependency_installer_observers")
    
    # Definisi package dan requirement
    PACKAGE_GROUPS = {
        'yolov5_req': lambda: _get_project_requirements('yolov5'),
        'torch_req': ['torch', 'torchvision', 'torchaudio'],
        'albumentations_req': ['albumentations'],
        'notebook_req': ['ipywidgets', 'tqdm'],
        'smartcash_req': lambda: _get_project_requirements('smartcash'),
        'opencv_req': ['opencv-python'],
        'matplotlib_req': ['matplotlib'],
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
        Dapatkan requirements untuk project tertentu dengan membaca requirements.txt.
        
        Args:
            project_name: Nama project (e.g. 'smartcash', 'yolov5')
        
        Returns:
            List requirements terdeteksi
        """
        # Default requirements jika file tidak ditemukan
        default_requirements = {
            'smartcash': ["pyyaml", "termcolor", "python-dotenv", "roboflow", "ultralytics", "matplotlib", "seaborn", "pandas"],
            'yolov5': ["matplotlib", "numpy", "opencv-python", "torch", "torchvision", "tqdm", "pillow", "requests", "scipy"]
        }
        
        # Lokasi potensial file requirements
        potential_paths = [
            Path(f'{project_name}/requirements.txt'),
            Path.cwd() / f'{project_name}/requirements.txt',
            Path.home() / f'{project_name}/requirements.txt'
        ]
        
        # Parse requirements dari file
        for path in potential_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        parsed_reqs = [re.match(r'^([a-zA-Z0-9_\-]+)', line.strip()).group(1) 
                                      for line in f if line.strip() and not line.startswith('#')]
                        if parsed_reqs:
                            # Gabungkan dengan default requirements dan remove duplikat
                            return list(dict.fromkeys(parsed_reqs + default_requirements.get(project_name, [])))
                except Exception:
                    pass
        
        return default_requirements.get(project_name, [])

    def _run_pip_install(packages: List[str]) -> Tuple[bool, str]:
        """
        Eksekusi instalasi package dengan pip.
        
        Args:
            packages: List package yang akan diinstall
            
        Returns:
            Tuple (success, error_message)
        """
        try:
            # Gabungkan package ke dalam satu command
            cmd = f"{sys.executable} -m pip install {' '.join(packages)}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stderr if result.returncode != 0 else ''
        except Exception as e:
            return False, str(e)
    
    def _check_package_status(package_checks: List[Tuple[str, str]], output_widget=None) -> None:
        """
        Periksa status paket yang terinstall.
        
        Args:
            package_checks: List tuple (display_name, import_name)
            output_widget: Widget untuk menampilkan output
        """
        output_widget = output_widget or ui_components['status']
        with output_widget:
            clear_output()
            display(create_info_alert("Memeriksa Status Instalasi", 'info', '🔍'))
            
            for display_name, import_name in package_checks:
                try:
                    module = __import__(import_name)
                    version = getattr(module, '__version__', 'Unknown')
                    version_display = f" (v{version})" if version != 'Unknown' else ''
                    display(create_status_indicator('success', f"{display_name}{version_display}"))
                except ImportError:
                    display(create_status_indicator('warning', f"{display_name} tidak terinstall"))

    def _on_install_packages(b):
        """Handler untuk tombol install packages dengan progress tracking."""
        with ui_components['status']:
            clear_output()
            start_time = time.time()
            
            # Update status panel
            update_status_panel(ui_components, "🚀 Memulai instalasi packages...", "info")
            
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
            
            # Hapus duplikat dan filter package kosong
            packages_to_install = list(dict.fromkeys([p for p in packages_to_install if p]))
            
            if not packages_to_install:
                update_status_panel(ui_components, "⚠️ Tidak ada package yang dipilih", 'warning')
                display(create_info_alert("Tidak ada package yang dipilih", 'warning'))
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
                # Tampilkan status dengan create_info_alert
                display(create_info_alert(f"Memulai instalasi: {pkg}", 'info', '📦'))
                
                # Jalankan instalasi
                success, error_msg = _run_pip_install([pkg])
                
                # Update progress
                if success:
                    installed_count += 1
                    display(create_info_alert(f"{pkg} berhasil diinstall", 'success', '✅'))
                else:
                    failed_packages.append((pkg, error_msg))
                    display(create_info_alert(f"Gagal install {pkg}: {error_msg}", 'error', '❌'))
                
                # Update progress bar
                progress_bar.update(1)
                ui_components['install_progress'].value = progress_bar.n
                percentage = int((progress_bar.n / len(packages_to_install)) * 100)
                ui_components['install_progress'].description = f"Proses: {percentage}%"
            
            # Tutup progress bar
            progress_bar.close()
            ui_components['install_progress'].layout.visibility = 'hidden'
            
            # Hitung durasi
            duration = time.time() - start_time
            
            # Update status panel berdasarkan hasil
            if failed_packages:
                update_status_panel(ui_components, f"⚠️ Instalasi selesai dengan {len(failed_packages)} error", 'warning')
            else:
                update_status_panel(ui_components, f"Semua {installed_count} package berhasil diinstall", 'success')
            
            # Buat widget metrics untuk ringkasan
            display(create_metric_display("✅ Berhasil", installed_count, is_good=installed_count > 0))
            display(create_metric_display("❌ Gagal", len(failed_packages), is_good=len(failed_packages) == 0))
            display(create_metric_display("⏱️ Waktu", f"{duration:.2f} detik"))
            
            # Tampilkan failed packages jika ada
            if failed_packages:
                error_details = "<br>".join([f"❌ {pkg}: {err}" for pkg, err in failed_packages])
                display(create_info_alert(f"<h3>Package Gagal Diinstall</h3><div>{error_details}</div>", 'error', '❌'))

    def _on_check_all(b):
        """Handler untuk tombol check all packages."""
        for key, widget in ui_components.items():
            if key in PACKAGE_GROUPS and hasattr(widget, 'value'):
                widget.value = True
        update_status_panel(ui_components, "✅ Semua package dipilih", 'success')

    def _on_uncheck_all(b):
        """Handler untuk tombol uncheck all packages."""
        for key, widget in ui_components.items():
            if key in PACKAGE_GROUPS and hasattr(widget, 'value'):
                widget.value = False
        update_status_panel(ui_components, "⚠️ Semua package tidak dipilih", 'warning')

    # Registrasi event handlers
    def on_install_packages_wrapped(b):
        """Wrapper dengan error handling untuk on_install_packages."""
        try:
            _on_install_packages(b)
        except Exception as e:
            with ui_components['status']:
                clear_output()
                display(create_status_indicator('error', f"❌ Error instalasi: {str(e)}"))
                
    ui_components['install_button'].on_click(on_install_packages_wrapped)
    def on_check_installations(b):
        """Handler untuk tombol cek instalasi dengan error handling."""
        try:
            _check_package_status(PACKAGE_CHECKS, ui_components['status'])
        except Exception as e:
            with ui_components['status']:
                clear_output()
                display(create_status_indicator('error', f"❌ Error cek instalasi: {str(e)}"))
    
    ui_components['check_button'].on_click(on_check_installations)
    ui_components['check_all_button'].on_click(_on_check_all)
    ui_components['uncheck_all_button'].on_click(_on_uncheck_all)
    
    # Register cleanup function
    def cleanup():
        """Cleanup resources."""
        if 'observer_group' in ui_components:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                observer_manager = ObserverManager()
                observer_manager.unregister_group(ui_components['observer_group'])
            except ImportError:
                pass
    
    ui_components['cleanup'] = cleanup

    return ui_components