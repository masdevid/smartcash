"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Refactored
Deskripsi: Handler untuk instalasi dependencies SmartCash dengan pendekatan modular dan pemrosesan YOLOv5 requirements yang lebih baik
"""

import sys
import subprocess
import threading
import queue
import time
import re
from typing import List, Tuple, Dict
from IPython.display import display, clear_output, HTML
from pathlib import Path

def setup_dependency_installer_handlers(ui_components, config=None):
    """Setup handler untuk instalasi dependencies SmartCash."""
    # Queue thread-safe untuk komunikasi
    status_queue = queue.Queue()
    # Import create_status_indicator from ui_utils
    try:
        from smartcash.utils.ui_utils import create_status_indicator
    except ImportError:
        from IPython.display import HTML
        # Fallback function if not available
        def create_status_indicator(status_type, message):
            colors = {'info': 'blue', 'success': 'green', 'warning': 'orange', 'error': 'red'}
            return HTML(
                f"<div style='padding: 5px; margin: 2px 0; border-left: 3px solid {colors[status_type]};'>"
                f"{message}"
                "</div>"
            )
    # Check for already installed packages
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

    installed_packages = _check_installed(PACKAGE_CHECKS)

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
    
    # Replace run_pip_install() with this version
    def run_pip_install(cmd: str, package_name: str) -> Tuple[bool, str]:
        """Eksekusi perintah pip install dengan progress streaming."""
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
                    # Send heartbeat to UI
                    status_queue.put({
                        'type': 'heartbeat',
                        'package': package_name
                    })
            
            return process.returncode == 0, f"✅ {package_name} berhasil diinstall"
        except Exception as e:
            return False, f"❌ Gagal install {package_name}: {str(e)}"

    def get_selected_packages():
        # Track unique packages
        unique_packages = set()
        selected_packages = []
        
        # Collect selected packages, skipping already installed ones
        for key, pkg_info in PACKAGE_GROUPS.items():
            if not ui_components[key].value:
                continue
                
            if key in ['yolov5_req', 'smartcash_req']:
                # Handle multi-package requirements
                for req in get_package_requirements(key):
                    if req not in unique_packages and req not in installed_packages:
                        unique_packages.add(req)
                        selected_packages.append((
                            f"{sys.executable} -m pip install {req}", 
                            f"{pkg_info['name']}: {req}"
                        ))
            elif pkg_info['command'] not in unique_packages:
                unique_packages.add(pkg_info['command'])
                selected_packages.append((pkg_info['command'], pkg_info['name']))
        
        # Add custom packages, skipping already installed ones
        for pkg in [p.strip() for p in ui_components['custom_packages'].value.strip().split('\n') if p.strip()]:
            cmd = f"{sys.executable} -m pip install {pkg}"
            if cmd not in unique_packages and pkg not in installed_packages:
                unique_packages.add(cmd)
                selected_packages.append((cmd, f"Custom: {pkg}"))
        return selected_packages
        
    def install_packages_thread():
        """Thread untuk instalasi packages."""
        try:
            # Reset progress
            status_queue.put({'type': 'progress', 'value': 0, 'description': 'Memulai instalasi...'})
            selected_packages = get_selected_packages()
            # Cek apakah ada package yang dipilih
            if not selected_packages:
                status_queue.put({
                    'type': 'status',
                    'status_type': 'warning',
                    'message': '⚠️ Tidak ada package yang dipilih untuk diinstall'
                })
                status_queue.put({'type': 'complete'})
                return
            
            # Flag force reinstall
            force_flag = "--force-reinstall" if ui_components['force_reinstall'].value else ""
            
            # Install packages
            total = len(selected_packages)
            start_time = time.time()
            last_update_time = start_time
            
            for i, (cmd, name) in enumerate(selected_packages):
                # Calculate progress
                progress = int((i / total) * 100)
                current_time = time.time()
                
                # Update progress
                status_queue.put({
                    'type': 'progress',
                    'value': progress,
                    'description': f'Installing {name}...'
                })
                
                # Pesan info
                status_queue.put({
                    'type': 'status',
                    'status_type': 'info',
                    'message': f'🔄 Menginstall {name}...'
                })
                
                # Jalankan instalasi
                full_cmd = f"{cmd} {force_flag}"
                
                # Installation with progress monitoring
                should_continue = [True]  # Mutable reference untuk thread komunikasi
                
                def check_progress():
                    wait_time = 0
                    while should_continue[0]:
                        time.sleep(5)
                        wait_time += 5
                        if wait_time >= 180:  # 3 minutes
                            status_queue.put({
                                'type': 'status', 'status_type': 'warning',
                                'message': f'⚠️ Instalasi {name} masih berjalan ({wait_time}s). Harap bersabar...'
                            })
                            wait_time = 0
                
                progress_thread = threading.Thread(target=check_progress)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Run installation with timing
                install_start = time.time()
                success, msg = run_pip_install(full_cmd, name)
                install_time = time.time() - install_start
                should_continue[0] = False
                
                # Update status with timing info for slow installs
                status_type = 'success' if success else 'error'
                if install_time > 15:
                    msg += f" (⏱️ {install_time:.1f}s)"
                status_queue.put({'type': 'status', 'status_type': status_type, 'message': msg})
                
                # Update timestamp
                last_update_time = current_time
            
            # Complete
            total_time = time.time() - start_time
            status_queue.put({
                'type': 'progress',
                'value': 100,
                'description': 'Instalasi selesai'
            })
            
            status_queue.put({
                'type': 'status',
                'status_type': 'success',
                'message': f'✅ Instalasi selesai dalam {total_time:.1f} detik'
            })
            
            status_queue.put({'type': 'complete'})
            
        except Exception as e:
            status_queue.put({
                'type': 'status',
                'status_type': 'error',
                'message': f'❌ Error: {str(e)}'
            })
            status_queue.put({'type': 'complete'})

    def _check_installed(checked_packages: List[Tuple[str, str]]) -> List[str]:
        """
        Check if packages are already installed and return a list of packages to skip.
        
        Args:
            checked_packages: List of tuples containing (display_name, import_name) for packages to check.
        
        Returns:
            List of packages that are already installed.
        """
        installed_packages = []
        
        for display_name, import_name in checked_packages:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                installed_packages.append(import_name)
                display(create_status_indicator(
                    'success', 
                    f'{display_name}: v{version} (sudah terinstall)'
                ))
            except ImportError:
                display(create_status_indicator(
                    'warning',
                    f'{display_name}: Tidak terinstall'
                ))
        
        return installed_packages
        
    def update_ui_thread():
        """Thread for updating UI from queue."""
        while True:
            try:
                # Get status from queue with shorter timeout
                status = status_queue.get(timeout=0.1)
                
                # Update UI based on status type
                if status['type'] == 'progress':
                    ui_components['install_progress'].value = status['value']
                    ui_components['install_progress'].description = status['description']
                elif status['type'] == 'heartbeat':
                    # Update progress description only
                    ui_components['install_progress'].description = f'Installing {status["package"]}...'
                    ui_components['install_progress'].value += 1  # Pulsing animation effect
                elif status['type'] == 'status':
                    with ui_components['status']:
                        display(create_status_indicator(
                            status['status_type'], 
                            status['message']
                        ))
                elif status['type'] == 'complete':
                    # Reset UI state
                    ui_components['install_button'].disabled = False
                    ui_components['check_button'].disabled = False
                    ui_components['check_all_button'].disabled = False
                    ui_components['uncheck_all_button'].disabled = False
                    ui_components['install_progress'].layout.visibility = 'hidden'
                    break
                    
                # Process next item immediately
                status_queue.task_done()
                
            except queue.Empty:
                # Check if main thread is still alive
                if not threading.main_thread().is_alive():
                    break
            except Exception as e:
                # Log error but continue processing
                print(f"UI update error: {str(e)}")
                continue
    
    def on_install_click(b):
        """Handler untuk tombol install."""
        # Update UI state
        ui_components['install_button'].disabled = True
        ui_components['check_button'].disabled = True
        ui_components['check_all_button'].disabled = True
        ui_components['uncheck_all_button'].disabled = True
        ui_components['install_progress'].layout.visibility = 'visible'
        
        with ui_components['status']:
            clear_output()
        
        # Clear queue
        while not status_queue.empty():
            status_queue.get()
            
        # Start UI update thread
        ui_thread = threading.Thread(target=update_ui_thread)
        ui_thread.daemon = True
        ui_thread.start()
        
        # Start installation thread
        install_thread = threading.Thread(target=install_packages_thread)
        install_thread.daemon = True
        install_thread.start()
    
    def on_check_click(b):
        """Handler untuk tombol cek instalasi."""
        with ui_components['status']:
            clear_output()
            display(create_status_indicator('info', '🔍 Memeriksa paket terinstall...'))
            
            # Check each package
            _check_installed(PACKAGE_CHECKS)
            
            # Check CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    display(create_status_indicator(
                        'success',
                        f'CUDA tersedia: {device_name}'
                    ))
                else:
                    display(create_status_indicator(
                        'info',
                        'CUDA tidak tersedia, menggunakan CPU'
                    ))
            except ImportError:
                pass
    
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
    
    # Cleanup function
    def cleanup():
        """Bersihkan resources."""
        while not status_queue.empty():
            status_queue.get()
    
    ui_components['cleanup'] = cleanup
    
    return ui_components