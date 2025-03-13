"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk instalasi dependencies SmartCash dengan pendekatan modular
"""

import sys
import subprocess
import threading
import queue
from typing import List, Tuple, Dict
from IPython.display import display, clear_output, HTML
from pathlib import Path

def setup_dependency_installer_handlers(ui_components, config=None):
    """Setup handler untuk instalasi dependencies SmartCash."""
    # Queue thread-safe untuk komunikasi
    status_queue = queue.Queue()
    
    # Mapping package groups
    PACKAGE_GROUPS = {
        'yolov5_req': {
            'name': 'YOLOv5 Requirements',
            'command': f"{sys.executable} -m pip install -r yolov5/requirements.txt"
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
    
    def run_pip_install(cmd: str, package_name: str) -> Tuple[bool, str]:
        """Eksekusi perintah pip install."""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True, f"✅ {package_name} berhasil diinstall"
        except subprocess.CalledProcessError as e:
            return False, f"❌ Gagal install {package_name}: {e.stderr}"
    
    def install_packages_thread():
        """Thread untuk instalasi packages."""
        try:
            # Reset progress
            status_queue.put({
                'type': 'progress', 
                'value': 0, 
                'description': 'Memulai instalasi...'
            })
            
            # Collect selected packages
            selected_packages = []
            for key, pkg_info in PACKAGE_GROUPS.items():
                if ui_components[key].value:
                    selected_packages.append((pkg_info['command'], pkg_info['name']))
            
            # Tambahkan custom packages
            custom_pkgs = ui_components['custom_packages'].value.strip().split('\n')
            custom_pkgs = [pkg.strip() for pkg in custom_pkgs if pkg.strip()]
            for pkg in custom_pkgs:
                selected_packages.append((
                    f"{sys.executable} -m pip install {pkg}", 
                    f"Custom package: {pkg}"
                ))
            
            # Cek apakah ada package yang dipilih
            if not selected_packages:
                status_queue.put({
                    'type': 'status',
                    'status_type': 'warning',
                    'message': 'Tidak ada package yang dipilih untuk diinstall'
                })
                status_queue.put({'type': 'complete'})
                return
            
            # Flag force reinstall
            force_flag = "--force-reinstall" if ui_components['force_reinstall'].value else ""
            
            # Install packages
            total = len(selected_packages)
            for i, (cmd, name) in enumerate(selected_packages):
                # Update progress
                progress = int((i / total) * 100)
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
                success, msg = run_pip_install(full_cmd, name)
                
                # Update status
                status_queue.put({
                    'type': 'status',
                    'status_type': 'success' if success else 'error',
                    'message': msg
                })
            
            # Complete
            status_queue.put({
                'type': 'progress',
                'value': 100,
                'description': 'Instalasi selesai'
            })
            
            status_queue.put({
                'type': 'status',
                'status_type': 'success',
                'message': 'Instalasi package selesai'
            })
            
            status_queue.put({'type': 'complete'})
            
        except Exception as e:
            status_queue.put({
                'type': 'status',
                'status_type': 'error',
                'message': f'Error: {str(e)}'
            })
            status_queue.put({'type': 'complete'})
    
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
    
    def create_status_indicator(status, message):
        """Buat indikator status dengan styling konsisten."""
        status_styles = {
            'success': {'icon': '✅', 'color': 'green'},
            'warning': {'icon': '⚠️', 'color': 'orange'},
            'error': {'icon': '❌', 'color': 'red'},
            'info': {'icon': 'ℹ️', 'color': 'blue'}
        }
        
        style = status_styles.get(status, status_styles['info'])
        
        return HTML(f"""
        <div style="margin: 5px 0; padding: 8px 12px; 
                    border-radius: 4px; background-color: #f8f9fa;">
            <span style="color: {style['color']}; font-weight: bold;"> 
                {style['icon']} {message}
            </span>
        </div>
        """)
    
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
            display(HTML("<h3>🔍 Checking installed packages</h3>"))
            
            # List of packages to check
            package_checks = [
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
            
            # Check each package
            for display_name, import_name in package_checks:
                try:
                    module = __import__(import_name)
                    version = getattr(module, '__version__', 'Unknown')
                    display(create_status_indicator(
                        'success', 
                        f'{display_name}: v{version}'
                    ))
                except ImportError:
                    display(create_status_indicator(
                        'warning',
                        f'{display_name}: Tidak terinstall'
                    ))
            
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