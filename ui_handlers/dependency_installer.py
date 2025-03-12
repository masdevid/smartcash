"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Refactored
Deskripsi: Handler untuk instalasi dependencies SmartCash dengan progress tracking dan pengurutan paket.
"""

import sys
import subprocess
import threading
import importlib
import queue
from typing import List, Tuple, Dict, Any
from IPython.display import display, clear_output, HTML

def setup_dependency_installer_handlers(ui_components, config=None):
    """
    Setup handler untuk instalasi dependencies SmartCash.
    
    Args:
        ui_components: Dictionary berisi komponen UI untuk instalasi
        config: Dictionary konfigurasi (opsional)
    
    Returns:
        Dictionary UI components yang sudah diupdate dengan handler
    """
    # Thread-safe queue untuk komunikasi antar thread
    status_queue = queue.Queue()
    
    def create_status_indicator(status, message):
        """Helper function untuk membuat indikator status dengan styling konsisten."""
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
    
    def run_pip_install(packages, force_reinstall=False):
        """Eksekusi pip install command dengan streaming output."""
        cmd = [sys.executable, "-m", "pip", "install"]
        
        # Tambahkan flag force reinstall jika diperlukan
        if force_reinstall:
            cmd.append("--force-reinstall")
        
        # Handle requirements.txt secara khusus
        if len(packages) == 1 and packages[0].endswith('.txt'):
            cmd.extend(["-r", packages[0]])
        else:
            cmd.extend(packages)
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Collect output
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            return return_code == 0, ''.join(output_lines)
        except Exception as e:
            return False, str(e)
    
    def update_ui_thread():
        """Thread untuk memperbarui UI dari queue."""
        while True:
            try:
                # Get status from queue with timeout
                status = status_queue.get(timeout=1)
                
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
            except queue.Empty:
                # Check if main thread is still alive
                if not threading.main_thread().is_alive():
                    break
    
    def install_packages_thread():
        """Thread utama untuk proses instalasi."""
        try:
            # Reset progress
            status_queue.put({
                'type': 'progress', 
                'value': 0, 
                'description': 'Memulai instalasi...'
            })
            
            # Package mapping dari kecil ke besar dengan metadata
            package_map = {
                'notebook_req': (['tqdm', 'ipywidgets'], 'Notebook tools'),
                'smartcash_req': (['termcolor', 'pyyaml', 'python-dotenv'], 'SmartCash utils'),
                'matplotlib_req': (['matplotlib'], 'Matplotlib'),
                'pandas_req': (['pandas'], 'Pandas'), 
                'seaborn_req': (['seaborn'], 'Seaborn'),
                'opencv_req': (['opencv-python'], 'OpenCV'),
                'albumentations_req': (['albumentations'], 'Albumentations'),
                'torch_req': ([
                    'torch', 'torchvision', 'torchaudio', 
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ], 'PyTorch'),
                'yolov5_req': (['yolov5/requirements.txt'], 'YOLOv5 requirements')
            }
            
            # Collect selected packages
            selected_packages = []
            for key, (packages, name) in package_map.items():
                if ui_components[key].value:
                    selected_packages.append((packages, name))
            
            # Add custom packages
            custom_pkgs = ui_components['custom_packages'].value.strip().split('\n')
            custom_pkgs = [pkg.strip() for pkg in custom_pkgs if pkg.strip()]
            if custom_pkgs:
                selected_packages.append((custom_pkgs, 'Custom packages'))
            
            # Check if anything to install
            if not selected_packages:
                status_queue.put({
                    'type': 'status',
                    'status_type': 'warning',
                    'message': 'Tidak ada package yang dipilih untuk diinstall'
                })
                status_queue.put({'type': 'complete'})
                return
            
            # Install packages one by one
            total = len(selected_packages)
            for i, (packages, name) in enumerate(selected_packages):
                progress = int((i / total) * 100)
                status_queue.put({
                    'type': 'progress',
                    'value': progress,
                    'description': f'Installing {name}...'
                })
                
                status_queue.put({
                    'type': 'status',
                    'status_type': 'info',
                    'message': f'🔄 Menginstall {name}...'
                })
                
                # Run pip install
                force = ui_components['force_reinstall'].value
                success, output = run_pip_install(packages, force)
                
                if success:
                    status_queue.put({
                        'type': 'status',
                        'status_type': 'success',
                        'message': f'{name} berhasil diinstall'
                    })
                else:
                    status_queue.put({
                        'type': 'status',
                        'status_type': 'error',
                        'message': f'Gagal install {name}: {output[:100]}...'
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
        packages_to_check = [
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
            ('termcolor', 'termcolor')
        ]
        
        with ui_components['status']:
            clear_output()
            display(create_status_indicator('info', '🔍 Memeriksa instalasi package...'))
            
            # Check each package
            for display_name, import_name in packages_to_check:
                try:
                    module = importlib.import_module(import_name)
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
        for key in [
            'yolov5_req', 'torch_req', 'smartcash_req', 
            'albumentations_req', 'notebook_req', 'opencv_req',
            'matplotlib_req', 'pandas_req', 'seaborn_req'
        ]:
            ui_components[key].value = True
    
    def on_uncheck_all(b):
        """Handler untuk tombol uncheck all."""
        for key in [
            'yolov5_req', 'torch_req', 'smartcash_req', 
            'albumentations_req', 'notebook_req', 'opencv_req',
            'matplotlib_req', 'pandas_req', 'seaborn_req'
        ]:
            ui_components[key].value = False
    
    # Register handlers
    ui_components['install_button'].on_click(on_install_click)
    ui_components['check_button'].on_click(on_check_click)
    ui_components['check_all_button'].on_click(on_check_all)
    ui_components['uncheck_all_button'].on_click(on_uncheck_all)
    
    # Add cleanup function
    def cleanup():
        """Clean up any resources."""
        # Empty queue
        while not status_queue.empty():
            status_queue.get()
    
    ui_components['cleanup'] = cleanup
    
    return ui_components