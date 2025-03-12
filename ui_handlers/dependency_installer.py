"""
File: smartcash/ui_handlers/dependency_installer.py
Deskripsi: Handler yang dioptimalkan untuk instalasi dependencies di lingkungan Colab.
"""

import os
import sys
import subprocess
import threading
import importlib
import queue
import time
from typing import List, Tuple, Optional, Dict, Any

# IPython imports
from IPython.display import display, clear_output, HTML

def setup_dependency_handlers(ui_components):
    """
    Setup handler untuk instalasi dependencies di Colab.
    
    Args:
        ui_components: Dictionary berisi komponen UI untuk instalasi
    
    Returns:
        Dictionary UI components yang sudah diupdate dengan handler
    """
    # Thread-safe queue untuk komunikasi
    status_queue = queue.Queue()
    
    def log_status(message: str, status_type: str = 'info'):
        """
        Log status dengan styling konsisten.
        """
        status_styles = {
            'info': {'icon': 'ℹ️', 'color': 'blue'},
            'success': {'icon': '✅', 'color': 'green'},
            'warning': {'icon': '⚠️', 'color': 'orange'},
            'error': {'icon': '❌', 'color': 'red'}
        }
        style = status_styles.get(status_type, status_styles['info'])
        
        with ui_components['status']:
            display(HTML(f"""
            <div style="margin: 5px 0; padding: 8px 12px; 
                        border-radius: 4px; background-color: #f8f9fa;">
                <span style="color: {style['color']}; font-weight: bold;"> 
                    {style['icon']} {message}
                </span>
            </div>
            """))
    
    def update_ui():
        """
        Thread untuk memperbarui UI secara berkala.
        """
        while True:
            try:
                # Coba ambil status dari queue dengan timeout
                status = status_queue.get(timeout=1)
                
                # Update UI berdasarkan status
                if status['type'] == 'progress':
                    ui_components['install_progress'].value = status['value']
                    ui_components['install_progress'].description = status['description']
                elif status['type'] == 'status':
                    log_status(status['message'], status['status_type'])
                elif status['type'] == 'complete':
                    # Reset UI state
                    ui_components['install_button'].disabled = False
                    ui_components['check_button'].disabled = False
                    ui_components['check_all_button'].disabled = False
                    ui_components['uncheck_all_button'].disabled = False
                    ui_components['install_progress'].layout.visibility = 'hidden'
                    break
            except queue.Empty:
                # Cek apakah thread utama masih berjalan
                if not threading.main_thread().is_alive():
                    break
    
    def run_pip_install(packages: List[str], force_reinstall: bool = False) -> Tuple[bool, str]:
        """
        Jalankan pip install untuk package tertentu.
        """
        force_flag = "--force-reinstall" if force_reinstall else ""
        extra_flags = "--no-deps" if any('torch' in pkg.lower() for pkg in packages) else ""
        
        # Khusus untuk requirements.txt
        if len(packages) == 1 and packages[0].endswith('.txt'):
            cmd = f"{sys.executable} -m pip install -r {packages[0]} {force_flag} -v"
        else:
            cmd = f"{sys.executable} -m pip install {' '.join(packages)} {force_flag} {extra_flags} -v"
        
        try:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Streaming output
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
            
            # Tunggu proses selesai
            return_code = process.wait()
            
            # Gabungkan output
            full_output = ''.join(output_lines)
            
            return return_code == 0, full_output
        except Exception as e:
            return False, str(e)
    
    def install_packages():
        """
        Proses instalasi package yang dipilih.
        """
        try:
            # Reset progress bar
            status_queue.put({
                'type': 'progress', 
                'value': 0, 
                'description': 'Memulai instalasi...'
            })
            
            # Mapping checkbox ke package, diurutkan dari paket terkecil
            package_map = {
                'notebook_req': (['tqdm', 'ipywidgets'], 'Notebook tools'),  # Paket terkecil dan ringan
                'smartcash_req': ([
                    'termcolor', 'pyyaml', 'python-dotenv'
                ], 'SmartCash utils'),
                'matplotlib_req': (['matplotlib'], 'Matplotlib'),
                'pandas_req': (['pandas'], 'Pandas'),
                'seaborn_req': (['seaborn'], 'Seaborn'),
                'opencv_req': (['opencv-python'], 'OpenCV'),
                'albumentations_req': (['albumentations'], 'Albumentations'),
                'torch_req': ([
                    'torch', 'torchvision', 'torchaudio', 
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ], 'PyTorch'),  # Paket paling besar dipindah ke akhir
                'yolov5_req': (['yolov5/requirements.txt'], 'YOLOv5 requirements')
            }
            
            # Fungsi untuk mengurutkan paket berdasarkan ukuran dan kompleksitas
            def sort_packages(packages_to_install):
                """
                Urutkan paket dari yang paling ringan ke yang paling berat.
                """
                package_weight = {
                    'Notebook tools': 1,
                    'SmartCash utils': 2,
                    'Matplotlib': 3,
                    'Pandas': 4,
                    'Seaborn': 5,
                    'OpenCV': 6,
                    'Albumentations': 7,
                    'PyTorch': 9,
                    'YOLOv5 requirements': 10
                }
                
                return sorted(
                    packages_to_install, 
                    key=lambda x: package_weight.get(x[1], 8)
                )
            
            # Daftar package untuk diinstall
            packages_to_install = []
            force_reinstall = ui_components['force_reinstall'].value
            
            # Cek package yang dipilih
            for key, (packages, display_name) in package_map.items():
                if ui_components[key].value:
                    packages_to_install.append((packages, display_name))
            
            # Tambahkan package kustom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            custom_packages = [pkg.strip() for pkg in custom_packages if pkg.strip()]
            if custom_packages:
                packages_to_install.append((custom_packages, 'Custom Packages'))
            
            # Cek apakah ada package yang akan diinstall
            if not packages_to_install:
                status_queue.put({
                    'type': 'status', 
                    'message': '⚠️ Tidak ada package yang dipilih untuk diinstall', 
                    'status_type': 'warning'
                })
                status_queue.put({'type': 'complete'})
                return
            
            # Urutkan paket
            packages_to_install = sort_packages(packages_to_install)
            
            # Proses instalasi
            total_packages = len(packages_to_install)
            for i, (packages, display_name) in enumerate(packages_to_install):
                # Update progress
                status_queue.put({
                    'type': 'progress', 
                    'value': int((i / total_packages) * 100), 
                    'description': f'Instalasi {display_name}'
                })
                
                # Kirim status memulai instalasi
                status_queue.put({
                    'type': 'status', 
                    'message': f'🔄 Menginstall {display_name}...', 
                    'status_type': 'info'
                })
                
                # Jalankan instalasi
                success, output = run_pip_install(packages, force_reinstall)
                
                # Kirim status hasil
                if success:
                    status_queue.put({
                        'type': 'status', 
                        'message': f'✅ {display_name} berhasil diinstall', 
                        'status_type': 'success'
                    })
                else:
                    status_queue.put({
                        'type': 'status', 
                        'message': f'❌ Gagal instalasi {display_name}: {output}', 
                        'status_type': 'error'
                    })
            
            # Selesai
            status_queue.put({
                'type': 'progress', 
                'value': 100, 
                'description': 'Instalasi selesai'
            })
            status_queue.put({
                'type': 'status', 
                'message': '🏁 Instalasi package selesai', 
                'status_type': 'success'
            })
            status_queue.put({'type': 'complete'})
        
        except Exception as e:
            # Tangani error yang tidak terduga
            status_queue.put({
                'type': 'status', 
                'message': f'❌ Error tidak terduga: {str(e)}', 
                'status_type': 'error'
            })
            status_queue.put({'type': 'complete'})
    
    def on_install_click(b):
        """
        Handler untuk tombol install.
        """
        # Reset UI
        ui_components['install_button'].disabled = True
        ui_components['check_button'].disabled = True
        ui_components['check_all_button'].disabled = True
        ui_components['uncheck_all_button'].disabled = True
        ui_components['install_progress'].layout.visibility = 'visible'
        ui_components['install_progress'].value = 0
        
        # Bersihkan queue
        while not status_queue.empty():
            status_queue.get()
        
        # Mulai thread update UI
        ui_update_thread = threading.Thread(target=update_ui)
        ui_update_thread.daemon = True
        ui_update_thread.start()
        
        # Mulai thread instalasi
        install_thread = threading.Thread(target=install_packages)
        install_thread.daemon = True
        install_thread.start()
    
    def on_check_click(b):
        """
        Handler untuk pengecekan instalasi package.
        """
        # Daftar package untuk dicek
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
            ('termcolor', 'termcolor'),
            ('Roboflow', 'roboflow'),
        ]
        
        # Bersihkan output sebelumnya
        with ui_components['status']:
            clear_output()
        
        # Cek instalasi
        results = check_package_installation(packages_to_check)
        
        # Tampilkan hasil
        log_status("🔍 Hasil Pemeriksaan Package:", 'info')
        for result in results:
            if result['status'] == 'installed':
                log_status(f"✅ {result['name']}: v{result['version']}", 'success')
            else:
                log_status(f"⚠️ {result['name']}: Tidak terinstall", 'warning')
        
        # Cek CUDA untuk PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                log_status(f"✅ CUDA tersedia: {device_name}", 'success')
            else:
                log_status("ℹ️ CUDA tidak tersedia, akan menggunakan CPU", 'info')
        except ImportError:
            pass
    
    def check_package_installation(packages: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Periksa status instalasi package.
        """
        results = []
        for display_name, import_name in packages:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'Unknown')
                results.append({
                    'name': display_name,
                    'status': 'installed',
                    'version': version
                })
            except ImportError:
                results.append({
                    'name': display_name,
                    'status': 'not_installed',
                    'version': None
                })
        return results
    
    def on_check_all(b):
        """
        Handler untuk mencentang semua checkbox.
        """
        checkboxes = [
            'yolov5_req', 'torch_req', 'smartcash_req', 
            'albumentations_req', 'notebook_req', 
            'opencv_req', 'matplotlib_req', 
            'pandas_req', 'seaborn_req'
        ]
        for name in checkboxes:
            ui_components[name].value = True
    
    def on_uncheck_all(b):
        """
        Handler untuk menghapus centang semua checkbox.
        """
        checkboxes = [
            'yolov5_req', 'torch_req', 'smartcash_req', 
            'albumentations_req', 'notebook_req', 
            'opencv_req', 'matplotlib_req', 
            'pandas_req', 'seaborn_req'
        ]
        for name in checkboxes:
            ui_components[name].value = False
    
    # Registrasi event handlers
    ui_components['install_button'].on_click(on_install_click)
    ui_components['check_button'].on_click(on_check_click)
    ui_components['check_all_button'].on_click(on_check_all)
    ui_components['uncheck_all_button'].on_click(on_uncheck_all)
    
    # Fungsi cleanup
    def cleanup():
        """
        Membersihkan sumber daya yang digunakan.
        """
        # Kosongkan queue
        while not status_queue.empty():
            status_queue.get()
    
    ui_components['cleanup'] = cleanup
    
    return ui_components