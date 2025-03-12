"""
File: smartcash/ui_handlers/dependency_installer.py
Deskripsi: Handler yang dioptimalkan untuk instalasi dependencies di lingkungan Colab.
"""

import os
import sys
import subprocess
import threading
import importlib
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
    # State variables untuk kontrol proses instalasi
    installation_state = {
        'is_installing': False,
        'installation_thread': None
    }
    
    # Logger minimal untuk pesan status
    def log_status(message: str, status_type: str = 'info'):
        """
        Log status dengan styling konsisten.
        
        Args:
            message: Pesan status
            status_type: Tipe status ('info', 'success', 'warning', 'error')
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
    
    def set_ui_state(busy: bool = False):
        """
        Atur status UI selama proses instalasi.
        
        Args:
            busy: Status sibuk atau tidak
        """
        ui_components['install_button'].disabled = busy
        ui_components['check_button'].disabled = busy
        ui_components['check_all_button'].disabled = busy
        ui_components['uncheck_all_button'].disabled = busy
        ui_components['install_progress'].layout.visibility = 'visible' if busy else 'hidden'
    
    def run_pip_install(packages: List[str], force_reinstall: bool = False) -> Tuple[bool, str]:
        """
        Jalankan pip install untuk package tertentu.
        
        Args:
            packages: Daftar package untuk diinstall
            force_reinstall: Flag untuk force reinstall
        
        Returns:
            Tuple (berhasil, pesan)
        """
        force_flag = "--force-reinstall" if force_reinstall else ""
        
        # Tambahkan opsi untuk menghindari konfllik torch
        extra_flags = "--no-deps" if any('torch' in pkg.lower() for pkg in packages) else ""
        
        cmd = f"{sys.executable} -m pip install {' '.join(packages)} {force_flag} {extra_flags}"
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True, 
                stderr=subprocess.STDOUT
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.output
    
    def install_yolov5_requirements(force_reinstall: bool = False) -> Tuple[bool, str]:
        """
        Install requirements untuk YOLOv5.
        
        Args:
            force_reinstall: Flag untuk force reinstall
        
        Returns:
            Tuple (berhasil, pesan)
        """
        from pathlib import Path
        
        # Cek keberadaan file requirements
        requirements_path = Path("yolov5/requirements.txt")
        if not requirements_path.exists():
            return False, "File requirements.txt untuk YOLOv5 tidak ditemukan."
        
        force_flag = "--force-reinstall" if force_reinstall else ""
        cmd = f"{sys.executable} -m pip install -r {requirements_path} {force_flag}"
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True, 
                stderr=subprocess.STDOUT
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.output
    
    def check_package_installation(packages: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Periksa status instalasi package.
        
        Args:
            packages: List tuple (nama_display, nama_import)
        
        Returns:
            List status instalasi package
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
    
    def install_packages():
        """
        Proses instalasi package yang dipilih.
        """
        # Pastikan tidak ada proses instalasi yang sedang berjalan
        def finalize_installation(success=True):
            """Finalisasi proses instalasi."""
            try:
                with ui_components['status']:
                    progress_bar = ui_components['install_progress']
                    progress_bar.value = progress_bar.max
                    progress_bar.description = "Selesai"
                    
                    if success:
                        log_status("🏁 Instalasi package selesai", 'success')
                    
                # Reset UI state di main thread
                import IPython.display
                IPython.display.display(IPython.display.HTML("""
                <script>
                document.querySelector('.jupyter-widgets[data-comm-id="' + 
                    document.querySelector('.jupyter-widgets[title="Install Packages"]').getAttribute('data-comm-id') + 
                    '"]').disabled = false;
                </script>
                """))
            except Exception as e:
                print(f"Error in finalization: {e}")
            finally:
                # Pastikan state direset
                installation_state['is_installing'] = False
                set_ui_state(busy=False)
        
        try:
            # Cegah proses berulang
            if installation_state['is_installing']:
                return
            
            # Set state instalasi
            installation_state['is_installing'] = True
            
            # Gunakan main thread untuk update UI
            import IPython.display
            IPython.display.display(IPython.display.HTML("""
            <script>
            document.querySelector('.jupyter-widgets[title="Install Packages"]').disabled = true;
            </script>
            """))
            
            # Bersihkan status sebelumnya
            with ui_components['status']:
                clear_output()
            
            # Mapping checkbox ke package
            package_map = {
                'yolov5_req': (install_yolov5_requirements, 'YOLOv5 requirements'),
                'torch_req': (
                    lambda force: run_pip_install([
                        'torch', 'torchvision', 'torchaudio', 
                        '--index-url', 'https://download.pytorch.org/whl/cu118'
                    ], force), 
                    'PyTorch'
                ),
                'smartcash_req': (
                    lambda force: run_pip_install([
                        'pyyaml', 'termcolor', 'tqdm', 'roboflow', 
                        'python-dotenv', 'ipywidgets'
                    ], force), 
                    'SmartCash requirements'
                ),
                'albumentations_req': (
                    lambda force: run_pip_install(['albumentations'], force), 
                    'Albumentations'
                ),
                'notebook_req': (
                    lambda force: run_pip_install(['ipywidgets', 'tqdm', 'matplotlib'], force), 
                    'Notebook tools'
                ),
                'opencv_req': (
                    lambda force: run_pip_install(['opencv-python'], force), 
                    'OpenCV'
                ),
                'matplotlib_req': (
                    lambda force: run_pip_install(['matplotlib'], force), 
                    'Matplotlib'
                ),
                'pandas_req': (
                    lambda force: run_pip_install(['pandas'], force), 
                    'Pandas'
                ),
                'seaborn_req': (
                    lambda force: run_pip_install(['seaborn'], force), 
                    'Seaborn'
                )
            }
            
            # Daftar package untuk diinstall
            packages_to_install = []
            force_reinstall = ui_components['force_reinstall'].value
            
            # Cek package yang dipilih
            for key, (install_func, display_name) in package_map.items():
                if ui_components[key].value:
                    packages_to_install.append((install_func, display_name))
            
            # Tambahkan package kustom
            custom_packages = ui_components['custom_packages'].value.strip().split('\n')
            custom_packages = [pkg.strip() for pkg in custom_packages if pkg.strip()]
            if custom_packages:
                custom_install_func = lambda force: run_pip_install(custom_packages, force)
                packages_to_install.append((custom_install_func, 'Custom Packages'))
            
            # Cek apakah ada package yang akan diinstall
            if not packages_to_install:
                log_status("⚠️ Tidak ada package yang dipilih untuk diinstall", 'warning')
                finalize_installation(False)
                return
            
            # Setup progress bar
            progress_bar = ui_components['install_progress']
            progress_bar.value = 0
            progress_bar.max = len(packages_to_install)
            progress_bar.layout.visibility = 'visible'
            
            # Proses instalasi
            install_success = True
            for i, (install_func, display_name) in enumerate(packages_to_install):
                log_status(f"🔄 Instalasi {display_name}...", 'info')
                
                # Jalankan instalasi
                success, output = install_func(force_reinstall)
                
                # Update progress
                progress_bar.value = i + 1
                progress_bar.description = f"{display_name}"
                
                # Tampilkan hasil
                if success:
                    log_status(f"✅ {display_name} berhasil diinstall", 'success')
                else:
                    log_status(f"❌ Gagal instalasi {display_name}: {output}", 'error')
                    install_success = False
            
            # Finalisasi
            finalize_installation(install_success)
        
        except Exception as e:
            log_status(f"❌ Error fatal: {str(e)}", 'error')
            finalize_installation(False)
    
    def on_install_click(b):
        """
        Handler untuk tombol install.
        """
        # Hindari multiple thread
        if not installation_state['is_installing']:
            thread = threading.Thread(target=install_packages)
            thread.daemon = True
            thread.start()
            installation_state['installation_thread'] = thread
    
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
        # Pastikan tidak ada thread yang masih berjalan
        if installation_state['installation_thread'] and installation_state['installation_thread'].is_alive():
            installation_state['is_installing'] = False
            installation_state['installation_thread'].join(timeout=1)
    
    ui_components['cleanup'] = cleanup
    
    return ui_components