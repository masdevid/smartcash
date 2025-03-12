"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Refactored
Deskripsi: Handler yang dioptimalkan untuk UI instalasi dependencies SmartCash.
"""

import sys
import subprocess
import threading
import importlib
from IPython.display import display, clear_output, HTML

def setup_dependency_handlers(ui_components):
    """Setup handler untuk komponen UI instalasi dependencies."""
    # State variables
    is_installing = False
    installation_thread = None
    
    # Import utilities jika tersedia
    try:
        from smartcash.utils.ui_utils import create_status_indicator
        from smartcash.utils.logger import get_logger
        logger = get_logger("dependency_installer")
        has_utils = True
    except ImportError:
        has_utils = False
        
        # Fallback untuk create_status_indicator jika utils tidak tersedia
        def create_status_indicator(status, message):
            status_styles = {
                'success': {'icon': '✅', 'color': 'green'},
                'warning': {'icon': '⚠️', 'color': 'orange'},
                'error': {'icon': '❌', 'color': 'red'},
                'info': {'icon': 'ℹ️', 'color': 'blue'}
            }
            style = status_styles.get(status, status_styles['info'])
            return HTML(f"""
            <div style="margin:5px 0;padding:8px 12px;border-radius:4px;background-color:#f8f9fa;">
                <span style="color:{style['color']};font-weight:bold;">{style['icon']} {message}</span>
            </div>
            """)
    
    # Handler untuk check all button
    def on_check_all(b):
        checkboxes = [child for child in ui_components['checkbox_grid'].children 
                     if hasattr(child, 'value')]
        for checkbox in checkboxes:
            checkbox.value = True
    
    # Handler untuk uncheck all button
    def on_uncheck_all(b):
        checkboxes = [child for child in ui_components['checkbox_grid'].children 
                     if hasattr(child, 'value')]
        for checkbox in checkboxes:
            checkbox.value = False
    
    # Function untuk menjalankan pip install
    def run_pip_command(package, force_reinstall=False):
        force_flag = "--force-reinstall" if force_reinstall else ""
        # Use --quiet for less verbose output but still show errors
        cmd = f"{sys.executable} -m pip install {package} {force_flag}"
        
        # Log the command being executed
        with ui_components['status']:
            display(HTML(f"<p><code>{cmd}</code></p>"))
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
    
    # Special handlers for package types
    def install_yolov5_requirements(force_reinstall=False):
        from pathlib import Path
        if Path("yolov5").exists() and Path("yolov5/requirements.txt").exists():
            force_flag = "--force-reinstall" if force_reinstall else ""
            cmd = f"{sys.executable} -m pip install -r yolov5/requirements.txt {force_flag}"
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                return True, "YOLOv5 requirements installed successfully"
            except subprocess.CalledProcessError as e:
                return False, f"Error installing YOLOv5 requirements: {e.stderr}"
        else:
            return False, "YOLOv5 directory not found. Please clone the repository first."
    
    # Check if package is installed
    def is_package_installed(package_name):
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    # Get installed package version
    def get_package_version(package_name):
        try:
            module = importlib.import_module(package_name)
            return getattr(module, "__version__", "Unknown")
        except (ImportError, AttributeError):
            return "Not installed"
    
    # Toggle UI state during installation
    def set_ui_busy_state(busy=True, clear_status=False):
        ui_components['install_button'].disabled = busy
        ui_components['check_button'].disabled = busy
        ui_components['check_all_button'].disabled = busy
        ui_components['uncheck_all_button'].disabled = busy
        
        # Always set visibility explicitly
        ui_components['install_progress'].layout.visibility = 'visible' if busy else 'hidden'
        
        # Clear status if requested
        if clear_status:
            with ui_components['status']:
                clear_output()
    
    # Main installation function
    def install_packages():
        nonlocal is_installing
        is_installing = True
        
        # Show progress bar and disable buttons
        ui_components['install_progress'].layout.visibility = 'visible'
        # Set UI to busy state
        set_ui_busy_state(True, clear_status=True)
        
        with ui_components['status']:
            # Collect selected packages
            packages_to_install = []
            
            # Checkbox mappings
            checkbox_map = {
                'yolov5_req': ("YOLOv5 requirements", install_yolov5_requirements),
                'torch_req': ("PyTorch", "torch torchvision torchaudio"),
                'smartcash_req': ("SmartCash requirements", "pyyaml termcolor tqdm roboflow python-dotenv ipywidgets"),
                'albumentations_req': ("Albumentations", "albumentations"),
                'notebook_req': ("Notebook tools", "ipywidgets tqdm matplotlib"),
                'opencv_req': ("OpenCV", "opencv-python"),
                'matplotlib_req': ("Matplotlib", "matplotlib"),
                'pandas_req': ("Pandas", "pandas"),
                'seaborn_req': ("Seaborn", "seaborn")
            }
            
            # Add selected packages
            any_checkbox_selected = False
            for key, (name, pkg) in checkbox_map.items():
                if key in ui_components and ui_components[key].value:
                    any_checkbox_selected = True
                    packages_to_install.append((name, pkg))
                    display(create_status_indicator("info", f"📋 {name} ditambahkan ke daftar instalasi"))
            
            # Add custom packages
            custom = ui_components['custom_packages'].value.strip()
            has_custom_packages = False
            if custom:
                custom_packages = [pkg.strip() for pkg in custom.split('\n') if pkg.strip()]
                has_custom_packages = len(custom_packages) > 0
                for pkg in custom_packages:
                    packages_to_install.append((f"Custom: {pkg}", pkg))
                    display(create_status_indicator("info", f"📋 Custom package {pkg} ditambahkan ke daftar instalasi"))
            
            # Only proceed if we have packages to install
            if not packages_to_install:
                display(create_status_indicator("warning", "⚠️ Tidak ada package yang dipilih untuk diinstall"))
                is_installing = False
                set_ui_busy_state(False)
                return
                
            # Warn if nothing is checked but custom packages exist
            if not any_checkbox_selected and has_custom_packages:
                display(create_status_indicator("info", "ℹ️ Hanya menginstall custom packages karena tidak ada package standard yang dipilih"))
            
            # Setup progress bar
            ui_components['install_progress'].value = 0
            ui_components['install_progress'].max = len(packages_to_install)
            
            # Force reinstall flag
            force_reinstall = ui_components['force_reinstall'].value
            
            # Install packages
            display(HTML("<h3>🚀 Memulai instalasi package</h3>"))
            
            for i, (name, pkg) in enumerate(packages_to_install):
                ui_components['install_progress'].value = i
                progress_pct = int((i+1) * 100 / len(packages_to_install))
                ui_components['install_progress'].description = f"{progress_pct}%"
                
                display(create_status_indicator("info", f"🔄 Menginstall {name}... ({i+1}/{len(packages_to_install)})"))
                
                # Install package
                try:
                    if callable(pkg):
                        # Custom install function
                        success, message = pkg(force_reinstall)
                    else:
                        # Standard pip install
                        success, message = run_pip_command(pkg, force_reinstall)
                    
                    if success:
                        display(create_status_indicator("success", f"✅ {name} berhasil diinstall"))
                    else:
                        display(create_status_indicator("error", f"❌ Error saat install {name}: {message}"))
                except Exception as e:
                    display(create_status_indicator("error", f"❌ Exception saat install {name}: {str(e)}"))
            
            # Installation complete
            ui_components['install_progress'].value = len(packages_to_install)
            display(HTML(
                """<div style="padding:10px;background:#d4edda;color:#155724;border-left:4px solid #28a745;margin-top:20px;">
                    <h3 style="margin-top:0;">✅ Instalasi Selesai</h3>
                    <p>Semua package telah diproses. Gunakan tombol 'Check Installations' untuk memeriksa hasil instalasi.</p>
                </div>"""
            ))
            
            # Reset state
            is_installing = False
            set_ui_busy_state(False)
    
    # Handler for install button
    def on_install(b):
        nonlocal installation_thread
        
        # Don't start if already installing
        if is_installing:
            return
        
        # Start installation in separate thread
        installation_thread = threading.Thread(target=install_packages)
        installation_thread.daemon = True
        installation_thread.start()
    
    # Handler for check button
    def on_check(b):
        with ui_components['status']:
            clear_output()
            
            display(HTML("<h3>🔍 Checking installed packages</h3>"))
            
            # List of packages to check
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
                ('python-dotenv', 'python-dotenv'),
            ]
            
            # Check packages
            for name, pkg in packages_to_check:
                installed = is_package_installed(pkg)
                version = get_package_version(pkg) if installed else "Not installed"
                
                if installed:
                    display(create_status_indicator("success", f"✅ {name}: v{version}"))
                else:
                    display(create_status_indicator("warning", f"⚠️ {name}: Not installed"))
            
            # Check CUDA if PyTorch is installed
            if is_package_installed('torch'):
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    try:
                        device_name = torch.cuda.get_device_name(0)
                        display(create_status_indicator("success", f"✅ CUDA available: {device_name}"))
                    except:
                        display(create_status_indicator("success", f"✅ CUDA is available"))
                else:
                    display(create_status_indicator("info", "ℹ️ CUDA is not available, using CPU only"))
    
    # Register handlers
    ui_components['install_button'].on_click(on_install)
    ui_components['check_button'].on_click(on_check)
    ui_components['check_all_button'].on_click(on_check_all)
    ui_components['uncheck_all_button'].on_click(on_uncheck_all)
    
    # Add cleanup function
    def cleanup():
        nonlocal is_installing
        is_installing = False
        set_ui_busy_state(False)
    
    ui_components['cleanup'] = cleanup
    
    return ui_components