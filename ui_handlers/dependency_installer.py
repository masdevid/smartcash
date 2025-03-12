"""
File: smartcash/ui_handlers/dependency_installer.py
Author: Refactored
Deskripsi: Handler untuk UI instalasi dependencies SmartCash dengan implementasi ObserverManager.
"""

import sys
import subprocess
import time
import threading
import importlib
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets
def setup_dependency_handlers(ui_components):
    """Setup handler untuk komponen UI instalasi dependencies."""
    # Inisialisasi dependencies jika tersedia
    logger, observer_manager = None, None
    
    # Import utility functions
    try:
        from smartcash.utils.ui_utils import create_status_indicator
        from smartcash.utils.logger import get_logger
        from smartcash.utils.observer import EventDispatcher, EventTopics
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        logger = get_logger("dependency_installer")
        observer_manager = ObserverManager(auto_register=True)
        
        # Unregister any existing observers to prevent duplication
        observer_manager.unregister_group("dependency_observers")
        has_utilities = True
    except ImportError as e:
        print(f"ℹ️ Menggunakan mode fallback: {str(e)}")
        has_utilities = False
    
    # State variables
    is_installing = False
    installation_thread = None
    
    # Handler untuk check all button
    def on_check_all(b):
        checkboxes = [child for child in ui_components['checkbox_grid'].children 
                     if isinstance(child, type(ui_components['yolov5_req']))]
        for checkbox in checkboxes:
            checkbox.value = True
    
    # Handler untuk uncheck all button
    def on_uncheck_all(b):
        checkboxes = [child for child in ui_components['checkbox_grid'].children 
                     if isinstance(child, type(ui_components['yolov5_req']))]
        for checkbox in checkboxes:
            checkbox.value = False
    
    # Function to run pip install command
    def run_pip_command(package, force_reinstall=False):
        """Run pip install command and return result."""
        force_flag = "--force-reinstall" if force_reinstall else ""
        cmd = f"{sys.executable} -m pip install {package} {force_flag}"
        
        try:
            # Log command to detailed logs
            if 'log_output' in ui_components:
                with ui_components['log_output']:
                    print(f"Running: {cmd}")
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                capture_output=True, 
                text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
    
    # Special handlers for different package types
    def install_yolov5_requirements(force_reinstall=False):
        """Install YOLOv5 requirements."""
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
    
    def install_torch(force_reinstall=False):
        """Install PyTorch."""
        try:
            # Check if running in Colab
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
        
        if is_colab:
            return True, "PyTorch is already installed in Google Colab"
        
        # Install PyTorch for local environment
        force_flag = "--force-reinstall" if force_reinstall else ""
        cmd = f"{sys.executable} -m pip install torch torchvision torchaudio {force_flag}"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return True, "PyTorch installed successfully"
        except subprocess.CalledProcessError as e:
            return False, f"Error installing PyTorch: {e.stderr}"
    
    def install_smartcash_requirements(force_reinstall=False):
        """Install core requirements for SmartCash."""
        # Try to find smartcash/requirements.txt first
        from pathlib import Path
        req_path = Path("smartcash/requirements.txt")
        
        if req_path.exists():
            force_flag = "--force-reinstall" if force_reinstall else ""
            cmd = f"{sys.executable} -m pip install -r {req_path} {force_flag}"
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                return True, "SmartCash requirements installed successfully"
            except subprocess.CalledProcessError as e:
                return False, f"Error installing SmartCash requirements: {e.stderr}"
        else:
            # Fallback to manual installation of core packages
            force_flag = "--force-reinstall" if force_reinstall else ""
            cmd = f"{sys.executable} -m pip install pyyaml termcolor tqdm roboflow python-dotenv ipywidgets {force_flag}"
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                return True, "SmartCash requirements installed successfully (fallback method)"
            except subprocess.CalledProcessError as e:
                return False, f"Error installing SmartCash requirements: {e.stderr}"
    
    # Check if a package is installed
    def is_package_installed(package_name):
        """Check if package is installed."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    # Get installed package version
    def get_package_version(package_name):
        """Get package version."""
        try:
            module = importlib.import_module(package_name)
            return getattr(module, "__version__", "Unknown")
        except (ImportError, AttributeError):
            return "Not installed"
    
    # Create collapsible log output
    def setup_logs_accordion():
        # Check if accordion already exists in UI components
        if 'logs_accordion' not in ui_components:
            # Create log area for install details
            log_output = widgets.Output(
                layout={'max_height': '200px', 'overflow': 'auto', 'border': '1px solid #ddd'}
            )
            
            # Create accordion for logs
            logs_accordion = widgets.Accordion(
                children=[log_output],
                selected_index=None,  # Collapsed by default
                layout={'margin': '10px 0'}
            )
            logs_accordion.set_title(0, "📋 Installation Logs")
            
            # Add to UI
            ui_components['logs_accordion'] = logs_accordion
            ui_components['log_output'] = log_output
            
            # Add to main UI container after status output
            ui_container = ui_components['ui']
            children_list = list(ui_container.children)
            insert_index = -1  # Default to end
            
            # Find status output to insert after it
            for i, child in enumerate(children_list):
                if child is ui_components['status']:
                    insert_index = i + 1
                    break
                    
            # Insert accordion
            if insert_index >= 0 and insert_index <= len(children_list):
                new_children = children_list[:insert_index] + [logs_accordion] + children_list[insert_index:]
                ui_container.children = tuple(new_children)
            else:
                # Fallback: append to end
                ui_container.children = tuple(list(ui_container.children) + [logs_accordion])
    
    # Create processing indicator
    def setup_processing_indicator():
        if 'processing_indicator' not in ui_components:
            indicator = widgets.HTML(
                value='<div style="display:none;"><span style="color:#e74c3c;"><i class="fa fa-circle-o-notch fa-spin"></i> Installation in progress...</span></div>',
                layout={'margin': '5px 0'}
            )
            ui_components['processing_indicator'] = indicator
            
            # Add to UI near the buttons
            # First find the action container which should contain the install button
            for key, widget in ui_components.items():
                if isinstance(widget, widgets.HBox) and hasattr(widget, 'children'):
                    if any(child is ui_components['install_button'] for child in widget.children):
                        # This is the container with our buttons
                        action_group = widget
                        # Add the indicator
                        children_list = list(action_group.children)
                        action_group.children = tuple(children_list + [indicator])
                        break
    
    # Setup logs accordion and processing indicator
    setup_logs_accordion()
    setup_processing_indicator()
    
    # Function to update UI state during operations
    def set_ui_busy_state(busy=True, clear_status=False):
        """Set UI busy state - disable/enable buttons appropriately."""
        ui_components['install_button'].disabled = busy
        ui_components['check_button'].disabled = busy
        ui_components['check_all_button'].disabled = busy
        ui_components['uncheck_all_button'].disabled = busy
        
        # Show/hide progress as needed
        if busy:
            ui_components['install_progress'].layout.visibility = 'visible'
            ui_components['processing_indicator'].value = '<div><span style="color:#e74c3c;"><i class="fa fa-circle-o-notch fa-spin"></i> Installation in progress...</span></div>'
            
            # Ensure logs are visible during installation
            if 'logs_accordion' in ui_components:
                ui_components['logs_accordion'].selected_index = 0  # Expanded
        else:
            ui_components['install_progress'].layout.visibility = 'hidden'
            ui_components['processing_indicator'].value = '<div style="display:none;"><span style="color:#e74c3c;"><i class="fa fa-circle-o-notch fa-spin"></i> Installation in progress...</span></div>'
            
        # Clear status if requested
        if clear_status:
            with ui_components['status']:
                clear_output()
                
            # Also clear logs
            if 'log_output' in ui_components:
                with ui_components['log_output']:
                    clear_output()
    
    # Main handler function for installation
    def install_packages():
        nonlocal is_installing
        is_installing = True
        
        # Set UI to busy state and clear previous output
        set_ui_busy_state(True, clear_status=True)
        
        with ui_components['status']:
            clear_output()
            
            # Collect selected packages
            packages_to_install = []
            
            # Check standard packages
            package_checkers = [
                (ui_components['yolov5_req'].value, "YOLOv5 requirements", install_yolov5_requirements),
                (ui_components['torch_req'].value, "PyTorch", install_torch),
                (ui_components['smartcash_req'].value, "SmartCash requirements", install_smartcash_requirements),
                (ui_components['albumentations_req'].value, "albumentations", "albumentations"),
                (ui_components['notebook_req'].value, "Notebook tools", "ipywidgets tqdm matplotlib"),
                (ui_components['opencv_req'].value, "OpenCV", "opencv-python"),
                (ui_components['matplotlib_req'].value, "Matplotlib", "matplotlib"),
                (ui_components['pandas_req'].value, "Pandas", "pandas"),
                (ui_components['seaborn_req'].value, "Seaborn", "seaborn")
            ]
            
            for selected, name, pkg in package_checkers:
                if selected:
                    packages_to_install.append((name, pkg))
                    display(create_status_indicator("info", f"📋 {name} ditambahkan ke daftar instalasi"))
            
            # Add custom packages
            custom = ui_components['custom_packages'].value.strip()
            if custom:
                custom_packages = [pkg.strip() for pkg in custom.split('\n') if pkg.strip()]
                for pkg in custom_packages:
                    packages_to_install.append((f"Custom: {pkg}", pkg))
                    display(create_status_indicator("info", f"📋 Custom package {pkg} ditambahkan ke daftar instalasi"))
            
            if not packages_to_install:
                display(create_status_indicator("warning", "⚠️ Tidak ada package yang dipilih untuk diinstall"))
                is_installing = False
                return
            
            # Initialize progress bar
            ui_components['install_progress'].value = 0
            ui_components['install_progress'].max = len(packages_to_install)
            ui_components['install_progress'].description = "0%"
            
            # Force reinstall flag
            force_reinstall = ui_components['force_reinstall'].value
            
            # Install packages
            display(HTML("<h3>🚀 Memulai instalasi package</h3>"))
            
            for i, (name, pkg) in enumerate(packages_to_install):
                if not is_installing:
                    break
                    
                ui_components['install_progress'].value = i
                progress_pct = int((i+1) * 100 / len(packages_to_install))
                ui_components['install_progress'].description = f"{progress_pct}%"
                
                display(create_status_indicator("info", f"🔄 Menginstall {name}... ({i+1}/{len(packages_to_install)})"))
                
                # Install package
                try:
                    if callable(pkg):
                        # Custom function for special packages
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
            if is_installing:
                ui_components['install_progress'].value = len(packages_to_install)
                
                # Show completion message
                display(HTML(
                    """<div style="padding: 10px; background: #d4edda; color: #155724; border-left: 4px solid #28a745; margin-top: 20px;">
                        <h3 style="margin-top: 0;">✅ Instalasi Selesai</h3>
                        <p>Semua package telah diproses. Gunakan tombol 'Check Installations' untuk memeriksa hasil instalasi.</p>
                    </div>"""
                ))
            else:
                display(create_status_indicator("warning", "⚠️ Instalasi dihentikan oleh pengguna"))
            
            # Reset state
            is_installing = False
            
            # Restore UI state after brief delay (for visual effect)
            time.sleep(1)
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
        # Set UI to busy state and clear previous output
        set_ui_busy_state(True, clear_status=True)
        
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
                ('python-dotenv', 'dotenv'),
                ('Roboflow', 'roboflow')
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
            
            # Restore UI state
            set_ui_busy_state(False)
    
    # Register handlers
    ui_components['install_button'].on_click(on_install)
    ui_components['check_button'].on_click(on_check)
    ui_components['check_all_button'].on_click(on_check_all)
    ui_components['uncheck_all_button'].on_click(on_uncheck_all)
    
    # Cleanup function
    def cleanup():
        nonlocal is_installing
        is_installing = False
        
        # Restore UI state if needed
        set_ui_busy_state(False)
        
        if observer_manager:
            try:
                observer_manager.unregister_group("dependency_observers")
                if logger:
                    logger.info("✅ Observer untuk dependency installer telah dibersihkan")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat membersihkan observer: {str(e)}")
    
    # Add cleanup function to ui_components
    ui_components['cleanup'] = cleanup
    
    return ui_components