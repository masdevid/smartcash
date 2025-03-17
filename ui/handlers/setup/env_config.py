"""
File: smartcash/ui/handlers/setup/env_config.py
Deskripsi: Handler untuk UI konfigurasi environment SmartCash di subdirektori setup
"""

import os
import shutil
import threading
import time
from IPython.display import display, HTML, clear_output
from pathlib import Path

from smartcash.ui.components.shared.alerts import create_info_alert, create_status_indicator
from smartcash.ui.handlers.shared.error_handler import handle_error

def setup_env_config_handlers(ui_components, env=None, config=None):
    """Setup handlers untuk UI konfigurasi environment SmartCash."""
    
    # Print debugging info to help troubleshoot
    with ui_components['status']:
        clear_output(wait=True)
        display(HTML("<p>🚀 Environment Config handler initialized</p>"))
        
        # Display current working directory for debugging
        cwd = Path.cwd()
        display(HTML(f"<p>Current directory: <code>{cwd}</code></p>"))
        
        # Check for configs folder
        config_dir = Path('configs')
        if config_dir.exists():
            config_files = list(config_dir.glob('*.y*ml'))
            display(HTML(f"<p>Found {len(config_files)} config files in configs folder</p>"))
        else:
            display(HTML("<p>⚠️ configs folder not found, will create</p>"))
            try:
                config_dir.mkdir(exist_ok=True)
            except Exception as e:
                display(HTML(f"<p>❌ Error creating configs folder: {str(e)}</p>"))
    
    # Initialize variables
    logger = env_manager = observer_manager = config_manager = None
    
    # Try to import SmartCash modules
    try:
        # Import modules
        from smartcash.common.logger import get_logger
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        logger = get_logger("env_config")
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        
        # Try to load config directly
        if config_manager:
            try:
                if Path('configs/colab_config.yaml').exists():
                    config = config_manager.load_config('configs/colab_config.yaml')
                elif Path('configs/base_config.yaml').exists():
                    config = config_manager.load_config('configs/base_config.yaml')
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Could not load config: {str(e)}")
        
        # Setup observer if available
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            observer_manager = ObserverManager.get_instance()
            if observer_manager:
                observer_manager.unregister_group("env_config_observers")
        except ImportError:
            pass
    except ImportError as e:
        with ui_components['status']:
            display(HTML(f"<p style='color:orange'>⚠️ Limited functionality mode - SmartCash modules not fully loaded: {str(e)}</p>"))
    
    # Helper functions
    def filter_drive_tree(tree_html):
        """Filter directory tree to focus on SmartCash"""
        if not tree_html or '/content/drive' not in tree_html:
            return tree_html
            
        try:
            pre_start = tree_html.find("<pre")
            pre_end = tree_html.find("</pre>")
            
            if pre_start == -1 or pre_end == -1:
                return tree_html
                
            header = tree_html[:pre_start + tree_html[pre_start:].find(">") + 1]
            content = tree_html[pre_start + tree_html[pre_start:].find(">") + 1:pre_end]
            
            lines = content.split("\n")
            filtered_lines = []
            inside_drive = False
            
            for line in lines:
                if '/content/drive' in line and 'MyDrive/SmartCash' not in line and not inside_drive:
                    continue
                    
                if 'SmartCash/' in line:
                    inside_drive = True
                    filtered_lines.append(line)
                elif inside_drive and ('│' not in line and '├' not in line and '└' not in line):
                    inside_drive = False
                elif inside_drive:
                    filtered_lines.append(line)
                elif '/content/drive' not in line:
                    filtered_lines.append(line)
            
            return header + "\n".join(filtered_lines) + "</pre>"
        except Exception:
            return tree_html
    
    def fallback_get_directory_tree(root_dir, max_depth=2):
        """Basic directory tree visualization without relying on environment manager"""
        root_dir = Path(root_dir)
        if not root_dir.exists():
            return f"<span style='color:red'>❌ Directory not found: {root_dir}</span>"
        
        # Focus on SmartCash folder for drive
        if '/content/drive' in str(root_dir):
            root_dir = Path('/content/drive/MyDrive/SmartCash')
            if not root_dir.exists():
                root_dir.mkdir(parents=True, exist_ok=True)
        
        result = "<pre style='margin:0;padding:5px;background:#f8f9fa;font-family:monospace;color:#333'>\n"
        result += f"<span style='color:#0366d6;font-weight:bold'>{root_dir.name}/</span>\n"
        
        def traverse_dir(path, prefix="", depth=0):
            if depth > max_depth: 
                return ""
                
            try:
                items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
            except PermissionError:
                return f"{prefix}└─ <span style='color:red'>❌ Permission denied</span>\n"
                
            tree = ""
            for i, item in enumerate(items):
                # Skip if it's not SmartCash related in drive
                if '/content/drive/MyDrive' in str(item) and '/SmartCash' not in str(item):
                    continue
                    
                is_last = i == len(items) - 1
                connector = "└─ " if is_last else "├─ "
                if item.is_dir():
                    tree += f"{prefix}{connector}<span style='color:#0366d6;font-weight:bold'>{item.name}/</span>\n"
                    next_prefix = prefix + ("   " if is_last else "│  ")
                    if depth < max_depth:
                        tree += traverse_dir(item, next_prefix, depth + 1)
                else:
                    tree += f"{prefix}{connector}{item.name}\n"
            return tree
        
        result += traverse_dir(root_dir)
        result += "</pre>"
        return result
    
    def sync_configs(source_dirs, target_dirs):
        """Sync config files from source to target directories"""
        total_files = copied_files = 0
        
        try:
            for source_dir in source_dirs:
                if not isinstance(source_dir, Path):
                    source_dir = Path(source_dir)
                
                if not source_dir.exists() or not source_dir.is_dir():
                    continue
                
                config_files = list(source_dir.glob('*.y*ml'))
                
                for config_file in config_files:
                    total_files += 1
                    
                    for target_dir in target_dirs:
                        if not isinstance(target_dir, Path):
                            target_dir = Path(target_dir)
                        
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_file = target_dir / config_file.name
                        
                        if not target_file.exists():
                            try:
                                shutil.copy2(config_file, target_file)
                                copied_files += 1
                                if logger:
                                    logger.info(f"✅ Copied {config_file.name} to {target_dir}")
                            except Exception as e:
                                if logger:
                                    logger.warning(f"⚠️ Failed to copy {config_file.name}: {str(e)}")
            
            return total_files, copied_files
        except Exception as e:
            if logger:
                logger.error(f"❌ Error syncing configs: {str(e)}")
            return total_files, copied_files
    
    def detect_environment():
        """Detect environment and update UI"""
        is_colab = False
        if env_manager:
            is_colab = env_manager.is_colab
            with ui_components['info_panel']:
                clear_output(wait=True)
                try:
                    system_info = env_manager.get_system_info()
                    info_html = f"""
                    <div style="background:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;color:#212529">
                        <h4 style="margin-top:0">📊 System Information</h4>
                        <ul>
                            <li><b>Python:</b> {system_info.get('python_version', 'Unknown')}</li>
                            <li><b>Base Directory:</b> {system_info.get('base_directory', 'Unknown')}</li>
                            <li><b>CUDA Available:</b> {'Yes' if system_info.get('cuda', {}).get('available', False) else 'No'}</li>
                        </ul>
                    </div>
                    """
                    display(HTML(info_html))
                except Exception as e:
                    display(HTML(f"<p>⚠️ Error getting system info: {str(e)}</p>"))
        else:
            try:
                import google.colab
                is_colab = True
            except ImportError:
                pass
                
            # Simple system information in fallback mode
            with ui_components['info_panel']:
                clear_output(wait=True)
                import sys, platform
                display(HTML(f"""
                <div style="background:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;color:#212529">
                    <h4 style="margin-top:0">📊 System Information</h4>
                    <ul>
                        <li><b>Python:</b> {platform.python_version()}</li>
                        <li><b>OS:</b> {platform.system()} {platform.release()}</li>
                        <li><b>Base Directory:</b> {Path.cwd()}</li>
                    </ul>
                </div>
                """))
        
        # Update UI based on environment
        ui_components['colab_panel'].value = """
            <div style="padding:10px;background:#d1ecf1;border-left:4px solid #0c5460;color:#0c5460;margin:10px 0">
                <h3 style="margin-top:0; color: inherit">☁️ Google Colab Terdeteksi</h3>
                <p>Project akan dikonfigurasi untuk berjalan di Google Colab. Koneksi ke Google Drive direkomendasikan.</p>
            </div>
        """ if is_colab else """
            <div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                <h3 style="margin-top:0; color: inherit">💻 Environment Lokal Terdeteksi</h3>
                <p>Project akan dikonfigurasi untuk berjalan di environment lokal.</p>
            </div>
        """
        
        return is_colab
    
    def check_smartcash_dir():
        """Check if SmartCash directory exists"""
        if not Path('smartcash').exists() or not Path('smartcash').is_dir():
            with ui_components['status']:
                clear_output(wait=True)
                alert_html = f"""
                <div style="padding:15px;background-color:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0;border-radius:4px">
                    <h3 style="margin-top:0">❌ Folder SmartCash tidak ditemukan!</h3>
                    <p>Repository belum di-clone dengan benar. Silakan jalankan cell clone repository terlebih dahulu.</p>
                    <ol>
                        <li>Jalankan cell repository clone (Cell 1.1)</li>
                        <li>Restart runtime (Runtime > Restart runtime)</li>
                        <li>Jalankan kembali notebook dari awal</li>
                    </ol>
                </div>
                """
                display(HTML(alert_html))
                return False
        return True
    
    # Initialize button handlers with explicit debugging
    def on_drive_connect(b):
        """Handler koneksi Google Drive"""
        with ui_components['status']:
            clear_output(wait=True)
            
            display(HTML("<p>🔄 Starting Google Drive connection...</p>"))
            
            try:
                # Always use the direct Colab approach for reliability
                from google.colab import drive
                display(HTML("<p>🔄 Mounting Google Drive...</p>"))
                drive.mount('/content/drive')
                
                # Setup dirs
                drive_path = Path('/content/drive/MyDrive/SmartCash')
                drive_path.mkdir(parents=True, exist_ok=True)
                display(HTML(f'<p>✅ SmartCash directory ready: <code>{drive_path}</code></p>'))
                
                # Create symlink
                if not Path('SmartCash_Drive').exists():
                    os.symlink(drive_path, 'SmartCash_Drive')
                    display(HTML('<p>✅ Symlink <code>SmartCash_Drive</code> created</p>'))
                else:
                    display(HTML('<p>ℹ️ Symlink already exists</p>'))
                
                # Sync configs
                configs_dir = drive_path / 'configs'
                configs_dir.mkdir(exist_ok=True)
                
                # Create local configs dir if it doesn't exist
                Path('configs').mkdir(exist_ok=True)
                
                # Copy default configs if needed
                for config_name in ['base_config.yaml', 'colab_config.yaml']:
                    source_path = Path(f'configs/{config_name}')
                    if not source_path.exists() and Path(f'/content/{config_name}').exists():
                        shutil.copy2(Path(f'/content/{config_name}'), source_path)
                
                # Sync configs between local and Drive
                local_configs = list(Path('configs').glob('*.y*ml'))
                
                missing_count = copied_count = 0
                if local_configs:
                    for config_file in local_configs:
                        drive_file = configs_dir / config_file.name
                        if not drive_file.exists():
                            missing_count += 1
                            try:
                                shutil.copy2(config_file, drive_file)
                                copied_count += 1
                                display(HTML(f'<p>✅ Copied <code>{config_file.name}</code> to Drive</p>'))
                            except Exception as e:
                                display(HTML(f'<p>⚠️ Failed to copy <code>{config_file.name}</code>: {str(e)}</p>'))
                
                display(HTML(
                    """<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                        <h3 style="margin-top:0">✅ Google Drive Connected</h3>
                        <p>Data will be stored in <code>/content/drive/MyDrive/SmartCash</code></p>
                    </div>"""
                ))
                
                # Display directory tree
                display(HTML("<h4>📂 SmartCash Directory Structure:</h4>"))
                display(HTML(fallback_get_directory_tree(drive_path, max_depth=2)))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to connect to Google Drive</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
    
    def on_dir_setup(b):
        """Setup directory structure"""
        with ui_components['status']:
            clear_output(wait=True)
            
            display(HTML("<p>🔄 Setting up directory structure...</p>"))
            
            try:
                # Define directories to create
                dirs = [
                    'data/train/images', 'data/train/labels',
                    'data/valid/images', 'data/valid/labels',
                    'data/test/images', 'data/test/labels',
                    'configs', 'runs/train/weights', 'logs', 'exports'
                ]
                
                created = existing = 0
                
                # Create directories
                for d in dirs:
                    path = Path(d)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        created += 1
                        display(HTML(f'<p>✅ Created directory: <code>{d}</code></p>'))
                    else:
                        existing += 1
                
                # Check if we're in Colab and Drive is mounted
                drive_mounted = Path('/content/drive/MyDrive').exists()
                
                if drive_mounted:
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    if not drive_path.exists():
                        drive_path.mkdir(parents=True, exist_ok=True)
                        display(HTML(f'<p>✅ Created SmartCash directory in Drive</p>'))
                    
                    # Create the same structure in Drive
                    for d in dirs:
                        drive_dir = drive_path / d
                        if not drive_dir.exists():
                            drive_dir.mkdir(parents=True, exist_ok=True)
                            created += 1
                            display(HTML(f'<p>✅ Created directory in Drive: <code>{d}</code></p>'))
                
                # Show completion message
                display(HTML(
                    f"""<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                        <h3 style="margin-top:0">✅ Directory Structure Created</h3>
                        <p>New directories: {created}, Existing: {existing}</p>
                        <pre style="margin:10px 0 0 10px;color:#155724;background:transparent;border:none">
data/
  ├── train/images/ & labels/
  ├── valid/images/ & labels/
  └── test/images/ & labels/
configs/
runs/train/weights/
logs/
exports/</pre>
                    </div>"""
                ))
                
                # Display tree
                display(HTML("<h4>📂 Project Directory Structure:</h4>"))
                display(HTML(fallback_get_directory_tree(Path.cwd(), max_depth=2)))
                
                if drive_mounted:
                    display(HTML("<h4>📂 Drive Directory Structure:</h4>"))
                    display(HTML(fallback_get_directory_tree(drive_path, max_depth=2)))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to set up directory structure</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
    
    # Register event handlers directly
    ui_components['drive_button'].on_click(on_drive_connect)
    ui_components['dir_button'].on_click(on_dir_setup)
    
    # Setup observer for module-specific events
    if observer_manager:
        observer_manager.create_logging_observer(
            event_types=["environment.drive.mount", "environment.directory.setup"],
            logger_name="env_config",
            name="EnvironmentLogObserver",
            group="env_config_observers"
        )
    
    # Run initialization
    if check_smartcash_dir():
        detect_environment()
    
    # To ensure cleanup of resources, add a cleanup function
    def cleanup():
        """Cleanup resources"""
        if observer_manager:
            observer_manager.unregister_group("env_config_observers")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components