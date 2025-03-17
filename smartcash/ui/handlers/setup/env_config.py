"""
File: smartcash/ui/handlers/setup/env_config.py
Deskripsi: Handler untuk UI konfigurasi environment SmartCash di subdirektori setup
"""

import os
import shutil
from pathlib import Path
from IPython.display import display, HTML, clear_output

def setup_env_config_handlers(ui_components, env=None, config=None):
    """Setup handlers untuk UI konfigurasi environment SmartCash."""
    
    # Print debugging info to console
    print("Setting up environment config handlers")
    print(f"Current directory: {Path.cwd()}")
    
    def detect_environment():
        """Detect and display environment info"""
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
        
        with ui_components['info_panel']:
            clear_output(wait=True)
            import sys, platform
            display(HTML(f"""
            <div style="background:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;color:#212529">
                <h4 style="margin-top:0">📊 System Information</h4>
                <ul>
                    <li><b>Python:</b> {platform.python_version()}</li>
                    <li><b>Platform:</b> {platform.system()} {platform.release()}</li>
                    <li><b>Directory:</b> {Path.cwd()}</li>
                    <li><b>Environment:</b> {'Google Colab' if is_colab else 'Local'}</li>
                </ul>
            </div>
            """))
        
        # Update UI based on environment
        ui_components['colab_panel'].value = """
            <div style="padding:10px;background:#d1ecf1;border-left:4px solid #0c5460;color:#0c5460;margin:10px 0">
                <h3 style="margin-top:0; color: inherit">☁️ Google Colab Environment Detected</h3>
                <p>Project will be configured for Google Colab. Connecting to Google Drive is recommended.</p>
            </div>
        """ if is_colab else """
            <div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                <h3 style="margin-top:0; color: inherit">💻 Local Environment Detected</h3>
                <p>Project will be configured for local development.</p>
            </div>
        """
        
        return is_colab
    
    def on_drive_connect(b):
        """Handler for Google Drive connection"""
        with ui_components['status']:
            clear_output(wait=True)
            display(HTML("<p>🔄 Connecting to Google Drive...</p>"))
            
            try:
                # Use direct Colab approach
                from google.colab import drive
                drive.mount('/content/drive')
                
                # Setup directories
                drive_path = Path('/content/drive/MyDrive/SmartCash')
                drive_path.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<p>✅ Created SmartCash directory: <code>{drive_path}</code></p>"))
                
                # Create symlink
                if not Path('SmartCash_Drive').exists():
                    os.symlink(drive_path, 'SmartCash_Drive')
                    display(HTML("<p>✅ Created symlink 'SmartCash_Drive'</p>"))
                else:
                    display(HTML("<p>ℹ️ Symlink already exists</p>"))
                
                # Create configs directory
                configs_dir = drive_path / 'configs'
                configs_dir.mkdir(parents=True, exist_ok=True)
                Path('configs').mkdir(exist_ok=True)
                
                # Sync configs 
                display(HTML("<p>🔄 Syncing configuration files...</p>"))
                
                # Check local configs
                local_configs = list(Path('configs').glob('*.y*ml'))
                
                copied_count = 0
                for config_file in local_configs:
                    drive_file = configs_dir / config_file.name
                    if not drive_file.exists():
                        try:
                            shutil.copy2(config_file, drive_file)
                            copied_count += 1
                            display(HTML(f"<p>✅ Copied <code>{config_file.name}</code> to Drive</p>"))
                        except Exception as e:
                            display(HTML(f"<p>⚠️ Failed to copy <code>{config_file.name}</code>: {str(e)}</p>"))
                
                # Check if any configs were copied
                if not local_configs:
                    display(HTML("<p>⚠️ No config files found to sync</p>"))
                elif copied_count == 0:
                    display(HTML("<p>ℹ️ All configs already synced to Drive</p>"))
                
                # Show success message
                display(HTML(
                    """<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                        <h3 style="margin-top:0">✅ Google Drive Connected</h3>
                        <p>Data will be stored in <code>/content/drive/MyDrive/SmartCash</code></p>
                    </div>"""
                ))
                
                # Display directory tree
                display(HTML("<h4>📂 SmartCash Directory Structure:</h4>"))
                display(HTML(get_dir_tree(drive_path, max_depth=2)))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to connect to Google Drive</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
    
    def on_dir_setup(b):
        """Handler for directory structure setup"""
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
                
                # Create local directories
                for d in dirs:
                    path = Path(d)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        created += 1
                        display(HTML(f"<p>✅ Created directory: <code>{d}</code></p>"))
                    else:
                        existing += 1
                
                # Create directories in Drive if mounted
                drive_mounted = Path('/content/drive/MyDrive').exists()
                if drive_mounted:
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    drive_path.mkdir(parents=True, exist_ok=True)
                    
                    display(HTML("<p>🔄 Creating directories in Drive...</p>"))
                    for d in dirs:
                        drive_dir = drive_path / d
                        if not drive_dir.exists():
                            drive_dir.mkdir(parents=True, exist_ok=True)
                            display(HTML(f"<p>✅ Created Drive directory: <code>{d}</code></p>"))
                
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
                
                # Display local directory tree
                display(HTML("<h4>📂 Project Directory Structure:</h4>"))
                display(HTML(get_dir_tree(Path.cwd(), max_depth=2)))
                
                if drive_mounted:
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    display(HTML("<h4>📂 Drive Directory Structure:</h4>"))
                    display(HTML(get_dir_tree(drive_path, max_depth=2)))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to set up directory structure</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
    
    def get_dir_tree(root_dir, max_depth=2):
        """Generate HTML directory tree"""
        root_dir = Path(root_dir)
        if not root_dir.exists():
            return f"<span style='color:red'>❌ Directory not found: {root_dir}</span>"
        
        result = "<pre style='margin:0;padding:5px;background:#f8f9fa;font-family:monospace;color:#333'>\n"
        result += f"<span style='color:#0366d6;font-weight:bold'>{root_dir.name}/</span>\n"
        
        def traverse_dir(path, prefix="", depth=0):
            if depth > max_depth: 
                return ""
                
            try:
                items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
            except (PermissionError, OSError):
                return f"{prefix}└─ <span style='color:red'>❌ Access error</span>\n"
                
            tree = ""
            for i, item in enumerate(items):
                # Skip non-SmartCash directories in Drive
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
    
    # Run initializations
    detect_environment()
    
    # Register button handlers directly
    ui_components['drive_button'].on_click(on_drive_connect)
    ui_components['dir_button'].on_click(on_dir_setup)
    
    return ui_components