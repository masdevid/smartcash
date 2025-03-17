"""
File: smartcash/ui/handlers/setup/env_config.py
Deskripsi: Handler untuk UI konfigurasi environment SmartCash di subdirektori setup
"""

import os
import shutil
import concurrent.futures
from pathlib import Path
from IPython.display import display, HTML, clear_output

from smartcash.ui.components.shared.alerts import create_status_indicator

def setup_env_config_handlers(ui_components, env=None, config=None):
    """Setup handlers untuk UI konfigurasi environment SmartCash."""
    
    # Create thread pool for background operations
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    # Initialize dependencies
    logger = env_manager = observer_manager = config_manager = None
    
    # Try loading SmartCash modules
    try:
        from smartcash.common.logger import get_logger
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        from smartcash.components.observer.manager_observer import ObserverManager
        
        logger = get_logger("env_config")
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        observer_manager = ObserverManager.get_instance()
        
        # Clean up previous observer group
        if observer_manager:
            observer_manager.unregister_group("env_config_observers")
    except ImportError as e:
        with ui_components['status']:
            display(HTML(f"<p style='color:#856404'>⚠️ Running in limited mode: {str(e)}</p>"))
    
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
                if '/content/drive' in line and 'SmartCash' not in line and not inside_drive:
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
        """Simple directory tree visualization"""
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
    
    def sync_configs(source_dirs, target_dirs):
        """Sync config files from source to target directories"""
        total_files = copied_files = 0
        
        try:
            for source_dir in source_dirs:
                source_dir = Path(source_dir)
                if not source_dir.exists() or not source_dir.is_dir():
                    continue
                
                config_files = list(source_dir.glob('*.y*ml'))
                
                for config_file in config_files:
                    total_files += 1
                    
                    for target_dir in target_dirs:
                        target_dir = Path(target_dir)
                        
                        try:
                            target_dir.mkdir(parents=True, exist_ok=True)
                            target_file = target_dir / config_file.name
                            
                            if not target_file.exists():
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
    
    def on_drive_connect(b):
        """Handler for Google Drive connection"""
        future = None
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", "🔄 Connecting to Google Drive..."))
            
            try:
                # Always use direct Colab approach for reliability
                from google.colab import drive
                drive.mount('/content/drive')
                
                # Setup directories
                drive_path = Path('/content/drive/MyDrive/SmartCash')
                drive_path.mkdir(parents=True, exist_ok=True)
                display(create_status_indicator("success", f"✅ SmartCash directory in Drive: {drive_path}"))
                
                # Create symlink
                if not Path('SmartCash_Drive').exists():
                    os.symlink(drive_path, 'SmartCash_Drive')
                    display(create_status_indicator("success", "✅ Created symlink 'SmartCash_Drive'"))
                
                # Create configs directory
                configs_dir = drive_path / 'configs'
                configs_dir.mkdir(parents=True, exist_ok=True)
                Path('configs').mkdir(exist_ok=True)
                
                # Sync configs asynchronously
                display(create_status_indicator("info", "🔄 Syncing configuration files..."))
                future = thread_pool.submit(sync_configs, [Path('configs')], [configs_dir])
                
                # Show confirmation
                display(HTML(
                    """<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                        <h3 style="margin-top:0">✅ Google Drive Connected</h3>
                        <p>Data will be stored in <code>/content/drive/MyDrive/SmartCash</code></p>
                    </div>"""
                ))
                
                # Display directory tree
                display(HTML("<h4>📂 SmartCash Directory Structure:</h4>"))
                tree_html = fallback_get_directory_tree(drive_path, max_depth=2)
                display(HTML(tree_html))
                
                # Update config if available
                if config_manager:
                    try:
                        config_manager.update_config({
                            'environment': {
                                'drive_mounted': True,
                                'drive_path': str(drive_path)
                            }
                        })
                    except Exception as e:
                        display(create_status_indicator("warning", f"⚠️ Could not update config: {str(e)}"))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to connect to Google Drive</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
            finally:
                # Wait for future if active
                if future and not future.done():
                    try:
                        total, copied = future.result(timeout=5)
                        if total > 0:
                            status = "success" if copied > 0 else "info"
                            message = f"✅ {copied} configs copied to Drive" if copied > 0 else "ℹ️ All configs already synced"
                            display(create_status_indicator(status, message))
                    except concurrent.futures.TimeoutError:
                        display(create_status_indicator("info", "🔄 Config syncing continues in background"))
    
    def on_dir_setup(b):
        """Handler for directory structure setup"""
        future = None
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", "🔄 Setting up directory structure..."))
            
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
                
                # Create directories in Drive if mounted
                drive_mounted = Path('/content/drive/MyDrive').exists()
                if drive_mounted:
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    drive_path.mkdir(parents=True, exist_ok=True)
                    
                    # Start async directory creation
                    display(create_status_indicator("info", "🔄 Creating directories in Drive..."))
                    future = thread_pool.submit(lambda: [
                        Path(drive_path / d).mkdir(parents=True, exist_ok=True) 
                        for d in dirs
                    ])
                
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
                tree_html = fallback_get_directory_tree(Path.cwd(), max_depth=2)
                display(HTML(tree_html))
                
                # Update config if available
                if config_manager:
                    try:
                        config_manager.update_config({
                            'environment': {
                                'setup_complete': True
                            }
                        })
                    except Exception as e:
                        display(create_status_indicator("warning", f"⚠️ Could not update config: {str(e)}"))
                
            except Exception as e:
                display(HTML(
                    f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                        <h3 style="margin-top:0">❌ Failed to set up directory structure</h3>
                        <p>Error: {str(e)}</p>
                    </div>"""
                ))
            finally:
                # Wait for and process future if active
                if future and not future.done():
                    try:
                        future.result(timeout=5)
                        if drive_mounted:
                            drive_path = Path('/content/drive/MyDrive/SmartCash')
                            display(HTML("<h4>📂 Drive Directory Structure:</h4>"))
                            tree_html = fallback_get_directory_tree(drive_path, max_depth=2)
                            display(HTML(tree_html))
                    except concurrent.futures.TimeoutError:
                        display(create_status_indicator("info", "🔄 Drive directory creation continues in background"))
    
    # Register event handlers
    ui_components['drive_button'].on_click(on_drive_connect)
    ui_components['dir_button'].on_click(on_dir_setup)
    
    # Setup observer if available
    if observer_manager:
        observer_manager.create_logging_observer(
            event_types=["environment.drive.mount", "environment.directory.setup"],
            logger_name="env_config",
            name="EnvironmentLogObserver",
            group="env_config_observers"
        )
    
    # Check for smartcash directory
    if not Path('smartcash').exists() or not Path('smartcash').is_dir():
        with ui_components['status']:
            display(HTML(
                """<div style="padding:15px;background-color:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0;border-radius:4px">
                    <h3 style="margin-top:0">❌ SmartCash folder not found!</h3>
                    <p>Repository hasn't been properly cloned. Please run the repository clone cell first.</p>
                    <ol>
                        <li>Run the repository clone cell (Cell 1.1)</li>
                        <li>Restart runtime (Runtime > Restart runtime)</li>
                        <li>Run the notebook from the beginning</li>
                    </ol>
                </div>"""
            ))
    else:
        # Detect environment
        is_colab = False
        try:
            import google.colab
            is_colab = True
        except ImportError:
            pass
            
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
    
    # Create cleanup function to shut down thread pool
    def cleanup():
        """Clean up resources"""
        try:
            thread_pool.shutdown(wait=False)
            if observer_manager:
                observer_manager.unregister_group("env_config_observers")
        except Exception:
            pass
            
    ui_components['cleanup'] = cleanup
    
    return ui_components