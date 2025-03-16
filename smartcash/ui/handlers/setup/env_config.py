"""
File: smartcash/ui/handlers/setup/env_config.py
Author: Refactored
Deskripsi: Handler untuk UI konfigurasi environment SmartCash di subdirektori setup
"""

from IPython.display import display, HTML, clear_output
from pathlib import Path

from smartcash.ui.components.shared.alerts import create_status_indicator
from smartcash.ui.handlers.shared.error_handler import handle_error
from smartcash.ui.handlers.shared.environment_handler import (
    detect_environment, filter_drive_tree, fallback_get_directory_tree, 
    sync_configs, check_smartcash_dir
)
from smartcash.ui.handlers.shared.observer_handler import setup_observer_handlers, register_ui_observer
from smartcash.ui.handlers.shared.config_handler import setup_config_handlers, update_config
from smartcash.ui.utils.logging_utils import setup_ipython_logging

def setup_env_config_handlers(ui_components, env=None, config=None):
    """Setup handlers untuk UI konfigurasi environment SmartCash."""
    # Get logger dari ui_components atau buat baru
    logger = ui_components.get('logger')
    if not logger:
        try:
            logger = setup_ipython_logging(ui_components, logger_name="env_config")
        except ImportError:
            pass
    
    # Setup observer
    ui_components = setup_observer_handlers(ui_components, observer_group="env_config_observers")
    
    # Setup config handler
    ui_components = setup_config_handlers(ui_components, config)
    
    # Register UI observers
    register_ui_observer(
        ui_components,
        ["environment.drive.mount", "environment.directory.setup"],
        observer_group="env_config_observers"
    )
    
    def on_drive_connect(b):
        """Handler koneksi Google Drive"""
        with ui_components['status']:
            clear_output()
            
            if env and hasattr(env, 'mount_drive'):
                try:
                    display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
                    success, message = env.mount_drive()
                    
                    if success:
                        display(create_status_indicator("info", "🔄 Membuat symlink ke Google Drive..."))
                        symlink_stats = env.create_symlinks() if hasattr(env, 'create_symlinks') else {'created': 0}
                        display(create_status_indicator("success", 
                            f"✅ Drive terhubung: {env.drive_path} ({symlink_stats.get('created', 0)} symlinks baru)"
                        ))
                        
                        display(create_status_indicator("info", "🔄 Memeriksa file konfigurasi..."))
                        drive_configs_dir = env.drive_path / 'configs'
                        
                        # Sync configs antara local dan drive
                        source_dirs = [Path('configs')]
                        target_dirs = [drive_configs_dir]
                        
                        try:
                            import importlib
                            from importlib.resources import files
                            source_module = files('smartcash.configs')
                            if source_module.is_dir():
                                source_dirs.append(Path(source_module))
                        except Exception:
                            smartcash_path = Path(__file__).parent.parent.parent
                            configs_path = smartcash_path / 'configs'
                            if configs_path.exists():
                                source_dirs.append(configs_path)
                        
                        total_files, synced_files = sync_configs(source_dirs, target_dirs, logger)
                        
                        if total_files > 0:
                            status = "success" if synced_files > 0 else "info"
                            message = f"✅ {synced_files} config disalin ke Drive" if synced_files > 0 else "ℹ️ Semua config sudah ada di Drive"
                            display(create_status_indicator(status, message))
                        
                        # Display tree - fokus ke SmartCash di Drive
                        display(HTML("<h4>📂 Struktur direktori SmartCash:</h4>"))
                        drive_path = env.drive_path if hasattr(env, 'drive_path') else Path('/content/drive/MyDrive/SmartCash')
                        raw_tree_html = env.get_directory_tree(drive_path, max_depth=2) if hasattr(env, 'get_directory_tree') else fallback_get_directory_tree(drive_path, max_depth=2)
                        tree_html = filter_drive_tree(raw_tree_html)
                        display(HTML(tree_html))
                        
                        # Update config
                        update_config(ui_components, {
                            'environment': {
                                'drive_mounted': True,
                                'drive_path': str(env.drive_path)
                            }
                        })
                    else:
                        display(create_status_indicator("error", f"❌ Gagal koneksi Drive: {message}"))
                except Exception as e:
                    handle_error(e, ui_components['status'], clear=False)
            else:
                # Fallback implementation for Google Colab
                try:
                    display(HTML('<p>🔄 Menghubungkan ke Google Drive...</p>'))
                    from google.colab import drive
                    drive.mount('/content/drive')
                    
                    # Setup dirs
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    drive_path.mkdir(parents=True, exist_ok=True)
                    display(HTML(f'<p>{"✅ Direktori dibuat" if not drive_path.exists() else "ℹ️ Direktori sudah ada"}: <code>{drive_path}</code></p>'))
                    
                    # Create symlink
                    if not Path('SmartCash_Drive').exists():
                        import os
                        os.symlink(drive_path, 'SmartCash_Drive')
                        display(HTML('<p>✅ Symlink <code>SmartCash_Drive</code> dibuat</p>'))
                    else:
                        display(HTML('<p>ℹ️ Symlink sudah ada</p>'))
                    
                    # Sync configs
                    configs_dir = drive_path / 'configs'
                    configs_dir.mkdir(exist_ok=True)
                    
                    source_dirs = [Path('configs')]
                    target_dirs = [configs_dir]
                    
                    total_files, synced_files = sync_configs(source_dirs, target_dirs)
                    
                    if total_files > 0:
                        status = "✅" if synced_files > 0 else "ℹ️"
                        message = f"{synced_files} config disalin" if synced_files > 0 else "Semua config sudah tersinkronisasi"
                        display(HTML(f'<p>{status} {message}</p>'))
                    
                    display(HTML(
                        """<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                            <h3 style="margin-top:0">✅ Google Drive Terhubung</h3>
                            <p>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></p>
                        </div>"""
                    ))
                    
                    # Update config
                    update_config(ui_components, {
                        'environment': {
                            'drive_mounted': True,
                            'drive_path': str(drive_path)
                        }
                    })
                except Exception as e:
                    display(HTML(
                        f"""<div style="padding:10px;background:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0">
                            <h3 style="margin-top:0">❌ Gagal koneksi Drive</h3>
                            <p>Error: {str(e)}</p>
                        </div>"""
                    ))
    
    def on_dir_setup(b):
        """Setup directory structure"""
        with ui_components['status']:
            clear_output()
            
            if env and hasattr(env, 'setup_directories'):
                try:
                    display(create_status_indicator("info", "🔄 Membuat struktur direktori..."))
                    use_drive = getattr(env, 'is_drive_mounted', False)
                    stats = env.setup_directories(use_drive=use_drive)
                    display(create_status_indicator("success", 
                        f"✅ Direktori dibuat: {stats['created']} baru, {stats['existing']} sudah ada"
                    ))
                    
                    # Display tree - tampilkan struktur project
                    display(HTML("<h4>📂 Struktur direktori project:</h4>"))
                    tree_path = getattr(env, 'base_dir', Path.cwd())
                    
                    # Jika Drive terhubung, fokus ke direktori SmartCash
                    if use_drive and getattr(env, 'is_drive_mounted', False) and hasattr(env, 'drive_path'):
                        tree_path = env.drive_path
                    
                    raw_tree_html = env.get_directory_tree(tree_path, max_depth=3) if hasattr(env, 'get_directory_tree') else fallback_get_directory_tree(tree_path, max_depth=3)
                    tree_html = filter_drive_tree(raw_tree_html)
                    display(HTML(tree_html))
                    
                    # Update config
                    update_config(ui_components, {
                        'environment': {
                            'is_colab': getattr(env, 'is_colab', False),
                            'drive_mounted': getattr(env, 'is_drive_mounted', False),
                            'base_dir': str(getattr(env, 'base_dir', Path.cwd())),
                            'setup_complete': True
                        }
                    })
                except Exception as e:
                    handle_error(e, ui_components['status'], clear=False)
            else:
                # Fallback implementation
                dirs = [
                    'data/train/images', 'data/train/labels',
                    'data/valid/images', 'data/valid/labels',
                    'data/test/images', 'data/test/labels',
                    'configs', 'runs/train/weights', 'logs', 'exports'
                ]
                
                display(HTML('<p>🔄 Membuat struktur direktori...</p>'))
                created = existing = 0
                
                for d in dirs:
                    path = Path(d)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        created += 1
                    else:
                        existing += 1
                
                display(HTML(
                    f"""<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                        <h3 style="margin-top:0">✅ Struktur Direktori Dibuat</h3>
                        <p>Direktori baru: {created}, sudah ada: {existing}</p>
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
                
                # Update config
                update_config(ui_components, {
                    'environment': {
                        'setup_complete': True
                    }
                })
    
    # Register event handlers
    ui_components['drive_button'].on_click(on_drive_connect)
    ui_components['dir_button'].on_click(on_dir_setup)
    
    # Run init
    if check_smartcash_dir(ui_components):
        detect_environment(ui_components, env)
    
    return ui_components