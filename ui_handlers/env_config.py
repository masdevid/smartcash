"""
File: smartcash/ui_handlers/env_config.py
Author: Perbaikan untuk sinkronisasi config
Deskripsi: Handler untuk UI konfigurasi environment SmartCash dengan fitur cek sync config YAML
"""

import threading, time, os, shutil, glob
from IPython.display import display, HTML, clear_output
from pathlib import Path

from smartcash.utils.ui_utils import create_status_indicator

def setup_env_config_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi environment SmartCash."""
    # Inisialisasi dependencies
    logger = env_manager = observer_manager = config_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        
        logger = get_logger("env_config")
        env_manager = EnvironmentManager(logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        config_manager = get_config_manager(logger=logger)
        observer_manager.unregister_group("env_config_observers")
    except ImportError as e:
        with ui_components['status']:
            display(HTML(f"<p style='color:red'>⚠️ Limited functionality - {str(e)}</p>"))
    
    def detect_environment():
        """Deteksi environment dan update UI"""
        is_colab = False
        if env_manager:
            is_colab = env_manager.is_colab
            with ui_components['info_panel']:
                clear_output()
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
                
            with ui_components['info_panel']:
                clear_output()
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
        ui_components['drive_button'].layout.display = '' if is_colab else 'none'
        return is_colab
    
    def fallback_get_directory_tree(root_dir, max_depth=2):
        """Fallback untuk directory tree view dengan filter khusus untuk Drive"""
        root_dir = Path(root_dir)
        if not root_dir.exists():
            return f"<span style='color:red'>❌ Directory not found: {root_dir}</span>"
        
        # Khusus untuk drive, tampilkan hanya folder SmartCash
        if '/content/drive' in str(root_dir):
            root_dir = Path('/content/drive/MyDrive/SmartCash')
            if not root_dir.exists():
                return f"<span style='color:orange'>⚠️ SmartCash folder tidak ditemukan di Google Drive</span>"
        
        result = "<pre style='margin:0;padding:5px;background:#f8f9fa;font-family:monospace;color:#333'>\n"
        result += f"<span style='color:#0366d6;font-weight:bold'>{root_dir.name}/</span>\n"
        
        def traverse_dir(path, prefix="", depth=0):
            if depth > max_depth: return ""
            # Skip jika bukan SmartCash directory di drive
            if '/content/drive' in str(path) and '/MyDrive/SmartCash' not in str(path):
                return ""
                
            items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
            tree = ""
            for i, item in enumerate(items):
                # Skip directory lain di drive yang bukan bagian SmartCash
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
        
    def sync_missing_configs(drive_configs_dir=None):
        """Cek dan salin config yang hilang"""
        if not env_manager or not env_manager.is_drive_mounted:
            return 0, 0
            
        try:
            drive_configs_dir = Path(drive_configs_dir or env_manager.drive_path / 'configs')
            drive_configs_dir.mkdir(parents=True, exist_ok=True)
            local_configs_dir = Path('configs')
            local_configs_dir.mkdir(parents=True, exist_ok=True)
            
            # Cari configs dari berbagai sumber
            source_files = []
            try:
                # Coba mencari di package
                import importlib
                try:
                    from importlib.resources import files
                    source_module = files('smartcash.configs')
                    if source_module.is_dir():
                        source_files = [p for p in source_module.iterdir() if p.name.endswith(('.yaml', '.yml'))]
                except Exception:
                    # Fallback ke path manual
                    smartcash_path = Path(__file__).parent.parent.parent
                    configs_path = smartcash_path / 'smartcash' / 'configs'
                    if configs_path.exists() and configs_path.is_dir():
                        source_files = list(configs_path.glob('*.y*ml'))
            except ImportError:
                pass
                
            # Gunakan configs lokal jika tidak ada
            if not source_files:
                source_files = list(local_configs_dir.glob('*.y*ml'))
                
            # Scan dengan pencarian luas
            if not source_files:
                for root_dir in ['', '.', '..', 'smartcash']:
                    yaml_files = list(Path(root_dir).glob('**/*.y*ml'))
                    config_files = [f for f in yaml_files if 'config' in f.name.lower()]
                    if config_files:
                        source_files = config_files
                        break
            
            # Sync files
            missing_configs = []
            copied_configs = []
            
            for config_file in source_files:
                filename = config_file.name
                drive_file = drive_configs_dir / filename
                local_file = local_configs_dir / filename
                
                # Jika tidak ada di drive, copy
                if not drive_file.exists():
                    missing_configs.append(filename)
                    try:
                        # Pastikan ada di local
                        if not local_file.exists():
                            shutil.copy2(config_file, local_file)
                        # Copy ke drive
                        shutil.copy2(local_file, drive_file)
                        copied_configs.append(filename)
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Gagal copy {filename}: {e}")
                
                # Copy dari drive ke local jika tidak ada
                elif not local_file.exists():
                    try:
                        shutil.copy2(drive_file, local_file)
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Gagal copy dari drive: {e}")
            
            return len(missing_configs), len(copied_configs)
        except Exception as e:
            if logger: logger.error(f"❌ Error sinkronisasi: {e}")
            return 0, 0
    
    def on_drive_connect(b):
        """Handler koneksi Google Drive"""
        with ui_components['status']:
            clear_output()
            
            if env_manager:
                display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
                success, message = env_manager.mount_drive()
                
                if success:
                    display(create_status_indicator("info", "🔄 Membuat symlink ke Google Drive..."))
                    symlink_stats = env_manager.create_symlinks()
                    display(create_status_indicator("success", 
                        f"✅ Drive terhubung: {env_manager.drive_path} ({symlink_stats['created']} symlinks baru)"
                    ))
                    
                    # Sync configs
                    display(create_status_indicator("info", "🔄 Memeriksa file konfigurasi..."))
                    drive_configs_dir = env_manager.drive_path / 'configs'
                    missing, copied = sync_missing_configs(drive_configs_dir)
                    
                    if missing > 0:
                        status = "success" if copied == missing else "warning"
                        message = f"✅ {copied} config disalin ke Drive" if copied == missing else f"⚠️ {copied}/{missing} config berhasil disalin"
                        display(create_status_indicator(status, message))
                    else:
                        display(create_status_indicator("info", "ℹ️ Semua config sudah ada di Drive"))
                    
                    # Display tree
                    display(HTML("<h4>📂 Struktur direktori:</h4>"))
                    tree_html = env_manager.get_directory_tree(max_depth=2) if hasattr(env_manager, 'get_directory_tree') else fallback_get_directory_tree(env_manager.base_dir, max_depth=2)
                    display(HTML(tree_html))
                    
                    # Update config
                    if config and config_manager:
                        config_manager.update_config({
                            'environment': {
                                'drive_mounted': True,
                                'drive_path': str(env_manager.drive_path)
                            }
                        })
                else:
                    display(create_status_indicator("error", f"❌ Gagal koneksi Drive: {message}"))
            else:
                # Fallback implementation
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
                        os.symlink(drive_path, 'SmartCash_Drive')
                        display(HTML('<p>✅ Symlink <code>SmartCash_Drive</code> dibuat</p>'))
                    else:
                        display(HTML('<p>ℹ️ Symlink sudah ada</p>'))
                    
                    # Sync configs
                    configs_dir = drive_path / 'configs'
                    configs_dir.mkdir(exist_ok=True)
                    local_configs = list(Path('configs').glob('*.y*ml')) if Path('configs').exists() else []
                    
                    missing_count = copied_count = 0
                    if local_configs:
                        for config_file in local_configs:
                            drive_file = configs_dir / config_file.name
                            if not drive_file.exists():
                                missing_count += 1
                                try:
                                    shutil.copy2(config_file, drive_file)
                                    copied_count += 1
                                except Exception:
                                    pass
                        
                        if missing_count > 0:
                            status = "✅" if copied_count == missing_count else "⚠️"
                            message = f"{copied_count} config disalin" if copied_count == missing_count else f"{copied_count}/{missing_count} berhasil disalin"
                            display(HTML(f'<p>{status} {message}</p>'))
                    
                    display(HTML(
                        """<div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                            <h3 style="margin-top:0">✅ Google Drive Terhubung</h3>
                            <p>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></p>
                        </div>"""
                    ))
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
            
            if env_manager:
                display(create_status_indicator("info", "🔄 Membuat struktur direktori..."))
                use_drive = getattr(env_manager, 'is_drive_mounted', False)
                stats = env_manager.setup_directories(use_drive=use_drive)
                display(create_status_indicator("success", 
                    f"✅ Direktori dibuat: {stats['created']} baru, {stats['existing']} sudah ada"
                ))
                
                # Display tree
                display(HTML("<h4>📂 Struktur direktori:</h4>"))
                tree_html = env_manager.get_directory_tree(max_depth=3) if hasattr(env_manager, 'get_directory_tree') else fallback_get_directory_tree(env_manager.base_dir, max_depth=3)
                display(HTML(tree_html))
                
                # Update config
                if config_manager:
                    config_manager.update_config({
                        'environment': {
                            'is_colab': env_manager.is_colab,
                            'drive_mounted': getattr(env_manager, 'is_drive_mounted', False),
                            'base_dir': str(env_manager.base_dir),
                            'setup_complete': True
                        }
                    })
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
    
    # Register event handlers
    ui_components['drive_button'].on_click(on_drive_connect)
    ui_components['dir_button'].on_click(on_dir_setup)
    
    # Setup observer
    if observer_manager:
        observer_manager.create_logging_observer(
            event_types=["environment.drive.mount", "environment.directory.setup"],
            logger_name="env_config",
            name="EnvironmentLogObserver",
            group="env_config_observers"
        )
    
    # Cek folder smartcash
    def check_smartcash_dir():
        """Cek apakah folder smartcash ada"""
        if not Path('smartcash').exists() or not Path('smartcash').is_dir():
            with ui_components['status']:
                clear_output()
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
    
    # Cleanup function
    def cleanup():
        """Cleanup resources"""
        if observer_manager:
            observer_manager.unregister_group("env_config_observers")
    
    ui_components['cleanup'] = cleanup
    
    # Run init
    if check_smartcash_dir():
        detect_environment()
    
    return ui_components