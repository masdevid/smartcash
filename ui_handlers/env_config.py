"""
File: smartcash/ui_handlers/env_config.py
Author: Refactored
Deskripsi: Handler untuk UI konfigurasi environment SmartCash dengan reusable utilities.
"""

from IPython.display import display, HTML, clear_output

def setup_env_handlers(ui):
    """
    Setup handler untuk UI konfigurasi environment SmartCash.
    
    Args:
        ui: Dictionary berisi komponen UI
        
    Returns:
        Dictionary UI components yang sudah diupdate dengan handler
    """
    # Import necessities
    try:
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.logger import get_logger
        from smartcash.utils.ui_utils import create_status_indicator
        from smartcash.utils.config_manager import get_config_manager
        has_dependencies = True
        logger = get_logger("env_config")
        env_manager = EnvironmentManager(logger=logger)
        config_manager = get_config_manager(logger=logger)
    except ImportError as e:
        has_dependencies = False
        print(f"ℹ️ Basic fallback mode: {str(e)}")
    
    # Deteksi environment
    def detect_environment():
        """Deteksi jenis environment dan update UI sesuai hasil deteksi"""
        is_colab = False
        
        if has_dependencies:
            is_colab = env_manager.is_colab
            
            # Display environment info
            with ui['info_panel']:
                clear_output()
                try:
                    system_info = env_manager.get_system_info()
                    
                    info_html = f"""
                    <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; color: #212529;">
                        <h4 style="margin-top: 0; color: #212529;">📊 System Information</h4>
                        <ul style="color: #212529;">
                            <li><b>Python:</b> {system_info.get('python_version', 'Unknown')}</li>
                            <li><b>Base Directory:</b> {system_info.get('base_dir', 'Unknown')}</li>
                            <li><b>CUDA Available:</b> {'Yes' if system_info.get('cuda_available', False) else 'No'}</li>
                        </ul>
                    </div>
                    """
                    display(HTML(info_html))
                except Exception as e:
                    display(HTML(f"<p>⚠️ Error getting system info: {str(e)}</p>"))
        else:
            # Fallback detection
            try:
                import google.colab
                is_colab = True
            except ImportError:
                is_colab = False
                
            # Display basic info
            with ui['info_panel']:
                clear_output()
                import sys, platform
                from pathlib import Path
                display(HTML(f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; color: #212529;">
                    <h4 style="margin-top: 0; color: #212529;">📊 System Information</h4>
                    <ul style="color: #212529;">
                        <li><b>Python:</b> {platform.python_version()}</li>
                        <li><b>OS:</b> {platform.system()} {platform.release()}</li>
                        <li><b>Base Directory:</b> {Path.cwd()}</li>
                    </ul>
                </div>
                """))
        
        # Update UI based on environment
        if is_colab:
            ui['colab_panel'].value = """
                <div style="padding: 10px; background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460; margin: 10px 0;">
                    <h3 style="margin-top: 0; color: #0c5460;">☁️ Google Colab Terdeteksi</h3>
                    <p>Project akan dikonfigurasi untuk berjalan di Google Colab. Koneksi ke Google Drive direkomendasikan untuk penyimpanan data.</p>
                </div>
            """
            ui['drive_button'].layout.display = ''
        else:
            ui['colab_panel'].value = """
                <div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                    <h3 style="margin-top: 0; color: #155724;">💻 Environment Lokal Terdeteksi</h3>
                    <p>Project akan dikonfigurasi untuk berjalan di environment lokal.</p>
                </div>
            """
            ui['drive_button'].layout.display = 'none'
        
        return is_colab
    
    # Handler untuk Google Drive connection
    def on_drive_connect(b):
        with ui['status']:
            clear_output()
            
            if has_dependencies:
                display(create_status_indicator("info", "🔄 Menghubungkan ke Google Drive..."))
                success, message = env_manager.mount_drive()
                
                if success:
                    # Create symlinks
                    display(create_status_indicator("info", "🔄 Membuat symlink ke Google Drive..."))
                    symlink_stats = env_manager.create_symlinks()
                    
                    display(create_status_indicator(
                        "success", 
                        f"✅ Google Drive terhubung ke {env_manager.drive_path} ({symlink_stats['created']} symlinks baru, {symlink_stats['existing']} sudah ada)"
                    ))
                    
                    # Display tree
                    display(HTML("<h4>📂 Struktur direktori:</h4>"))
                    tree_html = env_manager.get_directory_tree(max_depth=2)
                    display(HTML(tree_html))
                    
                else:
                    display(create_status_indicator("error", f"❌ Gagal terhubung ke Google Drive: {message}"))
            else:
                # Fallback implementation
                try:
                    display(HTML('<p>🔄 Menghubungkan ke Google Drive...</p>'))
                    from google.colab import drive
                    drive.mount('/content/drive')
                    
                    # Create SmartCash directory in Drive if needed
                    from pathlib import Path
                    drive_path = Path('/content/drive/MyDrive/SmartCash')
                    if not drive_path.exists():
                        drive_path.mkdir(parents=True)
                        display(HTML(
                            f'<p>✅ Direktori <code>{drive_path}</code> berhasil dibuat di Google Drive</p>'
                        ))
                    else:
                        display(HTML(
                            f'<p>ℹ️ Direktori <code>{drive_path}</code> sudah ada di Google Drive</p>'
                        ))
                        
                    # Create symlink
                    if not Path('SmartCash_Drive').exists():
                        import os
                        os.symlink(drive_path, 'SmartCash_Drive')
                        display(HTML(
                            '<p>✅ Symlink <code>SmartCash_Drive</code> berhasil dibuat</p>'
                        ))
                    else:
                        display(HTML(
                            '<p>ℹ️ Symlink <code>SmartCash_Drive</code> sudah ada</p>'
                        ))
                        
                    display(HTML(
                        """<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #155724;">✅ Google Drive Terhubung</h3>
                            <p>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></p>
                        </div>"""
                    ))
                    
                except Exception as e:
                    display(HTML(
                        f"""<div style="padding: 10px; background: #f8d7da; border-left: 4px solid #721c24; color: #721c24; margin: 10px 0;">
                            <h3 style="margin-top: 0; color: #721c24;">❌ Gagal Terhubung ke Google Drive</h3>
                            <p>Error: {str(e)}</p>
                        </div>"""
                    ))
    
    # Handler untuk directory structure setup
    def on_dir_setup(b):
        with ui['status']:
            clear_output()
            
            if has_dependencies:
                display(create_status_indicator("info", "🔄 Membuat struktur direktori..."))
                
                # Setup directories using EnvironmentManager
                use_drive = env_manager.is_drive_mounted if hasattr(env_manager, 'is_drive_mounted') else False
                stats = env_manager.setup_directories(use_drive=use_drive)
                
                # Display stats
                display(create_status_indicator(
                    "success", 
                    f"✅ Struktur Direktori Berhasil Dibuat: {stats['created']} direktori baru, {stats['existing']} sudah ada"
                ))
                
                # Display directory tree
                display(HTML("<h4>📂 Struktur direktori yang dibuat:</h4>"))
                tree_html = env_manager.get_directory_tree(max_depth=3)
                display(HTML(tree_html))
                
                # Save environment info to config
                if hasattr(config_manager, 'update_config'):
                    env_config = {
                        'environment': {
                            'is_colab': env_manager.is_colab,
                            'drive_mounted': env_manager.is_drive_mounted,
                            'base_dir': str(env_manager.base_dir)
                        }
                    }
                    config_manager.update_config(env_config)
                
            else:
                # Fallback implementation
                import os
                from pathlib import Path
                
                # Create necessary directories
                dirs = [
                    'data/train/images', 'data/train/labels',
                    'data/valid/images', 'data/valid/labels',
                    'data/test/images', 'data/test/labels',
                    'configs', 'runs/train/weights',
                    'logs', 'exports'
                ]
                
                display(HTML('<p>🔄 Membuat struktur direktori...</p>'))
                
                created = 0
                existing = 0
                for d in dirs:
                    path = Path(d)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                        created += 1
                    else:
                        existing += 1
                    
                display(HTML(
                    f"""<div style="padding: 10px; background: #d4edda; border-left: 4px solid #155724; color: #155724; margin: 10px 0;">
                        <h3 style="margin-top: 0; color: #155724;">✅ Struktur Direktori Berhasil Dibuat</h3>
                        <p>Direktori baru: {created}, sudah ada: {existing}</p>
                        <pre style="margin: 10px 0 0 10px; color: #155724; background: transparent; border: none;">
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
    ui['drive_button'].on_click(on_drive_connect)
    ui['dir_button'].on_click(on_dir_setup)
    
    # Run initial detection
    detect_environment()
    
    # Function to create status indicator (for fallback mode)
    def create_status_indicator(status, message):
        """Buat status indicator dengan styling konsisten."""
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
    
    return ui