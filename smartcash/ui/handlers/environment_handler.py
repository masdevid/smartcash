"""
File: smartcash/ui/handlers/environment_handler.py
Deskripsi: Handler bersama untuk manajemen environment SmartCash
"""

from IPython.display import display, HTML, clear_output
from pathlib import Path
import os
import shutil
import platform
import sys

def detect_environment(ui_components, env=None):
    """
    Deteksi environment dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (optional)
        
    Returns:
        Boolean menunjukkan apakah environment adalah Colab
    """
    is_colab = False
    
    if env and hasattr(env, 'is_colab'):
        is_colab = env.is_colab
        # Tampilkan informasi sistem
        with ui_components['info_panel']:
            clear_output(wait=True)
            try:
                system_info = env.get_system_info()
                info_html = f"""
                <div style="background:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;color:#212529">
                    <h4 style="margin-top:0">📊 System Information</h4>
                    <ul>
                        <li><b>Python:</b> {system_info.get('python_version', 'Unknown')}</li>
                        <li><b>Base Directory:</b> {system_info.get('base_directory', 'Unknown')}</li>
                        <li><b>CUDA Available:</b> {'Yes' if system_info.get('cuda', {}).get('available', False) else 'No'}</li>
                        <li><b>Platform:</b> {system_info.get('platform', 'Unknown')}</li>
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
            
        # Fallback system info
        with ui_components['info_panel']:
            clear_output(wait=True)
            display(HTML(f"""
            <div style="background:#f8f9fa;padding:10px;margin:5px 0;border-radius:5px;color:#212529">
                <h4 style="margin-top:0">📊 System Information</h4>
                <ul>
                    <li><b>Python:</b> {platform.python_version()}</li>
                    <li><b>Platform:</b> {platform.system()} {platform.release()}</li>
                    <li><b>Base Directory:</b> {Path.cwd()}</li>
                </ul>
            </div>
            """))
    
    # Update UI berdasarkan environment
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
    
    # Tampilkan tombol drive hanya di Colab
    ui_components['drive_button'].layout.display = '' if is_colab else 'none'
    return is_colab

def filter_drive_tree(tree_html):
    """
    Filter directory tree untuk fokus ke SmartCash di Google Drive.
    
    Args:
        tree_html: HTML string dari directory tree
        
    Returns:
        HTML string yang difilter
    """
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
        inside_smartcash = False
        
        for line in lines:
            if '/content/drive' in line and 'SmartCash' not in line and not inside_smartcash:
                continue
                
            if 'SmartCash/' in line or 'SmartCash_Drive' in line:
                inside_smartcash = True
                filtered_lines.append(line)
            elif inside_smartcash and ('│' not in line and '├' not in line and '└' not in line):
                inside_smartcash = False
            elif inside_smartcash:
                filtered_lines.append(line)
            elif '/content/drive' not in line:
                filtered_lines.append(line)
        
        return header + "\n".join(filtered_lines) + "</pre>"
    except Exception:
        return tree_html

def fallback_get_directory_tree(root_dir, max_depth=2):
    """
    Fallback implementation untuk directory tree jika env_manager tidak tersedia.
    
    Args:
        root_dir: Path direktori root
        max_depth: Kedalaman maksimum tree
        
    Returns:
        HTML string dari directory tree
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return f"<span style='color:red'>❌ Directory not found: {root_dir}</span>"
    
    # Khusus untuk drive, tampilkan hanya folder SmartCash
    if '/content/drive' in str(root_dir) and 'SmartCash' not in str(root_dir):
        root_dir = Path('/content/drive/MyDrive/SmartCash')
        if not root_dir.exists():
            return f"<span style='color:orange'>⚠️ SmartCash folder tidak ditemukan di Google Drive</span>"
    
    result = "<pre style='margin:0;padding:5px;background:#f8f9fa;font-family:monospace;color:#333'>\n"
    result += f"<span style='color:#0366d6;font-weight:bold'>{root_dir.name}/</span>\n"
    
    def traverse_dir(path, prefix="", depth=0):
        if depth > max_depth: return ""
        # Skip jika bukan SmartCash directory di drive
        if '/content/drive' in str(path) and 'SmartCash' not in str(path):
            return ""
            
        items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
        tree = ""
        for i, item in enumerate(items):
            # Skip directory lain di drive yang bukan bagian SmartCash
            if '/content/drive' in str(item) and 'SmartCash' not in str(item):
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

def sync_configs(source_dirs, target_dirs, logger=None):
    """
    Sinkronisasi config files dari source ke target directory.
    
    Args:
        source_dirs: List direktori sumber
        target_dirs: List direktori tujuan
        logger: Logger instance (optional)
        
    Returns:
        Tuple (total_files, copied_files)
    """
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

def check_smartcash_dir(ui_components):
    """
    Cek apakah folder smartcash ada dan tampilkan warning jika tidak ada.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean menunjukkan apakah folder smartcash ada
    """
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