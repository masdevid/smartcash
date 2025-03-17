"""
File: smartcash/ui/handlers/environment_handler.py
Deskripsi: Handler bersama untuk manajemen environment SmartCash
"""

from IPython.display import display, HTML, clear_output
from pathlib import Path
import os
import shutil
import re

def detect_environment(ui_components, env=None):
    """
    Deteksi environment dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (optional)
        
    Returns:
        Boolean menunjukkan apakah environment adalah Colab
    """
    # Fallback deteksi Colab
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        pass
    
    # Update panel Colab
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
        # Cari awal dan akhir dari <pre> tag
        pre_start = tree_html.find("<pre")
        pre_end = tree_html.find("</pre>")
        
        if pre_start == -1 or pre_end == -1:
            return tree_html
            
        # Pisahkan header, content, dan footer
        header = tree_html[:pre_start + tree_html[pre_start:].find(">") + 1]
        content = tree_html[pre_start + tree_html[pre_start:].find(">") + 1:pre_end]
        footer = tree_html[pre_end:]
        
        # Filter baris-baris yang berhubungan dengan SmartCash
        lines = content.split("\n")
        filtered_lines = []
        inside_smartcash = False
        
        for line in lines:
            # Deteksi direktori SmartCash
            if 'SmartCash/' in line or 'SmartCash_Drive' in line:
                inside_smartcash = True
                filtered_lines.append(line)
            # Deteksi folder drive root
            elif '/content/drive' in line and 'SmartCash' not in line and not inside_smartcash:
                # Skip direktori drive yang bukan SmartCash
                continue
            # Deteksi keluar dari subtree SmartCash
            elif inside_smartcash and ('│' not in line and '├' not in line and '└' not in line):
                inside_smartcash = False
            # Simpan baris jika masih dalam SmartCash atau bukan bagian dari Drive
            elif inside_smartcash or '/content/drive' not in line:
                filtered_lines.append(line)
        
        return header + "\n".join(filtered_lines) + footer
    except Exception:
        # Jika terjadi error, kembalikan tree asli
        return tree_html

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
        # Coba gunakan ConfigManager jika tersedia
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            # Gunakan fungsi sinkronisasi bawaan jika tersedia
            if hasattr(config_manager, 'sync_all_configs'):
                results = config_manager.sync_all_configs('merge')
                
                if logger:
                    logger.info(f"✅ Sinkronisasi config berhasil menggunakan ConfigManager")
                    
                # Ekstrak statistik dari hasil
                synced = len(results.get('synced', []))
                failed = len(results.get('failed', []))
                return synced + failed, synced
        except ImportError:
            # Lanjutkan dengan implementasi manual
            pass
            
        # Implementasi manual
        for source_dir in source_dirs:
            if not isinstance(source_dir, Path):
                source_dir = Path(source_dir)
            
            if not source_dir.exists() or not source_dir.is_dir():
                continue
            
            # Dapatkan semua file YAML/YAML di direktori sumber
            config_files = list(source_dir.glob('*.y*ml'))
            total_files += len(config_files)
            
            for config_file in config_files:
                for target_dir in target_dirs:
                    if not isinstance(target_dir, Path):
                        target_dir = Path(target_dir)
                    
                    # Buat direktori target jika belum ada
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / config_file.name
                    
                    # Salin file jika belum ada di target
                    if not target_file.exists():
                        try:
                            shutil.copy2(config_file, target_file)
                            copied_files += 1
                            if logger:
                                logger.info(f"✅ Copied {config_file.name} ke {target_dir}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"⚠️ Gagal menyalin {config_file.name}: {str(e)}")
                    elif config_file.stat().st_mtime > target_file.stat().st_mtime:
                        # File sumber lebih baru dari target
                        try:
                            shutil.copy2(config_file, target_file)
                            copied_files += 1
                            if logger:
                                logger.info(f"✅ Updated {config_file.name} yang lebih baru ke {target_dir}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"⚠️ Gagal update {config_file.name}: {str(e)}")
        
        if logger and total_files > 0:
            logger.info(f"🔄 Sinkronisasi config: {copied_files} dari {total_files} file disalin")
            
        return total_files, copied_files
    except Exception as e:
        if logger:
            logger.error(f"❌ Error sinkronisasi config: {str(e)}")
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