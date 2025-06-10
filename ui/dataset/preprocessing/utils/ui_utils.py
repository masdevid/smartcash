"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: UI utilities untuk preprocessing operations
"""

from typing import Dict, Any, Optional, Union, List, Callable
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from datetime import datetime
import json

from smartcash.common.logger import get_logger

# Constants
DEFAULT_UI_CONFIG = {
    'preprocessing': {
        'enabled': True,
        'target_split': 'all',
        'validate': {'enabled': True},
        'analysis': {'enabled': False},
        'normalization': {
            'method': 'minmax',
            'target_size': [640, 640]
        },
        'augmentation': {
            'enabled': False,
            'variations': 2,
            'balance_classes': False
        }
    },
    'performance': {
        'num_workers': 8,
        'batch_size': 32,
        'use_gpu': True
    }
}

def get_ui_logger(ui_components: Dict[str, Any]) -> Optional[Callable]:
    """Get logger from UI components with fallback to default logger"""
    logger = ui_components.get('logger')
    if not logger:
        logger = get_logger('preprocessing_ui')
        ui_components['logger'] = logger
    return logger

def log_preprocessing_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log preprocessing configuration dalam format yang rapi"""
    try:
        preprocessing = config.get('preprocessing', {})
        performance = config.get('performance', {})
        
        normalization = preprocessing.get('normalization', {})
        target_size = normalization.get('target_size', [640, 640])
        
        config_lines = [
            "🔧 <b>Konfigurasi Preprocessing</b>",
            f"📐 <b>Resolusi:</b> {target_size[0]}x{target_size[1]}",
            f"🎨 <b>Normalisasi:</b> {normalization.get('method', 'minmax')}",
            f"⚙️ <b>Workers:</b> {performance.get('num_workers', 8)}",
            f"🎯 <b>Target Split:</b> {preprocessing.get('target_split', 'all').capitalize()}",
            f"✅ <b>Validasi:</b> {'Aktif' if preprocessing.get('validate', {}).get('enabled', True) else 'Nonaktif'}",
            f"📊 <b>Analisis:</b> {'Aktif' if preprocessing.get('analysis', {}).get('enabled', False) else 'Nonaktif'}"
        ]
        
        log_to_accordion(ui_components, '<br>'.join(config_lines), 'info')
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Gagal menampilkan konfigurasi: {str(e)}")

def display_preprocessing_results(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Display preprocessing results dalam format yang rapi"""
    try:
        stats = result.get('stats', {})
        
        # Main summary
        summary_lines = [
            "<div style='margin-bottom: 10px;'>",
            "<h3>📊 <b>Hasil Preprocessing</b></h3>",
            f"<p>🖼️ <b>Total Diproses:</b> {stats.get('total_images', 0):,} gambar</p>",
            f"<p>⏱️ <b>Durasi:</b> {result.get('processing_time', 0):.1f} detik</p>",
            f"<p>📂 <b>Output:</b> <code>{result.get('output_dir', 'N/A')}</code></p>"
        ]
        
        # Splits detail
        splits = stats.get('splits', {})
        if splits:
            summary_lines.append("<p><b>📊 Detail per Split:</b></p><ul>")
            for split_name, split_stats in splits.items():
                img_count = split_stats.get('processed', 0)
                summary_lines.append(f"<li>{split_name.capitalize()}: {img_count:,} gambar</li>")
            summary_lines.append("</ul>")
        
        # UUID info jika ada
        if stats.get('uuid_processing'):
            summary_lines.append(
                f"<p>🔤 <b>UUID Processing:</b> {stats.get('uuid_files', 0)} files</p>"
            )
        
        summary_lines.append("</div>")
        
        # Display in output area
        output = ui_components.get('output_area')
        if output:
            with output:
                clear_output(wait=True)
                display(HTML(''.join(summary_lines)))
        
        # Also log to accordion
        log_to_accordion(ui_components, ''.join(summary_lines).replace('<br>', '\n'), 'success')
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Gagal menampilkan hasil: {str(e)}")
        handle_ui_error(ui_components, f"Gagal menampilkan hasil: {str(e)}")

    log_to_accordion(ui_components, '\n'.join(summary_lines), 'success')

def show_preprocessing_success(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Show preprocessing success dengan detailed stats"""
    try:
        stats = result.get('stats', {})
        
        success_lines = [
            "<div style='margin: 10px 0; padding: 10px; background-color: #e8f5e9; border-left: 4px solid #4caf50;'>",
            f"<h4 style='margin: 0 0 10px 0; color: #2e7d32;'>✅ <b>Preprocessing Berhasil!</b></h4>",
            f"<p style='margin: 5px 0;'>🖼️ <b>Total Gambar:</b> {stats.get('total_images', 0):,}</p>",
            f"<p style='margin: 5px 0;'>📂 <b>Output:</b> <code>{result.get('output_dir', 'N/A')}</code></p>",
            f"<p style='margin: 5px 0;'>⏱️ <b>Durasi:</b> {result.get('processing_time', 0):.1f} detik</p>"
        ]
        
        # Add normalization info if applied
        if stats.get('normalization_applied'):
            success_lines.append(
                f"<p style='margin: 5px 0;'>🎨 <b>Normalisasi:</b> {stats.get('normalization_method', 'applied').capitalize()}</p>"
            )
        
        success_lines.append("</div>")
        success_message = ''.join(success_lines)
        
        # Display in output area
        output = ui_components.get('output_area')
        if output:
            with output:
                clear_output(wait=True)
                display(HTML(success_message))
        
        # Also log to accordion
        log_to_accordion(ui_components, success_message.replace('<br>', '\n'), 'success')
        
        return True
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Gagal menampilkan pesan sukses: {str(e)}")
        return False

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """
    Log message ke UI accordion dengan styling dan level yang sesuai
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        level: Level log (info, success, warning, error)
    """
    try:
        # Dapatkan logger dari UI components atau gunakan default
        logger = ui_components.get('logger')
        if not logger:
            logger = get_ui_logger(ui_components)
        
        # Map level ke method logger yang sesuai
        log_methods = {
            'info': logger.info,
            'success': getattr(logger, 'success', logger.info),  # Fallback ke info jika tidak ada success
            'warning': logger.warning,
            'error': logger.error
        }
        
        # Hilangkan tag HTML untuk logging ke console
        console_message = message
        if '<' in message and '>' in message:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(message, 'html.parser')
            console_message = soup.get_text()
        
        # Log ke console
        log_method = log_methods.get(level.lower(), logger.info)
        log_method(console_message)
        
        # Update UI accordion jika ada
        log_accordion = ui_components.get('log_accordion')
        if log_accordion and hasattr(log_accordion, 'children') and len(log_accordion.children) > 0:
            log_output = log_accordion.children[0]
            if hasattr(log_output, 'append_stdout'):
                # Tambahkan timestamp dan styling berdasarkan level
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Tentukan warna berdasarkan level
                colors = {
                    'info': '#2196F3',      # Blue
                    'success': '#4CAF50',  # Green
                    'warning': '#FF9800',  # Orange
                    'error': '#F44336'     # Red
                }
                color = colors.get(level.lower(), '#000000')
                
                # Format pesan dengan HTML
                formatted_msg = f"""
                <div style='margin: 5px 0; padding: 5px; border-left: 3px solid {color};'>
                    <span style='color: #757575; font-size: 0.9em;'>{timestamp}</span>
                    <div style='margin-top: 3px;'>{message}</div>
                </div>
                """
                
                with log_output:
                    display(HTML(formatted_msg))
                
                # Auto-scroll ke bawah
                log_output.scroll_to_bottom()
        
        # Auto-expand accordion untuk error/warning
        if level.lower() in ['error', 'warning'] and 'log_accordion' in ui_components:
            if hasattr(ui_components['log_accordion'], 'selected_index'):
                ui_components['log_accordion'].selected_index = 0
                
    except Exception as e:
        # Fallback ke logging biasa jika terjadi error
        logger = get_ui_logger(ui_components)
        logger.error(f"Gagal menampilkan log: {str(e)}")
        if logger:
            logger.info(f"[RAW] {message}")

def clear_outputs(ui_components: Dict[str, Any], clear_logs: bool = True, clear_confirm: bool = True):
    """
    Clear UI output areas dengan opsi yang fleksibel
    
    Args:
        ui_components: Dictionary berisi komponen UI
        clear_logs: Jika True, bersihkan area log
        clear_confirm: Jika True, bersihkan area konfirmasi
    """
    try:
        # Clear log output jika diminta dan komponen tersedia
        if clear_logs and 'log_output' in ui_components:
            log_output = ui_components['log_output']
            if hasattr(log_output, 'clear_output'):
                with log_output:
                    log_output.clear_output(wait=True)
        
        # Clear confirmation area jika diminta dan komponen tersedia
        if clear_confirm and 'confirmation_area' in ui_components:
            confirm_area = ui_components['confirmation_area']
            if hasattr(confirm_area, 'clear_output'):
                with confirm_area:
                    confirm_area.clear_output(wait=True)
        
        # Reset progress tracker jika ada
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
            
        # Clear main output area jika ada
        output_area = ui_components.get('output_area')
        if output_area and hasattr(output_area, 'clear_output'):
            with output_area:
                output_area.clear_output(wait=True)
                
        return True
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Gagal membersihkan output: {str(e)}")
        return False

def handle_ui_error(ui_components: Dict[str, Any], error_msg: str, button_manager=None, exception: Optional[Exception] = None):
    """
    Handle error dengan UI updates dan proper state reset
    
    Args:
        ui_components: Dictionary berisi komponen UI
        error_msg: Pesan error yang akan ditampilkan
        button_manager: Instance ButtonManager untuk mengatur state tombol
        exception: Exception yang memicu error (opsional)
    """
    try:
        logger = get_ui_logger(ui_components)
        
        # Format pesan error
        full_error_msg = f"❌ {error_msg}"
        if exception:
            full_error_msg += f"\n\nDetail: {str(exception)}\n\n{type(exception).__name__}"
        
        # Log error
        logger.error(full_error_msg)
        
        # Update progress tracker jika ada
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            elif hasattr(progress_tracker, 'update'):
                progress_tracker.update(0, f"Error: {error_msg}")
        
        # Log ke accordion dengan styling
        error_html = f"""
        <div style='margin: 10px 0; padding: 10px; 
                   background-color: #ffebee; 
                   border-left: 4px solid #f44336;'>
            <h4 style='margin: 0 0 10px 0; color: #c62828;'>⚠️ <b>Terjadi Kesalahan</b></h4>
            <p style='margin: 5px 0;'>{error_msg}</p>
        """
        
        if exception:
            error_html += f"""
            <div style='margin-top: 10px; padding: 8px; 
                       background-color: rgba(0,0,0,0.05); 
                       border-radius: 4px; 
                       font-family: monospace; 
                       font-size: 0.9em;'>
                <b>{type(exception).__name__}:</b> {str(exception)}
            </div>
            """
        
        error_html += "</div>"
        
        log_to_accordion(ui_components, error_html, 'error')
        
        # Tampilkan notifikasi error
        try:
            from smartcash.ui.utils.fallback_utils import show_status_safe
            show_status_safe(error_msg, 'error', ui_components)
        except Exception as e:
            logger.error(f"Gagal menampilkan status error: {str(e)}")
        
        # Reset state tombol jika button_manager tersedia
        if button_manager is not None:
            try:
                if hasattr(button_manager, 'enable_buttons'):
                    button_manager.enable_buttons()
                elif hasattr(button_manager, 'reset_state'):
                    button_manager.reset_state()
            except Exception as e:
                logger.error(f"Gagal mereset state tombol: {str(e)}")
        
        return False
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Error dalam handle_ui_error: {str(e)}")
        logger.error(f"Pesan error asli: {error_msg}")
        if exception:
            logger.error(f"Exception asli: {str(exception)}")
        return False

def show_ui_success(ui_components: Dict[str, Any], message: str, button_manager=None, duration: int = 5):
    """
    Tampilkan pesan sukses dengan UI updates
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan sukses yang akan ditampilkan
        button_manager: Instance ButtonManager untuk mengatur state tombol (opsional)
        duration: Durasi tampilan notifikasi dalam detik (default: 5)
    """
    try:
        logger = get_ui_logger(ui_components)
        
        # Format pesan sukses
        success_msg = f"✅ {message}"
        
        # Log ke console
        logger.info(success_msg)
        
        # Tampilkan di log accordion dengan styling
        success_html = f"""
        <div style='margin: 10px 0; padding: 10px; 
                   background-color: #e8f5e9; 
                   border-left: 4px solid #4caf50;'>
            <h4 style='margin: 0 0 10px 0; color: #2e7d32;'>✅ <b>Sukses</b></h4>
            <p style='margin: 5px 0;'>{message}</p>
        </div>
        """
        
        log_to_accordion(ui_components, success_html, 'success')
        
        # Tampilkan notifikasi sukses
        try:
            from IPython.display import display, HTML
            from ipywidgets import Output
            
            # Tampilkan di output area jika ada
            output_area = ui_components.get('output_area')
            if output_area and hasattr(output_area, 'clear_output'):
                with output_area:
                    output_area.clear_output(wait=True)
                    display(HTML(success_html))
            
            # Tampilkan notifikasi toast
            try:
                import ipyvuetify as v
                from IPython.display import display as ipy_display
                
                toast = v.Snackbar(
                    v_model=True,
                    color="success",
                    timeout=duration * 1000,  # Convert to milliseconds
                    children=[f"✅ {message}"]
                )
                ipy_display(toast)
            except ImportError:
                # Fallback jika ipyvuetify tidak tersedia
                pass
                
        except Exception as e:
            logger.error(f"Gagal menampilkan notifikasi sukses: {str(e)}")
        
        # Reset state tombol jika button_manager tersedia
        if button_manager is not None:
            try:
                if hasattr(button_manager, 'enable_buttons'):
                    button_manager.enable_buttons()
                elif hasattr(button_manager, 'reset_state'):
                    button_manager.reset_state()
            except Exception as e:
                logger.error(f"Gagal mereset state tombol: {str(e)}")
        
        return True
        
    except Exception as e:
        logger = get_ui_logger(ui_components)
        logger.error(f"Error dalam show_ui_success: {str(e)}")
        logger.error(f"Pesan sukses asli: {message}")
        return False

def is_milestone_step(step: str, progress: int) -> bool:
    """Only log major milestones to prevent browser crash"""
    return progress in [0, 25, 50, 75, 100] or 'error' in step.lower()