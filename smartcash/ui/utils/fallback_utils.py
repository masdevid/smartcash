"""
File: smartcash/ui/utils/fallback_utils.py
Deskripsi: Utilitas untuk mengurangi duplikasi code fallback pada UI components
"""

from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import importlib
import sys

def import_with_fallback(module_path: str, fallback_value: Any = None) -> Any:
    """
    Import modul atau atribut dengan fallback jika gagal.
    
    Args:
        module_path: Path ke modul atau fungsi/kelas yang akan diimport
        fallback_value: Nilai yang akan dikembalikan jika import gagal
        
    Returns:
        Modul/fungsi/kelas yang diimport atau fallback_value
    """
    try:
        if '.' in module_path:
            parts = module_path.split('.')
            module_name = '.'.join(parts[:-1])
            attr_name = parts[-1]
            
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        else:
            return importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return fallback_value

def get_logger_safely(module_name: str = None) -> Any:
    """
    Dapatkan logger dengan fallback jika tidak tersedia.
    
    Args:
        module_name: Nama modul untuk logger
        
    Returns:
        Logger object atau None jika tidak tersedia
    """
    try:
        from smartcash.common.logger import get_logger
        return get_logger(module_name)
    except ImportError:
        # Buat dummy logger atau kembalikan None
        return None

def get_status_widget(ui_components: Dict[str, Any]) -> Any:
    """
    Dapatkan widget output status dari ui_components.
    
    Args:
        ui_components: Dictionary UI components
        
    Returns:
        Widget output status atau None jika tidak ditemukan
    """
    for key in ['status', 'output', 'log_output']:
        if key in ui_components and hasattr(ui_components[key], 'clear_output'):
            return ui_components[key]
    return None

def create_status_message(message: str, title: str='Status', status_type: str = 'info', show_icon: bool = True) -> str:
    """
    Buat HTML untuk pesan status.
    
    Args:
        message: Pesan yang akan ditampilkan
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        show_icon: Apakah perlu menampilkan icon
        
    Returns:
        HTML string untuk pesan status
    """
    from smartcash.ui.utils.constants import ALERT_STYLES, ICONS
    
    # Dapatkan style dari constants
    style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    icon = ICONS.get(status_type, ICONS.get('info'))
    bg_color = style['bg_color']
    text_color = style['text_color']
    
    icon_html = f"{icon} " if show_icon else ""
    
    return f"""
    <div style="padding:8px 12px; background-color:{bg_color}; 
               color:{text_color}; border-radius:4px; margin:5px 0;
               border-left:4px solid {text_color};">
        <h3 style="color:{text_color}">{icon_html} {title}</h3>
        <p style="margin:3px 0">{message}</p>
    </div>
    """

def show_status(message: str, status_type: str = 'info', ui_components: Dict[str, Any] = None) -> None:
    """
    Tampilkan pesan status pada widget output dengan fallback ke print.
    
    Args:
        message: Pesan yang akan ditampilkan
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        ui_components: Dictionary UI components
    """
    from smartcash.ui.utils.constants import ICONS
    
    status_widget = None
    if ui_components:
        status_widget = get_status_widget(ui_components)
    
    html_message = create_status_message(message, status_type)
    
    if status_widget:
        with status_widget:
            display(HTML(html_message))
    else:
        # Fallback ke print
        icon = ICONS.get(status_type, ICONS.get('info'))
        print(f"{icon} {message}")

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update status panel dengan pesan dan status tertentu.
    
    Args:
        ui_components: Dictionary UI components
        message: Pesan yang akan ditampilkan
        status_type: Tipe status ('info', 'success', 'warning', 'error')
    """
    if not ui_components or 'status_panel' not in ui_components:
        return
        
    status_panel = ui_components['status_panel']
    html_message = create_status_message(message, status_type)
    
    status_panel.value = html_message

def load_config_safely(config_path: str, logger=None) -> Dict[str, Any]:
    """
    Load config dengan SimpleConfigManager atau fallback jika gagal.
    
    Args:
        config_path: Path ke file konfigurasi atau nama modul konfigurasi
        logger: Logger untuk mencatat error
        
    Returns:
        Dictionary berisi konfigurasi atau dict kosong jika gagal
    """
    try:
        # Gunakan SimpleConfigManager
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        
        # Jika config_path berisi nama file, ekstrak nama modul
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            module_name = config_path.split('/')[-1].split('_')[0]
            if module_name.endswith('.yaml') or module_name.endswith('.yml'):
                module_name = module_name.rsplit('.', 1)[0]
        else:
            module_name = config_path
            
        # Coba dapatkan konfigurasi
        config = config_manager.get_module_config(module_name)
        if config:
            if logger:
                logger.info(f"✅ Konfigurasi berhasil dimuat dari SimpleConfigManager untuk modul {module_name}")
            return config
        
        # Fallback ke load dari config file
        config = config_manager.load_config(config_path)
        if config:
            if logger:
                logger.info(f"✅ Konfigurasi berhasil dimuat dari file {config_path}")
            
            # Simpan ke module config juga
            try:
                config_manager.save_module_config(module_name, config)
            except Exception as e:
                if logger:
                    logger.debug(f"⚠️ Gagal menyimpan ke module config: {str(e)}")
                    
            return config
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Gagal load config: {str(e)}")
    
    # Return empty dict sebagai fallback terakhir
    return {}


def get_augmentation_manager(config=None, logger=None) -> Any:
    """
    Dapatkan augmentation manager dengan fallback jika tidak tersedia.
    
    Args:
        config: Konfigurasi untuk augmentation manager
        logger: Logger untuk mencatat error
        
    Returns:
        Augmentation manager atau None jika tidak tersedia
    """
    try:
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        # Buat konfigurasi default jika tidak ada
        if not config:
            config = {'augmentation': {'output_dir': 'data/augmented'}, 'dataset_dir': 'data'}
        
        # Dapatkan parameter dari config
        data_dir = config.get('dataset_dir', 'data')
        num_workers = config.get('augmentation', {}).get('num_workers', 4)
        
        return AugmentationService(config=config, data_dir=data_dir, logger=logger, num_workers=num_workers)
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ AugmentationManager tidak tersedia: {e}")
        return None

def handle_download_status(
    ui_components: Dict[str, Any], 
    message: str, 
    status_type: str = 'info', 
    show_output: bool = True
) -> None:
    """
    Handler status download dengan fallback aman.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan status
        status_type: Tipe status
        show_output: Tampilkan di output widget jika tersedia
    """
    # Update status panel
    update_status_panel(ui_components, message, status_type)
    
    # Tampilkan di output widget jika diperlukan
    if show_output and 'status' in ui_components:
        status_widget = ui_components['status']
        with status_widget:
            display(create_status_message(message, status_type))

def import_with_fallback(module_path: str, fallback_value: Any = None) -> Any:
    """
    Import modul atau atribut dengan fallback jika gagal.
    
    Args:
        module_path: Path ke modul atau fungsi/kelas yang akan diimport
        fallback_value: Nilai yang akan dikembalikan jika import gagal
        
    Returns:
        Modul/fungsi/kelas yang diimport atau fallback_value
    """
    try:
        if '.' in module_path:
            parts = module_path.split('.')
            module_name = '.'.join(parts[:-1])
            attr_name = parts[-1]
            
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        else:
            return importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return fallback_value

def create_fallback_ui(ui_components: Dict[str, Any], message: str, status_type: str = "warning") -> Dict[str, Any]:
    """
    Buat UI fallback saat terjadi error.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan error/warning
        status_type: Tipe status pesan
        
    Returns:
        Dictionary ui_components yang diperbarui
    """
    try:
        import ipywidgets as widgets
        from smartcash.ui.utils.constants import COLORS, ICONS
        
        # Pastikan ui_components valid
        if ui_components is None or not isinstance(ui_components, dict): 
            ui_components = {'module_name': 'fallback'}
        
        # Buat status output jika belum ada
        if 'status' not in ui_components or ui_components['status'] is None:
            ui_components['status'] = widgets.Output(layout=widgets.Layout(
                width='100%', border='1px solid #ddd', min_height='100px', padding='10px'))
        
        # Buat UI container jika belum ada
        if 'ui' not in ui_components:
            header = widgets.HTML(f"<h2>{ICONS.get('warning', '⚠️')} SmartCash UI (Fallback Mode)</h2>")
            ui_components['ui'] = widgets.VBox([header, ui_components['status']], 
                                              layout=widgets.Layout(width='100%', padding='10px'))
        
        # Tambahkan komponen aug_options dengan nilai default yang aman jika tidak ada
        # Ini untuk mencegah error 'NoneType' is not iterable
        if 'aug_options' not in ui_components:
            # Buat dummy aug_options dengan nilai default yang aman
            dummy_selector = widgets.SelectMultiple(
                options=[('Combined', 'combined')],
                value=['combined'],
                description='Jenis:',
                layout=widgets.Layout(width='70%', height='80px')
            )
            ui_components['aug_options'] = widgets.VBox([dummy_selector])
        
        # Tampilkan pesan status
        show_status(message, status_type, ui_components)
        
        # Reset logging jika mungkin
        try:
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
        except Exception as e:
            print(f"Error saat reset logging: {str(e)}")
            
        # Log error ke file jika memungkinkan
        try:
            import logging
            import traceback
            logger = logging.getLogger('ui_logger')
            logger.error(f"{ICONS.get('error', '❌')} {message}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        except Exception:
            pass
    except Exception as e:
        # Fallback terakhir jika semua gagal
        print(f"Critical error in fallback UI: {str(e)}")
        from IPython.display import HTML
        ui_components = {
            'ui': HTML(f"<div style='color:red; padding:10px; border:1px solid red;'>⚠️ Error: {message}</div>"),
            'status': None,
            'aug_options': {'children': [{'value': ['combined']}]}
        }
    
    return ui_components

def try_operation(operation: Callable, logger=None, operation_name: str = "operasi", 
                 ui_components: Optional[Dict[str, Any]] = None) -> Any:
    """
    Jalankan operasi dengan error handling terpadu.
    
    Args:
        operation: Fungsi operasi yang akan dijalankan
        logger: Logger untuk logging
        operation_name: Nama operasi untuk pesan log
        ui_components: UI components untuk menampilkan status
        
    Returns:
        Hasil operasi atau None jika gagal
    """
    try:
        result = operation()
        if logger and result: logger.info(f"✅ {operation_name.capitalize()} berhasil")
        return result
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat {operation_name}: {str(e)}")
        if ui_components: show_status(f"⚠️ Error saat {operation_name}: {str(e)}", "warning", ui_components)
        return None