"""
File: smartcash/ui/config_cell/components/ui_factory.py
Deskripsi: UI Component Factory untuk Config Cell dengan shared components

Modul ini menyediakan factory functions untuk membuat komponen UI standar
yang digunakan di seluruh interface config cell menggunakan shared components.
"""
from typing import Dict, Any, Optional, List
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.config_cell.constants import StatusType
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

# Import shared components
from smartcash.ui.components import (
    create_header,
    create_status_panel,
    create_info_accordion,
    create_log_accordion,
    create_responsive_container,
    create_section_title,
    create_divider,
    create_error_card
)

__all__ = [
    'create_config_summary_panel',
    'create_log_components',
    'create_info_components',
    'create_config_cell_ui',
    'create_container'
]

logger = get_logger(__name__)

def create_container(title: str = None, container_id: str = None) -> Dict[str, Any]:
    """🏗️ Membuat styled container menggunakan shared components.
    
    Args:
        title: Judul opsional yang ditampilkan di bagian atas container
        container_id: Identifier unik opsional untuk container
        
    Returns:
        Dictionary berisi:
        - 'container': Widget VBox container
        - 'content_area': VBox dimana child components harus ditambahkan
    """
    # Gunakan create_responsive_container dari shared components
    container = create_responsive_container()
    
    # Tambahkan styling khusus untuk config cell
    container.layout.border = '1px solid #e0e0e0'
    container.layout.border_radius = '8px'
    container.layout.padding = '15px'
    container.layout.margin = '10px 0'
    container.layout.box_shadow = '0 2px 4px rgba(0,0,0,0.1)'
    
    # Tambahkan ID untuk debugging dan testing
    if container_id:
        container.add_class(f'container-{container_id}')
    
    content_area = widgets.VBox(
        layout=widgets.Layout(
            margin='10px 0 0 0',
            width='100%'
        )
    )
    
    if title:
        # Gunakan create_section_title dari shared components
        header = create_section_title(title)
        divider = create_divider()
        container.children = [header, divider, content_area]
    else:
        container.children = [content_area]
    
    return {
        'container': container,
        'content_area': content_area
    }


def create_config_summary_panel() -> widgets.VBox:
    """📋 Membuat panel summary konfigurasi menggunakan shared components.
    
    Returns:
        widgets.VBox: Panel summary yang dikonfigurasi
    """
    summary_container = create_responsive_container()
    summary_container.layout.border = '1px dashed #e0e0e0'
    summary_container.layout.border_radius = '4px'
    summary_container.layout.padding = '10px'
    summary_container.layout.margin = '10px 0'
    summary_container.layout.display = 'none'  # Hidden by default
    
    return summary_container


def create_log_components(module_name: str) -> Dict[str, Any]:
    """📝 Membuat komponen log menggunakan shared components.
    
    Args:
        module_name: Nama module
        
    Returns:
        Dictionary berisi komponen log
    """
    try:
        # Gunakan create_log_accordion dari shared components
        log_accordion = create_log_accordion(module_name)
        
        # Extract output widget jika tersedia
        log_output = None
        if hasattr(log_accordion, 'children') and len(log_accordion.children) > 0:
            # Cari output widget di dalam accordion
            for child in log_accordion.children:
                if hasattr(child, 'children'):
                    for grandchild in child.children:
                        if isinstance(grandchild, widgets.Output):
                            log_output = grandchild
                            break
        
        # Fallback jika tidak ditemukan
        if log_output is None:
            log_output = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    max_height='300px',
                    border='1px solid #ddd',
                    overflow='auto'
                )
            )
        
        return {
            'log_accordion': log_accordion,
            'log_output': log_output,
            'entries_container': widgets.VBox()  # Container untuk log entries
        }
        
    except Exception as e:
        logger.error(f"❌ Gagal membuat komponen log untuk {module_name}: {str(e)}")
        
        # Fallback ke komponen basic
        fallback_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='300px',
                border='1px solid #ff6b6b'
            )
        )
        
        fallback_accordion = widgets.Accordion(
            children=[fallback_output],
            titles=[f"📝 {module_name} Logs (Fallback)"]
        )
        
        return {
            'log_accordion': fallback_accordion,
            'log_output': fallback_output,
            'entries_container': widgets.VBox()
        }


def create_info_components(module_name: str) -> Dict[str, Any]:
    """ℹ️ Membuat komponen info menggunakan shared components.
    
    Args:
        module_name: Nama module
        
    Returns:
        Dictionary berisi komponen info
    """
    try:
        # Gunakan create_info_accordion dari shared components
        info_accordion = create_info_accordion(module_name)
        
        # Extract content widget jika tersedia
        info_content = None
        if hasattr(info_accordion, 'children') and len(info_accordion.children) > 0:
            info_content = info_accordion.children[0]
        
        # Fallback jika tidak ditemukan
        if info_content is None:
            info_content = widgets.HTML(
                value=f"<div style='padding: 10px;'>ℹ️ Info untuk {module_name}</div>"
            )
        
        return {
            'accordion': info_accordion,
            'content': info_content
        }
        
    except Exception as e:
        logger.error(f"❌ Gagal membuat komponen info untuk {module_name}: {str(e)}")
        
        # Fallback ke komponen basic
        fallback_content = widgets.HTML(
            value=f"<div style='padding: 10px; color: #ff6b6b;'>⚠️ Error loading info untuk {module_name}</div>"
        )
        
        fallback_accordion = widgets.Accordion(
            children=[fallback_content],
            titles=[f"ℹ️ {module_name} Info (Fallback)"]
        )
        
        return {
            'accordion': fallback_accordion,
            'content': fallback_content
        }


def create_config_cell_ui(
    module: str,
    handler: ConfigCellHandler,
    parent_module: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """🏭 Factory function untuk membuat config cell UI lengkap.
    
    Args:
        module: Nama module
        handler: Instance config handler
        parent_module: Nama parent module opsional
        config: Dictionary konfigurasi opsional
        **kwargs: Argumen tambahan
        
    Returns:
        Dictionary berisi semua komponen UI
    """
    ui_components: Dict[str, Any] = {}
    
    try:
        config = config or {}
        
        # Buat child UI components jika handler mendukung
        child_components = {}
        if hasattr(handler, 'create_ui_components'):
            child_components = handler.create_ui_components(config)
        
        # Dapatkan child content container dengan fallback
        child_content = child_components.get('container', widgets.VBox())
        
        # Setup header dengan overridable defaults menggunakan shared components
        header_title = child_components.get(
            'header_title',
            module.replace('_', ' ').title()
        )
        header_description = child_components.get(
            'header_description',
            f"Konfigurasi untuk {module}"
        )
        header_icon = child_components.get('header_icon', "⚙️")
        
        # Buat komponen menggunakan shared components
        ui_components.update({
            'header': create_header(header_title, header_description, header_icon),
            'status_panel': create_status_panel("Siap", "info"),
            'child_components': child_components,
            'child_content': child_content,
            'config_summary_panel': create_config_summary_panel()
        })
        
        # Buat komponen log menggunakan shared components
        log_components = create_log_components(module)
        ui_components.update({
            'log_output': log_components['log_output'],
            'log_accordion': log_components['log_accordion'],
            'log_entries_container': log_components.get('entries_container')
        })
        
        # Buat komponen info menggunakan shared components
        info_components = create_info_components(module)
        ui_components.update({
            'info_box': info_components['content'],
            'info_accordion': info_components['accordion']
        })
        
        # Buat main container menggunakan shared components
        main_container = create_responsive_container()
        main_container.layout.border = '1px solid #e0e0e0'
        main_container.layout.border_radius = '8px'
        main_container.layout.padding = '15px'
        main_container.layout.margin = '10px 0'
        main_container.layout.box_shadow = '0 2px 4px rgba(0,0,0,0.1)'
        
        # Arrange komponen
        components_to_add = [
            ui_components['header'],
            ui_components['status_panel'],
            ui_components['config_summary_panel'],
            child_content,
            ui_components['log_accordion'],
            ui_components['info_accordion']
        ]
        
        # Filter out None components
        valid_components = [comp for comp in components_to_add if comp is not None]
        main_container.children = tuple(valid_components)
        
        ui_components['container'] = main_container
        
        return ui_components
        
    except Exception as e:
        error_msg = f"Gagal membuat komponen UI untuk {module}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # Return error container menggunakan shared component
        error_card = create_error_card(
            title="Configuration Error",
            value=module,
            description=error_msg
        )
        
        return {
            'container': error_card,
            'error': error_msg,
            'header': widgets.HTML(f"<div style='color: #ff6b6b;'>❌ Error: {module}</div>"),
            'status_panel': create_status_panel(f"Error: {error_msg}", "error"),
            'child_content': error_card
        }