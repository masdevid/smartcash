"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: UI konfigurasi split dataset yang memanfaatkan komponen dari parent class
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display
import logging

# Initializers
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer

# Local components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

logger = logging.getLogger(__name__)

# Constants
MODULE_NAME = "split_config"

class SplitConfigInitializer(ConfigCellInitializer):
    """🎯 Inisialisasi komponen UI konfigurasi dataset split."""
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: str = MODULE_NAME,
        title: str = "📊 Dataset Split Configuration",
        children: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Inisialisasi split config initializer.
        
        Parent class (ConfigCellInitializer) akan otomatis membuat:
        - ParentComponentManager dengan header, status panel, log accordion, info box
        - UILoggerBridge untuk logging terpusat
        - Component registry untuk tracking
        
        Args:
            config: Dictionary konfigurasi opsional
            parent_id: ID parent component opsional
            component_id: Identifier unik untuk komponen ini
            title: Judul tampilan untuk komponen
            children: List konfigurasi child component opsional
            **kwargs: Argumen keyword tambahan untuk parent class
        """
        # Inisialisasi parent class - ini akan membuat semua komponen dasar
        super().__init__(
            config=config or {},
            parent_id=parent_id,
            component_id=component_id,
            title=title,
            children=children or [],
            **kwargs
        )
        
    def create_handler(self) -> SplitConfigHandler:
        """🔧 Membuat dan mengembalikan instance SplitConfigHandler.
        
        Handler akan memiliki akses ke logger bridge dari parent.
        """
        handler = SplitConfigHandler(self.config)
        # Logger bridge sudah ada dari parent class
        if hasattr(handler, 'set_logger_bridge') and self._logger_bridge:
            handler.set_logger_bridge(self._logger_bridge)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Membuat komponen UI spesifik untuk split configuration.
        
        Parent class sudah menyediakan:
        - Container dengan header dan status panel
        - Log accordion dan info box
        - Logger bridge untuk logging
        
        Method ini hanya perlu membuat form components spesifik.
        
        Args:
            config: Dictionary konfigurasi
        
        Returns:
            Dictionary berisi komponen UI form split
        """
        try:
            # Buat form components untuk split configuration
            form_components = create_split_form(config)
            
            # Tambahkan logger bridge dari parent ke form components
            form_components['logger_bridge'] = self._logger_bridge
            
            # Buat layout dengan form components
            layout_components = create_split_layout(form_components)
            
            # Return komponen untuk ditambahkan ke parent container
            # Parent class akan menambahkan ini ke content_area
            return {
                'container': layout_components['container'],
                'form_components': form_components,
                'layout': layout_components,
                **form_components  # Include individual form components
            }
            
        except Exception as e:
            self._logger.error(f"❌ Gagal membuat UI components: {str(e)}", exc_info=True)
            raise
            
    def setup_handlers(self) -> None:
        """⚡ Setup event handler menggunakan logger bridge dari parent.
        
        Parent class sudah setup:
        - Basic event handlers
        - Logger bridge integration
        - Error handling
        """
        try:
            # Panggil implementasi parent terlebih dahulu
            super().setup_handlers()
            
            # Setup custom event handler untuk split UI
            from smartcash.ui.dataset.split.handlers.event_handlers import setup_event_handlers
            
            # Pass logger bridge dari parent ke event handlers
            setup_event_handlers(self, self.ui_components)
            
            # Update status panel dari parent component
            if hasattr(self.parent_component, 'update_status'):
                self.parent_component.update_status("✅ Event handlers berhasil di-setup", "success")
            
            self._logger.debug("✅ Berhasil setup event handler untuk dataset split UI")
            
        except Exception as e:
            self._logger.error(f"❌ Gagal setup event handler: {str(e)}", exc_info=True)
            # Update status panel dengan error
            if hasattr(self.parent_component, 'update_status'):
                self.parent_component.update_status(f"❌ Error: {str(e)}", "error")
            raise


def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """🎯 Membuat dan menampilkan container konfigurasi split.
    
    Fungsi ini menginisialisasi UI konfigurasi split dan menampilkannya di notebook.
    Semua komponen dasar (header, status panel, log, info) sudah disediakan oleh parent class.
    
    Args:
        config: Dictionary konfigurasi opsional untuk override defaults.
        **kwargs: Argumen tambahan untuk diteruskan ke initializer.
    """
    try:
        # Inisialisasi split config
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Inisialisasi UI - parent class akan membuat semua komponen
        container = initializer.initialize()
        
        # Tampilkan container
        display(container)
        
        logger.info("✅ Split config cell berhasil dibuat dan ditampilkan")
        
    except Exception as e:
        error_msg = f"❌ Gagal membuat split config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Gunakan error handler dari parent class
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        error_widget = create_error_response(
            error_message=error_msg,
            error=e,
            title="🚨 Error in Dataset Split Configuration",
            include_traceback=True
        )
        display(error_widget)


def get_split_config_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """📦 Membuat dan mengembalikan komponen UI split config untuk akses programmatik.
    
    Fungsi ini membuat UI konfigurasi split dan mengembalikan dictionary komponen
    termasuk komponen dari parent class.
    
    Args:
        config: Dictionary konfigurasi opsional untuk override defaults.
        **kwargs: Argumen tambahan untuk diteruskan ke initializer.
               
    Returns:
        Dictionary berisi semua komponen UI termasuk:
        - parent: ParentComponentManager
        - container: Root container
        - status_panel: Status panel dari parent
        - log_accordion: Log accordion dari parent  
        - form_components: Form components spesifik split
    """
    try:
        # Inisialisasi split config
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Inisialisasi UI tanpa display
        initializer.initialize()
        
        # Return semua komponen termasuk dari parent
        return {
            'initializer': initializer,
            'container': initializer.get_container(),
            'parent': initializer.parent_component,
            'ui_components': initializer.ui_components,
            'handler': initializer.handler,
            'logger_bridge': initializer._logger_bridge
        }
        
    except Exception as e:
        error_msg = f"❌ Gagal membuat split config components: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise