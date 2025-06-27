"""
File: smartcash/ui/dataset/split/split_init.py

Modul ini mengimplementasikan UI konfigurasi split dataset dengan memperluas ConfigCellInitializer.
Mengikuti pola template method dimana parent class menangani inisialisasi umum,
registrasi komponen, dan error handling, sementara class ini mengimplementasikan
komponen UI spesifik dan logika bisnis untuk dataset splitting.
Semua error ditangani melalui sistem error handling terpusat dari parent class.
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
        
        Args:
            config: Dictionary konfigurasi opsional
            parent_id: ID parent component opsional
            component_id: Identifier unik untuk komponen ini
            title: Judul tampilan untuk komponen
            children: List konfigurasi child component opsional
            **kwargs: Argumen keyword tambahan untuk parent class
        """
        # Inisialisasi parent class terlebih dahulu
        super().__init__(
            config=config or {},
            parent_id=parent_id,
            component_id=component_id,
            title=title,
            children=children or [],
            **kwargs
        )
        
        # Inisialisasi handler menggunakan atribut _handler dari parent
        self._handler = None
        
    def create_handler(self) -> SplitConfigHandler:
        """🔧 Membuat dan mengembalikan instance SplitConfigHandler."""
        return SplitConfigHandler(self.config)
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Membuat komponen UI untuk konfigurasi dataset split.
        
        Args:
            config: Dictionary konfigurasi
        
        Returns:
            Dictionary berisi komponen UI. Parent class akan menangani
            penambahan komponen-komponen ini ke container.
        """
        try:
            self._logger.debug("🚀 Membuat komponen UI untuk konfigurasi dataset split")
            
            # Membuat komponen form
            ui_components = create_split_form(config or {})
            
            # Membuat layout utama
            layout_components = create_split_layout(ui_components)
            ui_components.update(layout_components)
            
            self._logger.debug("✅ Berhasil membuat komponen UI")
            return ui_components
            
        except Exception as e:
            self._logger.error(f"❌ Gagal membuat komponen UI: {str(e)}", exc_info=True)
            # Kembalikan dict kosong untuk membiarkan parent menangani error
            return {}
    
    def setup_handlers(self) -> None:
        """⚡ Setup event handler untuk komponen UI."""
        try:
            # Panggil implementasi parent terlebih dahulu
            super().setup_handlers()
            
            # Setup custom event handler kita
            from smartcash.ui.dataset.split.handlers.event_handlers import setup_event_handlers
            setup_event_handlers(self, self.ui_components)
            
            self._logger.debug("✅ Berhasil setup event handler untuk dataset split UI")
            
        except Exception as e:
            self._logger.error(f"❌ Gagal setup event handler: {str(e)}", exc_info=True)
            raise


def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """🎯 Membuat dan menampilkan container konfigurasi split yang mandiri.
    
    Fungsi ini menginisialisasi UI konfigurasi split dan menampilkannya di notebook.
    Fungsi ini tidak mengembalikan dictionary UI components, tetapi hanya menampilkan UI.
    Untuk akses programmatik ke komponen UI, gunakan get_split_config_components().
    
    Args:
        config: Dictionary konfigurasi opsional untuk override defaults.
        **kwargs: Argumen tambahan untuk diteruskan ke initializer.
    """
    try:
        # Inisialisasi split config dengan config dan kwargs yang diberikan
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Inisialisasi UI (ini juga akan mendaftarkan komponen dan setup handler)
        container = initializer.initialize()
        
        # Tampilkan container
        display(container)
        
        logger.info("✅ Split config cell berhasil dibuat dan ditampilkan")
        
    except Exception as e:
        error_msg = f"❌ Gagal membuat split config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Gunakan error handler dari parent class untuk tampilan error yang konsisten
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
    untuk akses programmatik tanpa menampilkan UI di notebook.
    
    Args:
        config: Dictionary konfigurasi opsional untuk override defaults.
        **kwargs: Argumen tambahan untuk diteruskan ke initializer.
               
    Returns:
        Dictionary berisi komponen UI untuk akses programmatik.
    """
    try:
        # Inisialisasi split config dengan config dan kwargs yang diberikan
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Inisialisasi UI (ini juga akan mendaftarkan komponen dan setup handler)
        container = initializer.initialize()
        
        # Kembalikan komponen UI untuk akses programmatik
        return {
            **initializer.ui_components,
            'container': initializer.parent_component.container,
            'content_area': initializer.parent_component.content_area,
            'initializer': initializer
        }
        
    except Exception as e:
        error_msg = f"❌ Gagal membuat komponen split config: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Kembalikan error widget
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        error_widget = create_error_response(
            error_message=error_msg,
            error=e,
            title="🚨 Error in Dataset Split Configuration",
            include_traceback=True
        )
        return {'error': error_widget}