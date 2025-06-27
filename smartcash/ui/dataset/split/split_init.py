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

class SplitConfigInitializer(ConfigCellInitializer[SplitConfigHandler]):
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
            # Buat form component
            form_components = create_split_form(self._handler, config)
            
            # Buat layout component dengan form
            layout_components = create_split_layout(
                handler=self._handler,
                form_components=form_components,
                config=config
            )
            
            # Return semua komponen
            return {
                **form_components,
                **layout_components
            }
            
        except Exception as e:
            logger.error(f"❌ Gagal membuat UI components: {str(e)}", exc_info=True)
            # Fail-fast: raise exception untuk di-handle oleh parent
            raise


def display_split_config(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """🎨 Entry point untuk menampilkan UI konfigurasi split dataset.
    
    Fungsi ini membuat dan menampilkan UI konfigurasi dataset split di Jupyter notebook.
    
    Args:
        config: Dictionary konfigurasi opsional untuk override defaults.
        **kwargs: Argumen tambahan untuk diteruskan ke initializer.
    """
    try:
        # Inisialisasi split config dengan config dan kwargs yang diberikan
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Inisialisasi dan dapatkan container
        container = initializer.initialize()
        
        # Display container
        display(container)
        
        logger.info("✅ Split config UI berhasil ditampilkan")
        
    except Exception as e:
        error_msg = f"❌ Gagal menampilkan split config UI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Display error widget
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        error_widget = create_error_response(
            error_message=error_msg,
            error=e,
            title="🚨 Error in Dataset Split Configuration",
            include_traceback=True
        )
        display(error_widget)


def create_split_config_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """🏗️ Factory function untuk membuat komponen split config tanpa menampilkan.
    
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
            'container': container,
            'content_area': initializer.parent_component.content_area if initializer.parent_component else None,
            'initializer': initializer,
            'handler': initializer.get_handler()
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
        return {'error': error_widget, 'error_message': error_msg}