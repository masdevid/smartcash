"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer dengan shared state dan YAML persistence

Modul ini menyediakan class ConfigCellInitializer yang berfungsi sebagai lapisan orkestrasi
untuk UI konfigurasi. Menangani inisialisasi, lifecycle management, dan registrasi komponen
sambil mendelegasikan pembuatan komponen UI ke modul components.
"""

from __future__ import annotations

# Standard library
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, Union, List
# Third-party
import ipywidgets as widgets

# SmartCash - UI Components
from smartcash.ui.config_cell.components import component_registry
from smartcash.ui.config_cell.components.ui_parent_components import ParentComponentManager, create_parent_component
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import create_error_response
from smartcash.ui.utils.logger_bridge import UILoggerBridge
from smartcash.ui.utils.logging_utils import (
    restore_stdout,
    allow_tqdm_display
)

# Type variables
T = TypeVar('T', bound=ConfigCellHandler)

# Logger setup
logger = logging.getLogger(__name__)

class ConfigCellInitializer(Generic[T], ABC):
    """🎯 Orkestrasi inisialisasi dan lifecycle configuration cell.
    
    Abstract base class ini menangani core initialization flow, registrasi komponen,
    dan lifecycle management UI konfigurasi. Mendelegasikan pembuatan komponen UI
    ke modul components dan fokus pada orkestrasi.
    
    Type Parameters:
        T: Type dari configuration handler, harus subclass dari ConfigCellHandler
        
    Subclass harus mengimplementasikan:
        - create_handler(): Membuat dan return instance configuration handler
        - create_ui_components(): Membuat dan return dictionary komponen UI
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: Optional[str] = None,
        logger_bridge: Optional[UILoggerBridge] = None,
        title: Optional[str] = None,
        children: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """🚀 Inisialisasi config cell dengan config dan parent ID opsional.
        
        Args:
            config: Dictionary konfigurasi opsional
            parent_id: ID parent component (untuk nested components)
            component_id: ID unik untuk komponen ini
            logger_bridge: Logger bridge untuk UI logging (internal use)
            title: Judul untuk parent component
            children: List konfigurasi child component
            **kwargs: Argumen tambahan untuk ekstensibilitas
        """
        # Core state
        self.config = config or {}
        self.parent_id = parent_id
        self.component_id = component_id or self.__class__.__name__.replace('Initializer', '').lower()
        self.title = title or f"⚙️ {self.component_id.replace('_', ' ').title()}"
        
        # Children configuration
        self._children_config = children or []
        self._children: List[Any] = []
        
        # UI Components placeholder
        self.ui_components: Dict[str, Any] = {}
        self.parent_component: Optional[ParentComponentManager] = None
        
        # Handler akan diinisialisasi nanti
        self._handler: Optional[T] = None
        
        # Logging infrastructure
        self._logger_bridge = logger_bridge
        self._logger = logger  # Default logger, akan diupdate di _setup_logging
        
        # Initialization state tracking
        self._is_initialized = False
        
        # Store additional kwargs untuk extensibility
        self._kwargs = kwargs
    
    @abstractmethod
    def create_handler(self) -> T:
        """🔧 Membuat dan mengembalikan instance configuration handler.
        
        Subclass harus mengimplementasikan ini untuk menyediakan handler spesifik mereka.
        
        Returns:
            Instance dari configuration handler
        """
        pass
    
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Membuat dan mengembalikan dictionary komponen UI.
        
        Subclass harus mengimplementasikan ini untuk menyediakan UI components spesifik.
        Handler instance dapat diakses melalui self._handler.
        
        Args:
            config: Dictionary konfigurasi
            
        Returns:
            Dictionary berisi komponen UI (widgets)
        """
        pass
    
    def initialize(self) -> widgets.Widget:
        """🚀 Entry point utama untuk inisialisasi komponen UI.
        
        Menjalankan full initialization flow dan mengembalikan root container widget.
        Method ini mengorkestrasi semua langkah inisialisasi dengan error handling.
        
        Returns:
            Root container widget yang siap ditampilkan
            
        Raises:
            Exception: Jika inisialisasi gagal (error akan di-handle dan ditampilkan)
        """
        try:
            # Prevent double initialization
            if self._is_initialized:
                self._logger.warning(f"⚠️ {self.component_id} sudah diinisialisasi, skip re-initialization")
                return self.get_container()
            
            # Step 1: Buat parent component PERTAMA
            self.parent_component = ParentComponentManager(
                parent_id=self.component_id,
                title=self.title
            )
            
            # Step 2: Setup logging infrastructure
            self._setup_logging()
            
            # Step 3: Inisialisasi handler
            self._handler = self.create_handler()
            self._logger.debug(f"✅ Handler {type(self._handler).__name__} berhasil dibuat")
            
            # Step 4: Buat dan setup UI components
            self._initialize_ui()
            
            # Step 5: Inisialisasi children jika ada
            self._initialize_children()
            
            # Step 6: Registrasi komponen
            self._register_component()
            
            # Step 7: Restore output settings
            self._restore_output_settings()
            
            # Mark sebagai initialized
            self._is_initialized = True
            
            self._logger.info(f"✅ {self.__class__.__name__} berhasil diinisialisasi")
            
            return self.get_container()
            
        except Exception as e:
            self._logger.error(f"❌ Gagal menginisialisasi {self.__class__.__name__}: {str(e)}", exc_info=True)
            
            # Create error widget dengan context
            error_widget = create_error_response(
                error_message=f"Initialization failed: {str(e)}",
                error=e,
                title=f"🚨 Error in {self.title}",
                include_traceback=True
            )
            
            # Ensure we have a container
            if not self.parent_component:
                self.parent_component = ParentComponentManager(
                    parent_id=f"{self.component_id}_error",
                    title=f"❌ {self.title} (Error)"
                )
            
            # Set error widget as content
            self.parent_component.content_area.children = (error_widget,)
            
            return self.parent_component.container
    
    def _initialize_ui(self) -> None:
        """🎨 Inisialisasi komponen UI dan tambahkan ke parent component."""
        # Buat komponen UI utama
        self.ui_components = self.create_ui_components(self.config)
        
        # Tambahkan komponen utama ke parent component
        if 'container' in self.ui_components:
            # Jika komponen menyediakan container sendiri, gunakan sebagai main content
            self.parent_component.content_area.children = (self.ui_components['container'],)
        else:
            # Otherwise, tambahkan semua komponen ke content area
            widgets_to_add = [
                widget for key, widget in self.ui_components.items()
                if isinstance(widget, widgets.Widget)
            ]
            if widgets_to_add:
                self.parent_component.content_area.children = tuple(widgets_to_add)
            else:
                # Jika tidak ada widget, buat placeholder
                placeholder = widgets.HTML("<div style='padding: 20px; text-align: center; color: #666;'>📦 Komponen UI sedang dimuat...</div>")
                self.parent_component.content_area.children = (placeholder,)
    
    def _initialize_children(self) -> None:
        """👶 Inisialisasi child components jika ada yang dikonfigurasi."""
        if not self._children_config:
            return
            
        for child_config in self._children_config:
            try:
                # Buat child component menggunakan factory function
                child = create_parent_component(
                    parent_id=f"{self.component_id}.{child_config['id']}",
                    **{k: v for k, v in child_config.items() if k != 'id'}
                )
                self._children.append(child)
                
                # Tambahkan child ke parent component
                self.parent_component.add_child_component(
                    child_id=child_config['id'],
                    component=child,
                    config=child_config.get('config', {})
                )
                
            except Exception as e:
                self._logger.error(
                    f"❌ Gagal menginisialisasi child component {child_config.get('id')}: {str(e)}",
                    exc_info=True
                )
    
    def _register_component(self) -> None:
        """📋 Register komponen ini dengan component registry dan setup parent-child relationships."""
        # Generate full component ID dengan parent prefix jika parent exists
        full_component_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
        
        # Siapkan component data dengan container dan content area
        component_data = {
            **getattr(self, 'ui_components', {}),
            'container': self.parent_component.container,
            'content_area': self.parent_component.content_area,
            'initializer': self  # Referensi ke initializer untuk akses programmatic
        }
        
        # Register komponen utama
        component_registry.register_component(
            component_id=full_component_id,
            component=component_data,
            parent_id=self.parent_id
        )
        
        self._logger.debug(f"📋 Component {full_component_id} berhasil didaftarkan")
    
    def get_container(self) -> widgets.Widget:
        """📦 Dapatkan root container widget.
        
        Returns:
            Root container widget dari parent component
        """
        # Ensure parent component exists
        if not self.parent_component:
            raise RuntimeError(f"❌ Parent component belum diinisialisasi untuk {self.component_id}")
        
        # Register komponen jika belum
        if not self._is_initialized:
            self._register_component()
            
        return self.parent_component.container

    def cleanup(self) -> None:
        """🧹 Release semua resource dan unregister komponen."""
        try:
            # Unregister from component registry
            full_component_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
            component_registry.unregister_component(full_component_id)
            
            # Cleanup children
            for child in self._children:
                if hasattr(child, 'cleanup'):
                    child.cleanup()
            
            # Reset state
            self._is_initialized = False
            self._handler = None
            self.ui_components.clear()
            
            self._logger.debug(f"🧹 {self.__class__.__name__} cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"❌ Error during cleanup: {str(e)}", exc_info=True)
    
    def _restore_output_settings(self) -> None:
        """📺 Restore stdout/stderr settings setelah UI initialization.
        
        UI components mungkin suppress output selama initialization,
        method ini memastikan output normal dikembalikan.
        """
        # Restore stdout/stderr jika disuppress
        if hasattr(self, 'ui_components') and self.ui_components:
            restore_stdout()
            
        # Pastikan tqdm bisa menampilkan progress bars
        allow_tqdm_display()
        
        self._logger.debug("✅ Output settings restored")

    def _setup_logging(self) -> None:
        """📝 Inisialisasi infrastructure logging dan redirect semua log ke parent's log accordion."""
        try:
            # Inisialisasi logger bridge dengan parent's UI components jika available
            parent_components = {}
            if self.parent_id:
                parent = component_registry.get_component(self.parent_id)
                if parent and hasattr(parent, 'get'):
                    parent_components = parent
            
            # Gunakan parent's UI components jika available, otherwise gunakan milik kita
            ui_components = parent_components.get('ui_components', {}) or self.ui_components
            
            # Inisialisasi logger bridge
            self._logger_bridge = UILoggerBridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.component_id}"
            )
            
            # Setup logger
            self._logger = self._logger_bridge.logger
            
            # Mark UI sebagai ready untuk flush buffered logs
            if hasattr(self._logger_bridge, 'set_ui_ready'):
                self._logger_bridge.set_ui_ready(True)
                
            self._logger.debug(f"✅ Logging initialized untuk {self.component_id}")
            
        except Exception as e:
            # Fallback ke basic logging jika UI logging setup gagal
            import logging
            self._logger = logging.getLogger(f"smartcash.ui.{self.component_id}")
            self._logger.warning(f"⚠️ Failed to initialize UI logging: {str(e)}", exc_info=True)
    
    def _setup_component_registry(self) -> None:
        """📋 Register komponen ini di component registry."""
        # Buat component ID menggunakan module hierarchy (e.g., 'dataset.split')
        self._component_id = (
            f"{self.parent_id}.{self.component_id}" 
            if self.parent_id 
            else self.component_id
        )
        
        # Register di component registry dengan handler
        component_registry.register_component(
            component_id=self._component_id,
            component={
                'handler': self._handler,
                'initializer': self,
                'config': self.config,
                'ui_components': getattr(self, 'ui_components', {})
            },
            parent_id=self.parent_id
        )
        
        self._logger.debug(f"📋 Component {self._component_id} registered in registry")
    
    def get_handler(self) -> Optional[T]:
        """🔧 Dapatkan instance configuration handler.
        
        Returns:
            Configuration handler instance atau None jika belum diinisialisasi
        """
        return self._handler
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """🔄 Update konfigurasi dan refresh UI.
        
        Args:
            new_config: Dictionary konfigurasi baru
        """
        try:
            # Update config
            self.config.update(new_config)
            
            # Update handler config jika ada
            if self._handler and hasattr(self._handler, 'update_config'):
                self._handler.update_config(new_config)
            
            # Re-initialize UI dengan config baru
            if self._is_initialized:
                self._initialize_ui()
                self._logger.info(f"✅ Config updated dan UI refreshed untuk {self.component_id}")
            
        except Exception as e:
            self._logger.error(f"❌ Gagal update config: {str(e)}", exc_info=True)
            raise
    
    def __repr__(self) -> str:
        """String representasi untuk debugging."""
        return (
            f"{self.__class__.__name__}("
            f"component_id='{self.component_id}', "
            f"parent_id='{self.parent_id}', "
            f"initialized={self._is_initialized})"
        )