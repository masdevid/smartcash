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
    restore_stdout
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
            config: Dictionary konfigurasi
            parent_id: ID parent component opsional untuk relasi hierarkis
            component_id: Identifier unik opsional untuk komponen ini
            logger_bridge: Logger bridge opsional untuk UI logging
            title: Judul opsional untuk komponen
            children: List konfigurasi child component opsional
            **kwargs: Parameter konfigurasi tambahan
        """
        self.config = config or {}
        self.parent_id = parent_id
        self.component_id = component_id or self.__class__.__name__
        self.title = title or self.component_id
        
        # Inisialisasi dictionary komponen UI
        self.ui_components: Dict[str, Any] = {}
        
        # Store logger bridge jika provided, akan diinisialisasi setelah komponen UI dibuat
        self._logger_bridge = logger_bridge
        self._handler: Optional[T] = None  # Inisialisasi atribut protected
        self._is_initialized = False
        self._suppress_output = False
        self._original_stdout = None
        self._original_stderr = None
        self._logger = logger.getChild(self.component_id)
        
        # Inisialisasi parent component manager
        self.parent_component = ParentComponentManager(
            parent_id=self.component_id,
            title=self.title
        )
        
        # Inisialisasi logger bridge setelah parent component dibuat
        if self._logger_bridge is None:
            self._logger_bridge = UILoggerBridge(
                ui_components={
                    'parent': self.parent_component,
                    'container': getattr(self.parent_component, 'container', None)
                },
                logger_name=f"{self.__class__.__name__.lower()}_bridge"
            )
        
        # Store konfigurasi children untuk lazy initialization
        self._children_config = children or []
        self._children: List[Any] = []
    
    def _setup_output_suppression(self) -> None:
        """⚡ Setup output suppression menggunakan logging utilities proyek."""
        from smartcash.ui.utils.logging_utils import (
            setup_aggressive_log_suppression,
            setup_stdout_suppression
        )
        
        # Setup aggressive log suppression untuk library yang berisik
        setup_aggressive_log_suppression()
        
        # Setup stdout/stderr suppression jika ada UI components
        if hasattr(self, 'ui_components') and self.ui_components:
            setup_stdout_suppression()
            
        self._logger.debug("✅ Output suppression enabled")
        
    def _restore_output(self) -> None:
        """🔄 Restore original output settings menggunakan logging utilities proyek."""
        from smartcash.ui.utils.logging_utils import (
            restore_stdout,
            allow_tqdm_display
        )
        
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
                from smartcash.ui.config_cell.components.component_registry import component_registry
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
        
        # Hanya register jika belum terdaftar
        if not component_registry.get_component(self._component_id):
            component_registry.register_component(
                component_id=self._component_id,
                component={
                    **self.ui_components,
                    'container': self.parent_component.container,
                    'content_area': self.parent_component.content_area
                },
                parent_id=self.parent_id
            )
            
            # Setup children jika ada
            if self._children_config:
                self._initialize_children()
    
    @property
    def handler(self) -> T:
        """🔧 Lazy initialization dari configuration handler."""
        if self._handler is None:
            self._handler = self.create_handler()
        return self._handler
    
    @abstractmethod
    def create_handler(self) -> T:
        """🏭 Membuat dan return instance configuration handler.
        
        Returns:
            Instance dari ConfigCellHandler subclass.
        """
        pass
        
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Membuat dan return komponen UI berdasarkan config yang diberikan.
        
        Args:
            config: Nilai konfigurasi saat ini
            
        Returns:
            Dictionary komponen UI.
        """
        pass
        
    def setup_handlers(self) -> None:
        """⚡ Setup event handler untuk komponen UI.
        
        Method ini bisa dioverride oleh subclass untuk setup
        event handler, observer, atau callback yang diperlukan UI.
        Implementasi parent class harus dipanggil terlebih dahulu menggunakan super().
        """
        # Default implementation - bisa dioverride oleh subclass
        self._logger.debug("🔗 Setting up base event handlers")
        pass
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> widgets.Widget:
        """🚀 Inisialisasi komponen UI dan return root widget.
        
        Args:
            config: Konfigurasi opsional untuk override initial config
            
        Returns:
            Root widget yang berisi UI yang diinisialisasi
        """
        if config is not None:
            self.config = config
            
        try:
            # Setup output suppression jika diperlukan
            if self._suppress_output:
                self._setup_output_suppression()
            
            # Buat handler menggunakan protected attribute secara langsung
            self._handler = self.create_handler()
            
            # Buat komponen UI menggunakan parent component system
            self._setup_ui_components()
            
            # Inisialisasi child components jika ada
            self._initialize_children()
            
            # Setup event handlers
            self.setup_handlers()
            
            # Register komponen
            self._register_component()
            
            # Mark sebagai initialized
            self._is_initialized = True
            
            # Pastikan parent component's container terinisialisasi dengan benar
            if not hasattr(self.parent_component, 'container') or self.parent_component.container is None:
                self._logger.error("❌ Parent component container tidak terinisialisasi")
                raise RuntimeError("Parent component container tidak terinisialisasi")
            
            # Dapatkan container widget dari parent component
            container = self.parent_component.container
            
            # Pastikan kita return widget
            if not isinstance(container, widgets.Widget):
                error_msg = f"❌ Expected widget tapi dapat {type(container).__name__}"
                self._logger.error(error_msg)
                return create_error_response("Initialization Error", error_msg)
                
            self._logger.info(f"✅ {self.__class__.__name__} berhasil diinisialisasi")
            return container
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi {self.__class__.__name__}: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            error_widget = create_error_response("Initialization Error", error_msg)
            if not isinstance(error_widget, widgets.Widget):
                # Jika create_error_response tidak return widget, buat basic widget
                return widgets.HTML(f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 4px;'>{error_msg}</div>")
            return error_widget
            
        finally:
            # Restore output settings jika disuppress
            if self._suppress_output:
                self._restore_output()
                
    def _setup_ui_components(self) -> None:
        """🎨 Setup komponen UI menggunakan parent component system."""
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
        # Register komponen sebelum return container
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