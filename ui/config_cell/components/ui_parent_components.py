"""
File: smartcash/ui/config_cell/components/ui_parent_components.py
Deskripsi: Parent Component Management untuk Config Cell

Modul ini menangani pembuatan dan management parent component
yang dapat contain dan orchestrate child components dalam struktur hierarkis.
"""
from typing import Dict, Any, Optional, List, Type, TypeVar, Callable, Union
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.ui.config_cell.components.component_registry import component_registry
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import handle_ui_errors

# Import shared components
from smartcash.ui.components import (
    create_responsive_container,
    create_section_title,
    create_divider,
    create_error_card,
    create_header,
    create_status_panel,
    create_log_accordion,
    create_info_accordion
)

logger = get_logger(__name__)
T = TypeVar('T', bound=ConfigCellHandler)

def create_container(title: str = None, container_id: str = None) -> Dict[str, Any]:
    """🏗️ Membuat styled container menggunakan shared components untuk mengelompokkan komponen UI terkait.
    
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
    
    # Tambahkan ID untuk debugging dan testing yang lebih mudah
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


class ParentComponentManager:
    """🎯 Mengelola relasi parent-child antara komponen UI."""
    
    def __init__(self, parent_id: str, title: str = None):
        """🚀 Inisialisasi dengan parent ID unik dan judul opsional.
        
        Args:
            parent_id: Identifier unik untuk parent component ini
            title: Judul opsional untuk parent component
        """
        self.parent_id = parent_id
        self.title = title or f"📦 Parent_{parent_id}"
        self.children: List[str] = []
        self.components: Dict[str, Any] = {}
        self._initialize_container()
    
    def _initialize_container(self):
        """🏗️ Inisialisasi parent container menggunakan UI factory."""
        try:
            parent_ui = create_container(title=self.title, container_id=self.parent_id)
            self.container = parent_ui['container']
            self.content_area = parent_ui['content_area']
            
            # Pastikan container dan content_area adalah widget yang valid
            if not isinstance(self.container, widgets.Widget):
                raise TypeError(f"Container harus berupa Widget, dapat {type(self.container)}")
            if not isinstance(self.content_area, widgets.Widget):
                raise TypeError(f"Content area harus berupa Widget, dapat {type(self.content_area)}")
                
            logger.debug(f"✅ Container initialized untuk parent {self.parent_id}")
            
        except Exception as e:
            logger.error(f"❌ Gagal menginisialisasi container untuk {self.parent_id}: {str(e)}")
            # Fallback ke basic container menggunakan shared components
            self.container = create_responsive_container()
            self.container.layout.border = '1px solid #ff6b6b'
            self.container.layout.border_radius = '4px'
            self.container.layout.padding = '10px'
            self.container.layout.margin = '5px 0'
            
            self.content_area = widgets.VBox()
            
            error_header = widgets.HTML(f"<div style='color: #ff6b6b; font-weight: bold;'>⚠️ {self.title}</div>")
            self.container.children = [error_header, self.content_area]
    
    @handle_ui_errors
    def add_child_component(
        self, 
        child_id: str, 
        component: Union[Dict[str, Any], str],
        handler: Optional[ConfigCellHandler] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """➕ Tambahkan child component ke parent ini.
        
        Args:
            child_id: ID unik untuk child component
            component: Component data (dict) atau module name (string)
            handler: Handler konfigurasi opsional
            config: Dictionary konfigurasi opsional
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen yang ditambahkan
        """
        try:
            # Jika component adalah string (module name), buat menggunakan factory
            if isinstance(component, str):
                child_comp = create_config_cell_ui(
                    module=component,
                    handler=handler,
                    config=config or {},
                    **kwargs
                )
            else:
                child_comp = component
            
            # Pastikan child_comp adalah dictionary yang valid
            if not isinstance(child_comp, dict):
                raise TypeError(f"Child component harus berupa dictionary, dapat {type(child_comp)}")
            
            # Register child component
            full_child_id = f"{self.parent_id}.{child_id}"
            component_registry.register_component(
                component_id=full_child_id,
                component=child_comp,
                parent_id=self.parent_id
            )
            
            # Tambahkan ke children list
            if child_id not in self.children:
                self.children.append(child_id)
            
            # Store component reference
            self.components[child_id] = child_comp
            
            # Tambahkan child container ke content area jika ada
            if 'container' in child_comp and isinstance(child_comp['container'], widgets.Widget):
                current_children = list(self.content_area.children)
                if child_comp['container'] not in current_children:
                    current_children.append(child_comp['container'])
                    self.content_area.children = tuple(current_children)
            
            logger.debug(f"✅ Child component {child_id} berhasil ditambahkan ke {self.parent_id}")
            return child_comp
            
        except Exception as e:
            error_msg = f"❌ Gagal menambahkan child component {child_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Buat error widget menggunakan shared component
            error_card = create_error_card(
                title="Component Error",
                value=f"Child {child_id}",
                description=error_msg
            )
            
            current_children = list(self.content_area.children)
            current_children.append(error_card)
            self.content_area.children = tuple(current_children)
            
            return {'error': error_msg, 'error_widget': error_card}
    
    def get_child_component(self, child_id: str) -> Optional[Dict[str, Any]]:
        """🔍 Dapatkan child component berdasarkan ID.
        
        Args:
            child_id: ID child component yang akan diambil
            
        Returns:
            Dictionary child component atau None jika tidak ditemukan
        """
        # Cek di local components terlebih dahulu
        if child_id in self.components:
            return self.components[child_id]
        
        # Cek di registry
        full_child_id = f"{self.parent_id}.{child_id}"
        return component_registry.get_component(full_child_id)
    
    def remove_child_component(self, child_id: str) -> bool:
        """➖ Hapus child component.
        
        Args:
            child_id: ID child component yang akan dihapus
            
        Returns:
            bool: True jika berhasil dihapus, False jika tidak ditemukan
        """
        if child_id not in self.children:
            return False
            
        try:
            # Hapus dari container
            child_comp = self.get_child_component(child_id)
            if child_comp and 'container' in child_comp:
                current_children = list(self.content_area.children)
                if child_comp['container'] in current_children:
                    current_children.remove(child_comp['container'])
                    self.content_area.children = tuple(current_children)
            
            # Cleanup registry
            full_child_id = f"{self.parent_id}.{child_id}"
            component_registry.unregister_component(full_child_id)
            
            # Cleanup local references
            self.children.remove(child_id)
            if child_id in self.components:
                del self.components[child_id]
            
            logger.debug(f"✅ Child component {child_id} berhasil dihapus dari {self.parent_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gagal menghapus child component {child_id}: {str(e)}")
            return False
    
    def display(self):
        """📺 Tampilkan parent component dan semua children-nya."""
        if self.container:
            display(self.container)
        else:
            logger.error(f"❌ Container tidak tersedia untuk {self.parent_id}")
    
    def get_all_children(self) -> List[str]:
        """📋 Dapatkan list semua child ID."""
        return self.children.copy()
    
    def clear_children(self):
        """🧹 Hapus semua child components."""
        children_to_remove = self.children.copy()
        for child_id in children_to_remove:
            self.remove_child_component(child_id)
    
    def update_title(self, new_title: str):
        """✏️ Update judul parent component."""
        self.title = new_title
        # Re-initialize container dengan judul baru
        self._initialize_container()
    
    def get_status(self) -> Dict[str, Any]:
        """📊 Dapatkan status parent component."""
        return {
            'parent_id': self.parent_id,
            'title': self.title,
            'children_count': len(self.children),
            'children_ids': self.children.copy(),
            'container_type': type(self.container).__name__,
            'is_initialized': self.container is not None
        }


def create_parent_component(
    parent_id: str,
    title: Optional[str] = None,
    children: Optional[List[Dict[str, Any]]] = None
) -> ParentComponentManager:
    """🏭 Membuat parent component manager baru.
    
    Args:
        parent_id: Identifier unik untuk parent component
        title: Judul opsional untuk parent component
        children: List konfigurasi child component opsional dengan key:
            - id: Identifier unik untuk child
            - module: Nama module (diteruskan ke create_config_cell_ui)
            - handler: Instance config handler opsional
            - config: Dictionary konfigurasi opsional
            - **kwargs: Argumen tambahan untuk create_config_cell_ui
            
    Returns:
        Instance ParentComponentManager
    """
    try:
        parent = ParentComponentManager(parent_id, title)
        
        # Tambahkan children jika provided
        if children:
            for child in children:
                if 'id' not in child:
                    logger.warning(f"⚠️ Child component tanpa ID akan diabaikan: {child}")
                    continue
                    
                parent.add_child_component(
                    child_id=child['id'],
                    component=child.get('module', child.get('component', '')),
                    handler=child.get('handler'),
                    config=child.get('config'),
                    **{k: v for k, v in child.items() 
                       if k not in ('id', 'module', 'component', 'handler', 'config')}
                )
        
        return parent
        
    except Exception as e:
        logger.error(f"❌ Gagal membuat parent component {parent_id}: {str(e)}")
        
        # Return fallback parent dengan error indication
        fallback_parent = ParentComponentManager(parent_id, f"❌ Error: {title or parent_id}")
        return fallback_parent


def create_config_cell_ui(
    module: str,
    handler: ConfigCellHandler,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """🏭 Factory function untuk membuat config cell UI menggunakan shared components.
    
    Args:
        module: Nama module
        handler: Instance config handler
        config: Dictionary konfigurasi opsional
        **kwargs: Argumen tambahan
        
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        config = config or {}
        
        # Buat komponen UI menggunakan shared components
        ui_components = {
            'header': create_header(
                title=module.replace('_', ' ').title(),
                description=f"Configuration for {module}",
                icon="⚙️"
            ),
            'status_panel': create_status_panel("Ready", "info"),
            'log_accordion': create_log_accordion(module),
            'info_accordion': create_info_accordion(module)
        }
        
        # Buat child components jika handler support
        if hasattr(handler, 'create_ui_components'):
            child_components = handler.create_ui_components(config)
            ui_components.update(child_components)
        
        # Buat main container menggunakan shared components
        main_container = create_responsive_container()
        
        # Arrange components
        components_to_add = [
            ui_components.get('header'),
            ui_components.get('status_panel'),
            ui_components.get('child_content', widgets.VBox()),
            ui_components.get('log_accordion'),
            ui_components.get('info_accordion')
        ]
        
        # Filter out None components
        valid_components = [comp for comp in components_to_add if comp is not None]
        main_container.children = tuple(valid_components)
        
        ui_components['container'] = main_container
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Gagal membuat config cell UI untuk {module}: {str(e)}")
        
        # Return error container menggunakan shared component
        error_card = create_error_card(
            title="Configuration Error",
            value=module,
            description=f"Failed to create UI: {str(e)}"
        )
        
        return {
            'container': error_card,
            'error': str(e)
        }


# Tambahan utility functions untuk parent component management
def get_parent_component_info(parent_id: str) -> Optional[Dict[str, Any]]:
    """📋 Dapatkan informasi parent component dari registry.
    
    Args:
        parent_id: ID parent component
        
    Returns:
        Dictionary informasi parent component atau None jika tidak ditemukan
    """
    return component_registry.get_component(parent_id)


def update_parent_component_title(parent_id: str, new_title: str) -> bool:
    """✏️ Update judul parent component.
    
    Args:
        parent_id: ID parent component
        new_title: Judul baru
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        parent_info = component_registry.get_component(parent_id)
        if parent_info and 'initializer' in parent_info:
            initializer = parent_info['initializer']
            if hasattr(initializer, 'parent_component'):
                initializer.parent_component.update_title(new_title)
                return True
        return False
    except Exception as e:
        logger.error(f"❌ Gagal update title untuk {parent_id}: {str(e)}")
        return False


def cleanup_parent_component(parent_id: str) -> bool:
    """🧹 Cleanup parent component dan semua children-nya.
    
    Args:
        parent_id: ID parent component
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        parent_info = component_registry.get_component(parent_id)
        if parent_info and 'initializer' in parent_info:
            initializer = parent_info['initializer']
            if hasattr(initializer, 'cleanup'):
                initializer.cleanup()
                return True
        
        # Fallback cleanup dari registry
        component_registry.unregister_component(parent_id)
        return True
        
    except Exception as e:
        logger.error(f"❌ Gagal cleanup parent component {parent_id}: {str(e)}")
        return False


def list_all_parent_components() -> List[str]:
    """📋 Dapatkan list semua parent component yang terdaftar.
    
    Returns:
        List ID parent components
    """
    try:
        all_components = component_registry.list_components()
        parent_components = []
        
        for comp_id in all_components:
            comp_info = component_registry.get_component(comp_id)
            if comp_info and 'parent_component' in comp_info:
                parent_components.append(comp_id)
        
        return parent_components
        
    except Exception as e:
        logger.error(f"❌ Gagal mendapatkan list parent components: {str(e)}")
        return []