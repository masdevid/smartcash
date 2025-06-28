"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Base class untuk config cell yang menyediakan parent components dan mengelola lifecycle

ARCHITECTURE OVERVIEW:
===================
ConfigCellInitializer adalah base class yang mengimplementasikan Template Method Pattern
untuk membuat UI configuration cells dengan struktur yang konsisten. Class ini:

1. MENYEDIAKAN komponen parent yang WAJIB ada di SEMUA config cells:
   - Header (judul + deskripsi + icon)
   - Status Panel (untuk menampilkan status operasi)
   - Log Accordion (untuk menampilkan log messages)
   - Info Accordion (untuk informasi/dokumentasi)
   - Container structure yang konsisten

2. MENGELOLA lifecycle dan infrastructure:
   - Logger bridge untuk centralized logging
   - Component registry untuk tracking
   - Event handler setup
   - Error handling
   - Resource cleanup

3. MENDELEGASIKAN ke child class:
   - Pembuatan form/UI components spesifik (create_child_components)
   - Pembuatan handler spesifik (create_handler)
   - Content info accordion spesifik (get_info_content)

DESIGN PATTERN:
==============
Template Method Pattern dimana parent class mendefinisikan skeleton algoritma
(struktur UI dan lifecycle), sementara child class mengisi detail spesifik.

HIERARCHY:
=========
ConfigCellInitializer (this class)
    ├── Membuat: Header, Status Panel, Log Accordion, Info Accordion
    ├── Mengelola: Logger Bridge, Registry, Lifecycle
    └── Child Classes (e.g., SplitConfigInitializer)
        └── Membuat: Form components spesifik saja

IMPORTANT RULES:
===============
1. Child class TIDAK BOLEH membuat ulang parent components
2. Child class HANYA membuat form/UI components spesifik
3. Parent components SELALU dibuat oleh ConfigCellInitializer
4. Semua logging HARUS melalui logger bridge
5. Status updates HARUS melalui status panel yang disediakan parent
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
from smartcash.ui.config_cell.components.ui_parent_components import ParentComponentManager
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import create_error_response
from smartcash.ui.utils.logger_bridge import UILoggerBridge
from smartcash.ui.utils.logging_utils import (
    setup_aggressive_log_suppression,
    setup_stdout_suppression,
    restore_stdout,
    allow_tqdm_display
)

# Import shared components untuk parent
from smartcash.ui.components import (
    create_header,
    create_status_panel,
    create_log_accordion,
    create_info_accordion
)

# Type variables
T = TypeVar('T', bound=ConfigCellHandler)

# Logger setup
logger = logging.getLogger(__name__)

class ConfigCellInitializer(Generic[T], ABC):
    """🎯 Base class untuk config cell yang menyediakan parent components.
    
    WHAT THIS CLASS DOES:
    ====================
    1. Membuat dan mengelola SEMUA parent components (header, status, log, info)
    2. Setup infrastructure (logger bridge, registry, event handlers)
    3. Mengelola lifecycle (initialization, cleanup)
    4. Menyediakan template untuk child classes
    
    WHAT CHILD CLASSES DO:
    =====================
    1. Implement create_handler() - membuat handler spesifik
    2. Implement create_child_components() - HANYA form/UI spesifik
    3. Optional: Override get_info_content() - custom info content
    4. Optional: Override setup_handlers() - custom event handlers (MUST call super())
    
    PARENT COMPONENTS (dibuat otomatis oleh class ini):
    ==================================================
    - Header: Judul, deskripsi, dan icon
    - Status Panel: Menampilkan status operasi real-time
    - Log Accordion: Menampilkan semua log messages
    - Info Accordion: Dokumentasi/informasi tambahan
    
    USAGE EXAMPLE:
    =============
    ```python
    class MyConfigInitializer(ConfigCellInitializer):
        def create_handler(self):
            return MyConfigHandler(self.config)
            
        def create_child_components(self, config):
            # HANYA buat form components, JANGAN buat header/status/log/info!
            return {
                'my_input': widgets.Text(value=config.get('value', '')),
                'my_button': widgets.Button(description='Process')
            }
    ```
    
    ANTI-PATTERNS (JANGAN LAKUKAN):
    ==============================
    1. ❌ Child class membuat header sendiri
    2. ❌ Child class membuat status panel sendiri
    3. ❌ Child class membuat log accordion sendiri
    4. ❌ Child class bypass logger bridge
    5. ❌ Child class mengabaikan parent lifecycle
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        **kwargs
    ) -> None:
        """🚀 Inisialisasi config cell dengan parent components.
        
        Args:
            config: Dictionary konfigurasi untuk handler dan UI
            parent_id: ID parent component untuk relasi hierarkis
            component_id: Identifier unik untuk komponen ini (default: class name)
            title: Judul untuk header (default: component_id)
            description: Deskripsi untuk header (default: auto-generated)
            icon: Icon emoji untuk header (default: ⚙️)
            **kwargs: Parameter tambahan untuk extensibility
            
        Note:
            Constructor ini HANYA setup state awal. Actual UI creation
            terjadi di initialize() method untuk mendukung lazy loading.
        """
        self.config = config or {}
        self.parent_id = parent_id
        self.component_id = component_id or self.__class__.__name__
        self.title = title or self.component_id.replace('_', ' ').title()
        self.description = description or f"Configuration for {self.title}"
        self.icon = icon or "⚙️"
        
        # Component dictionaries
        self.ui_components: Dict[str, Any] = {}      # Semua UI components
        self.parent_components: Dict[str, Any] = {}  # Parent components saja
        self.child_components: Dict[str, Any] = {}   # Child components saja
        
        # Handler dan state
        self._handler: Optional[T] = None
        self._is_initialized = False
        self._logger = logger.getChild(self.component_id)
        
        # Setup parent component manager first
        self.parent_component = ParentComponentManager(
            parent_id=self.component_id,
            title=self.title
        )
        
        # Setup logger bridge after parent component is ready
        try:
            self._logger_bridge = UILoggerBridge(
                ui_components={'parent': self.parent_component},
                logger_name=f"{self.component_id}_bridge"
            )
        except Exception as e:
            # Fallback jika UILoggerBridge gagal
            self._logger.warning(f"Failed to create UILoggerBridge: {e}")
            self._logger_bridge = None
        
        # Setup shared configuration manager
        self._shared_config_manager = None
        self._unsubscribe_func = None
        self._setup_shared_config()
        
    @property
    def logger(self):
        """Get the logger instance to use for logging.
        
        Returns:
            The logger bridge if available, otherwise the module logger
        """
        if hasattr(self, '_logger_bridge') and self._logger_bridge:
            return self._logger_bridge
        return self._logger
        
    @property
    def handler(self) -> T:
        """🔧 Get configuration handler dengan lazy initialization.
        
        Returns:
            Configuration handler instance
            
        Note:
            Handler dibuat on-demand saat pertama kali diakses.
            Ini memungkinkan child class setup state sebelum handler creation.
        """
        if self._handler is None:
            self._handler = self.create_handler()
            # Inject logger bridge ke handler jika support
            if hasattr(self._handler, 'set_logger_bridge') and hasattr(self, '_logger_bridge'):
                self._handler.set_logger_bridge(self._logger_bridge)
        return self._handler
    
    @abstractmethod
    def create_handler(self) -> T:
        """🏭 Create configuration handler instance.
        
        MUST BE IMPLEMENTED BY CHILD CLASS.
        
        Returns:
            Instance dari ConfigCellHandler subclass yang sesuai
            
        Example:
            ```python
            def create_handler(self):
                return MySpecificConfigHandler(self.config)
            ```
            
        Note:
            - Method ini dipanggil lazy saat handler pertama kali diakses
            - Handler HARUS extend ConfigCellHandler
            - Logger bridge akan di-inject otomatis setelah creation
        """
        pass
        
    @abstractmethod
    def create_child_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Create child-specific UI components.
        
        MUST BE IMPLEMENTED BY CHILD CLASS.
        
        IMPORTANT:
        =========
        Method ini HANYA membuat form/UI components spesifik untuk functionality
        child class. JANGAN membuat header, status panel, log accordion, atau
        info accordion karena sudah dibuat oleh parent class.
        
        Args:
            config: Current configuration values untuk initialize UI
            
        Returns:
            Dictionary berisi HANYA child UI components, contoh:
            {
                'input_field': widgets.Text(...),
                'slider': widgets.FloatSlider(...),
                'checkbox': widgets.Checkbox(...),
                'container': widgets.VBox([...])  # Optional container
            }
            
        Best Practices:
            1. Gunakan config untuk set initial values
            2. Beri nama descriptive untuk setiap component
            3. Group related components dalam container
            4. JANGAN membuat parent components (header, status, etc)
            
        Example:
            ```python
            def create_child_components(self, config):
                return {
                    'name_input': widgets.Text(
                        value=config.get('name', ''),
                        placeholder='Enter name'
                    ),
                    'save_button': widgets.Button(
                        description='Save',
                        button_style='success'
                    )
                }
            ```
        """
        pass
        
    def get_info_content(self) -> Optional[widgets.Widget]:
        """📚 Get content untuk info accordion.
        
        OPTIONAL - Override untuk custom info content.
        
        Returns:
            Widget untuk info content atau None untuk default
            
        Example:
            ```python
            def get_info_content(self):
                return widgets.HTML('''
                    <h4>How to use this configuration</h4>
                    <ul>
                        <li>Step 1: Enter values</li>
                        <li>Step 2: Click save</li>
                    </ul>
                ''')
            ```
            
        Note:
            Jika return None, info accordion akan use generic content.
            Best practice: Provide helpful documentation untuk users.
        """
        return None
        
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Template method yang orchestrate UI creation.
        
        INTERNAL METHOD - JANGAN OVERRIDE DI CHILD CLASS!
        
        Method ini mengimplementasikan template untuk UI creation:
        1. Membuat parent components (header, status, log, info)
        2. Memanggil create_child_components() untuk child-specific UI
        3. Menyusun semua components dalam struktur yang konsisten
        4. Setup logger bridge dengan semua components
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Complete UI components dictionary dengan struktur:
            {
                # Parent components (dibuat oleh method ini)
                'header': Header widget,
                'status_panel': Status panel widget,
                'log_accordion': Log accordion widget,
                'info_accordion': Info accordion widget,
                
                # Child components (dari create_child_components)
                'child_container': Container untuk child components,
                ...child specific components...
                
                # Infrastructure
                'logger_bridge': UILoggerBridge instance,
                'container': Root container dengan semua components
            }
            
        Raises:
            Exception: Jika child component creation gagal
        """
        try:
            # === 1. CREATE PARENT COMPONENTS ===
            self._logger.debug("Creating parent components...")
            
            # Header
            header = create_header(
                title=self.title,
                description=self.description,
                icon=self.icon
            )
            
            # Status panel
            status_panel = create_status_panel(
                message=f"✅ {self.title} ready",
                status_type="info"
            )
            
            # Log accordion
            log_accordion = create_log_accordion(
                module_name=self.component_id.lower(),
                height='200px'
            )
            
            # Info accordion
            info_content = self.get_info_content()
            if info_content is None:
                # Default info content
                info_content = widgets.HTML(f"""
                    <div style='padding: 10px;'>
                        <p>{self.description}</p>
                        <p style='color: #666; font-size: 0.9em;'>
                            Component ID: {self.component_id}<br>
                            Parent ID: {self.parent_id or 'None'}
                        </p>
                    </div>
                """)
            
            info_accordion = create_info_accordion(
                content=info_content,
                title=f"ℹ️ About {self.title}"
            )
            
            # Store parent components
            self.parent_components = {
                'header': header,
                'status_panel': status_panel,
                'log_accordion': log_accordion,
                'info_accordion': info_accordion
            }
            
            # === 2. CREATE CHILD COMPONENTS ===
            self._logger.debug("Creating child components...")
            self.child_components = self.create_child_components(config)
            
            # === 3. UPDATE LOGGER BRIDGE ===
            # Logger bridge perlu di-reinitialize dengan semua components
            self._logger.debug("Updating logger bridge with all components...")
            
            if self._logger_bridge is not None:
                # Collect all components
                all_components = {
                    **self.parent_components,
                    **self.child_components,
                    'parent': self.parent_component,
                    'log_accordion': self.parent_components.get('log_accordion'),
                    'log_output': self.parent_components.get('log_accordion')  # Fallback
                }
                
                try:
                    # Create new logger bridge dengan complete UI components
                    self._logger_bridge = UILoggerBridge(
                        ui_components=all_components,
                        logger_name=f"{self.component_id}_bridge"
                    )
                    
                    # Re-assign logger if needed
                    if hasattr(self._logger_bridge, 'logger'):
                        self._logger = self._logger_bridge.logger
                except Exception as e:
                    self._logger.warning(f"Failed to update logger bridge: {e}")
            else:
                self._logger.debug("Logger bridge not available, skipping update")
            
            # === 4. ASSEMBLE FINAL STRUCTURE ===
            self._logger.debug("Assembling UI structure...")
            
            # Get child container or create default
            child_container = None
            if 'container' in self.child_components and isinstance(self.child_components['container'], widgets.Widget):
                child_container = self.child_components['container']
            else:
                # Auto-create container from individual widgets
                child_widgets = [
                    widget for widget in self.child_components.values()
                    if isinstance(widget, widgets.Widget)
                ]
                if child_widgets:
                    child_container = widgets.VBox(
                        children=child_widgets,
                        layout=widgets.Layout(
                            width='100%',
                            padding='10px',
                            gap='10px'
                        )
                    )
                else:
                    child_container = widgets.HTML(
                        "<p style='padding: 20px; color: #666;'>No child components defined</p>"
                    )
            
            # Create main container dengan struktur konsisten
            children = []
            if header and isinstance(header, widgets.Widget):
                children.append(header)
                
            # Add status panel with proper spacing
            if status_panel and isinstance(status_panel, widgets.Widget):
                status_container = widgets.VBox(
                    [status_panel],
                    layout=widgets.Layout(
                        margin='0 0 10px 0',
                        padding='5px',
                        border='1px solid #e0e0e0',
                        border_radius='4px',
                        background='#f9f9f9'
                    )
                )
                children.append(status_container)
            
            # Add child container with proper spacing
            if child_container and isinstance(child_container, widgets.Widget):
                children.append(child_container)
                
            # Add log accordion if available
            log_widget = None
            if log_accordion and isinstance(log_accordion, widgets.Widget):
                log_widget = log_accordion
            elif 'log_accordion' in self.parent_components and isinstance(self.parent_components['log_accordion'], widgets.Widget):
                log_widget = self.parent_components['log_accordion']
                
            if log_widget:
                log_container = widgets.VBox(
                    [log_widget],
                    layout=widgets.Layout(
                        margin='10px 0 0 0',
                        width='100%'
                    )
                )
                children.append(log_container)
                
            # Add info accordion if available
            if info_accordion and isinstance(info_accordion, widgets.Widget):
                info_container = widgets.VBox(
                    [info_accordion],
                    layout=widgets.Layout(
                        margin='10px 0 0 0',
                        width='100%'
                    )
                )
                children.append(info_container)

            main_container = widgets.VBox(
                children=children,
                layout=widgets.Layout(
                    width='100%',
                    padding='15px',
                    gap='10px',
                    border='1px solid #ddd',
                    border_radius='8px'
                )
            )
            
            # === 5. COMBINE ALL COMPONENTS ===
            self.ui_components = {
                **self.parent_components,
                **self.child_components,
                'child_container': child_container,
                'container': main_container,
                'logger_bridge': self._logger_bridge
            }
            
            self._logger.debug("✅ UI components created successfully")
            return self.ui_components
            
        except Exception as e:
            self._logger.error(f"❌ Failed to create UI components: {str(e)}", exc_info=True)
            raise
            
    def setup_handlers(self) -> None:
        """⚡ Setup event handlers untuk UI components.
        
        OVERRIDE CAREFULLY - ALWAYS CALL super().setup_handlers() FIRST!
        
        Base implementation:
        1. Setup basic handlers untuk parent components
        2. Connect logger bridge
        3. Setup error handling
        
        Child class bisa override untuk add custom handlers:
        ```python
        def setup_handlers(self):
            super().setup_handlers()  # WAJIB!
            # Add custom handlers here
            self.ui_components['my_button'].on_click(self._on_button_click)
        ```
        
        Note:
            - SELALU call super().setup_handlers() di awal override
            - Akses components via self.ui_components dictionary
            - Use self._logger_bridge untuk logging
        """
        self._logger.debug("Setting up base event handlers...")
        
        # Setup save/reset handlers jika ada
        if 'save_button' in self.ui_components:
            self.ui_components['save_button'].on_click(
                lambda b: self._on_save_click()
            )
            
        if 'reset_button' in self.ui_components:
            self.ui_components['reset_button'].on_click(
                lambda b: self._on_reset_click()
            )
            
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> widgets.Widget:
        """🚀 Initialize UI dan return root container.
        
        PUBLIC METHOD - Called untuk create dan display UI.
        
        Process:
        1. Setup output suppression
        2. Create UI components (parent + child)
        3. Setup event handlers
        4. Register di component registry
        5. Return root container
        
        Args:
            config: Optional config override (default: use self.config)
            
        Returns:
            Root container widget ready untuk display
            
        Raises:
            Exception: Jika initialization gagal
            
        Usage:
            ```python
            initializer = MyConfigInitializer(config)
            container = initializer.initialize()
            display(container)
            ```
        """
        if self._is_initialized:
            self._logger.warning("Already initialized, returning existing container")
            return self.get_container()
            
        try:
            self._logger.info(f"🚀 Initializing {self.component_id}...")
            
            # Use provided config or check shared config or default
            if config is not None:
                self.config = config
            else:
                # Try get from shared config first
                shared_config = self._get_shared_config()
                if shared_config:
                    self.config = shared_config
                    self._logger.info(f"📡 Loaded config from shared manager")
                # Else use existing self.config
                
            # Setup output suppression
            self._setup_output_suppression()
            
            # Create all UI components
            self._create_ui_components()
            
            # Setup event handlers
            self.setup_handlers()
            
            # Register component
            self._register_component()
            
            # Mark as initialized
            self._is_initialized = True
            
            # Restore output
            self._restore_output()
            
            self._logger.info(f"✅ {self.component_id} initialized successfully")
            
            return self.get_container()
            
        except Exception as e:
            self._logger.error(f"❌ Initialization failed: {str(e)}", exc_info=True)
            self._restore_output()
            
            # Return error container
            return create_error_response(
                error_message=str(e),
                error=e,
                title=f"Failed to initialize {self.title}"
            )
            
    def _create_ui_components(self) -> None:
        """🎨 Internal method untuk create UI components.
        
        INTERNAL - Jangan call langsung atau override!
        """
        # Create components via template method
        self.ui_components = self.create_ui_components(self.config)
        
        # Update parent component dengan UI
        if 'container' in self.ui_components:
            self.parent_component.content_area.children = (
                self.ui_components['container'],
            )
            
    def _setup_output_suppression(self) -> None:
        """⚡ Setup output suppression untuk clean UI.
        
        INTERNAL - Suppress noisy library outputs.
        """
        setup_aggressive_log_suppression()
        if self.ui_components:
            setup_stdout_suppression()
        self._logger.debug("Output suppression enabled")
        
    def _restore_output(self) -> None:
        """🔄 Restore output settings.
        
        INTERNAL - Restore normal output behavior.
        """
        restore_stdout()
        allow_tqdm_display()
        self._logger.debug("Output restored")
        
    def _register_component(self) -> None:
        """📋 Register component di registry.
        
        INTERNAL - Setup component tracking.
        """
        full_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
        
        component_registry.register_component(
            component_id=full_id,
            component={
                **self.ui_components,
                'initializer': self
            },
            parent_id=self.parent_id
        )
        
        self._logger.debug(f"Registered component: {full_id}")
        
    def get_container(self) -> widgets.Widget:
        """📦 Get root container widget.
        
        Returns:
            Root container widget
            
        Note:
            Safe to call multiple times, akan return same container.
        """
        if not self._is_initialized:
            raise RuntimeError(f"{self.component_id} not initialized. Call initialize() first.")
        return self.ui_components.get('container', self.parent_component.container)
        
    def update_status(self, message: str, status_type: str = "info") -> None:
        """📊 Update status panel dengan message.
        
        PUBLIC METHOD - Gunakan untuk update status dari handlers.
        
        Args:
            message: Status message untuk display
            status_type: Type ('success', 'info', 'warning', 'error')
            
        Example:
            ```python
            self.update_status("Processing...", "info")
            self.update_status("✅ Complete!", "success")
            self.update_status("❌ Failed", "error")
            ```
        """
        if 'status_panel' in self.ui_components:
            from smartcash.ui.components import update_status_panel
            update_status_panel(
                self.ui_components['status_panel'],
                message,
                status_type
            )
            
    def _on_save_click(self) -> None:
        """💾 Default save button handler dengan config sharing.
        
        INTERNAL - Override setup_handlers() untuk custom behavior.
        """
        try:
            self.update_status("💾 Saving configuration...", "info")
            
            # Extract current config
            current_config = self.handler.extract_config(self.ui_components)
            
            # Save config
            success = self.handler.save_config(self.ui_components)
            
            if success:
                self.update_status("✅ Configuration saved", "success")
                
                # Broadcast to other components
                self.broadcast_config_change(current_config)
                
            else:
                self.update_status("❌ Save failed", "error")
                
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}", "error")
            self._logger.error(f"Save failed: {str(e)}", exc_info=True)
            
    def _on_reset_click(self) -> None:
        """🔄 Default reset button handler.
        
        INTERNAL - Override setup_handlers() untuk custom behavior.
        """
        try:
            self.update_status("🔄 Resetting to defaults...", "info")
            self.handler.reset_ui(self.ui_components)
            self.update_status("✅ Reset to defaults", "success")
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}", "error")
            self._logger.error(f"Reset failed: {str(e)}", exc_info=True)
            
    def _setup_shared_config(self) -> None:
        """Initialize shared configuration manager and set up subscription.
        
        This method is called during initialization to set up the shared config
        manager and subscribe to configuration updates.
        """
        try:
            from smartcash.ui.config_cell.managers.shared_config_manager import get_shared_config_manager
            
            # Get shared config manager for this component's parent
            parent_module = self.parent_id.split('.')[0] if self.parent_id else 'global'
            self._shared_config_manager = get_shared_config_manager(parent_module)
            
            # Subscribe to config updates
            self._unsubscribe_func = self._shared_config_manager.subscribe(
                self.component_id,
                self._on_shared_config_update
            )
            
            self._logger.debug(f"✅ Initialized shared config for {self.component_id}")
            
        except Exception as e:
            self._logger.warning(f"⚠️ Failed to setup shared config: {e}", exc_info=True)
            self._shared_config_manager = None
            self._unsubscribe_func = None
            
    def _on_shared_config_update(self, config: Dict[str, Any]) -> None:
        """Handle updates from shared configuration.
        
        Args:
            config: New configuration data
        """
        if not config:
            return
            
        try:
            self._logger.info("🔄 Updating from shared configuration...")
            
            # Update local config
            self.config.update(config)
            
            # Update UI components if initialized
            if self._is_initialized and hasattr(self.handler, 'update_ui_from_config'):
                self.handler.update_ui_from_config(self.ui_components, config)
                self.update_status("✅ Configuration updated from shared source", "success")
                
        except Exception as e:
            self._logger.error(f"❌ Failed to apply shared config: {e}", exc_info=True)
            self.update_status(f"❌ Failed to apply shared config: {e}", "error")
    
    def _get_shared_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration from shared manager if available.
        
        Returns:
            Shared configuration or None
        """
        if self._shared_config_manager:
            try:
                return self._shared_config_manager.get_config(self.component_id)
            except Exception as e:
                self._logger.debug(f"Could not get shared config: {e}")
        return None
    
    def refresh_from_shared_config(self) -> None:
        """PUBLIC METHOD - Manually refresh configuration from shared storage.
        
        Use this to sync with changes from other cells.
        """
        if not self._is_initialized:
            self._logger.warning("Cannot refresh - component not initialized")
            return
            
        shared_config = self._get_shared_config()
        if shared_config:
            self._on_shared_config_update(shared_config)
            self._logger.info("🔄 Refreshed from shared configuration")
        else:
            self._logger.info("No shared configuration found")
    
    def cleanup(self) -> None:
        """🧹 Cleanup resources dan unregister component.
        
        PUBLIC METHOD - Call saat component tidak digunakan lagi.
        
        Cleanup process:
        1. Unsubscribe from shared config
        2. Unregister dari component registry
        3. Clear semua component dictionaries
        4. Reset state
        
        Note:
            Ini TIDAK menghapus widgets dari display.
            Hanya cleanup internal state dan references.
        """
        try:
            # Unsubscribe from shared config
            if self._unsubscribe_func:
                self._unsubscribe_func()
                self._unsubscribe_func = None
            
            # Unregister from component registry
            full_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
            component_registry.unregister_component(full_id)
            
            # Clear components
            self.ui_components.clear()
            self.parent_components.clear()
            self.child_components.clear()
            
            # Reset state
            self._is_initialized = False
            self._handler = None
            self._shared_config_manager = None
            
            self._logger.debug(f"✅ Cleanup completed for {self.component_id}")
            
        except Exception as e:
            self._logger.error(f"❌ Cleanup error: {str(e)}", exc_info=True)