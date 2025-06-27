"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: Implementation of dataset split configuration UI following ConfigCellInitializer pattern

PATTERN COMPLIANCE:
==================
Modul ini mengikuti pattern ConfigCellInitializer dengan:
1. HANYA membuat child components (form untuk split configuration)
2. TIDAK membuat parent components (header, status, log, info) - sudah dari parent
3. Menggunakan logger bridge dari parent untuk semua logging
4. Update status melalui parent's status panel
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display
import logging

# Initializers - Import parent class yang menyediakan infrastructure
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer

# Local components - HANYA form components, TIDAK ada parent components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

logger = logging.getLogger(__name__)

# Constants
MODULE_NAME = "split_config"

class SplitConfigInitializer(ConfigCellInitializer):
    """🎯 Dataset split configuration UI implementation.
    
    WHAT THIS CLASS DOES:
    ====================
    1. Implements create_handler() - membuat SplitConfigHandler
    2. Implements create_child_components() - HANYA form untuk split config
    3. Implements get_info_content() - info spesifik tentang dataset split
    4. Overrides setup_handlers() - event handlers untuk form components
    
    WHAT THIS CLASS DOES NOT DO:
    ============================
    1. ❌ TIDAK membuat header (sudah dari parent)
    2. ❌ TIDAK membuat status panel (sudah dari parent)
    3. ❌ TIDAK membuat log accordion (sudah dari parent)
    4. ❌ TIDAK membuat info accordion (sudah dari parent)
    5. ❌ TIDAK manage logger bridge (sudah dari parent)
    
    PARENT PROVIDES:
    ===============
    - Header dengan title "📊 Dataset Split Configuration"
    - Status panel untuk show operation status
    - Log accordion yang connected ke logger bridge
    - Info accordion dengan content dari get_info_content()
    - Logger bridge untuk centralized logging
    - Component registry untuk tracking
    
    USAGE:
    ======
    ```python
    # Create dan display split configuration UI
    from smartcash.ui.dataset.split import create_split_config_cell
    create_split_config_cell()
    
    # Atau dengan custom config
    create_split_config_cell(config={'split_ratios': {'train': 0.8, ...}})
    ```
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: str = MODULE_NAME,
        title: str = "Dataset Split Configuration",  # Emoji is handled by icon parameter
        description: str = "Konfigurasi pembagian dataset untuk training, validation, dan testing",
        **kwargs
    ):
        """Initialize split config UI.
        
        Args:
            config: Configuration dictionary untuk split settings
            parent_id: Parent component ID untuk hierarchical organization
            component_id: Unique identifier untuk component ini
            title: Title untuk header (dibuat oleh parent)
            description: Description untuk header (dibuat oleh parent)
            **kwargs: Additional arguments untuk parent class
            
        Note:
            Parent class (ConfigCellInitializer) akan otomatis:
            - Create header dengan title dan description
            - Create status panel
            - Create log accordion connected ke logger bridge
            - Create info accordion dengan content dari get_info_content()
            - Setup logger bridge untuk centralized logging
        """
        # Pass semua arguments ke parent class
        super().__init__(
            config=config or {},
            parent_id=parent_id,
            component_id=component_id,
            title=title,
            description=description,
            icon="📊",  # Single emoji for the header
            **kwargs
        )
        
    def create_handler(self) -> SplitConfigHandler:
        """🔧 Create split configuration handler.
        
        REQUIRED IMPLEMENTATION dari parent abstract method.
        
        Returns:
            SplitConfigHandler instance dengan config dari constructor
            
        Note:
            Parent class akan automatically inject logger bridge ke handler
            setelah creation (jika handler support set_logger_bridge method).
        """
        return SplitConfigHandler(self.config)
    
    def create_child_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎨 Create HANYA form components untuk split configuration.
        
        REQUIRED IMPLEMENTATION dari parent abstract method.
        
        CRITICAL: Method ini HANYA membuat form/UI components spesifik untuk
        split configuration. JANGAN membuat header, status panel, log accordion,
        atau info accordion karena sudah dibuat oleh parent class!
        
        Args:
            config: Current configuration untuk initialize form values
        
        Returns:
            Dictionary berisi HANYA form components:
            - Sliders untuk train/valid/test ratios
            - Checkboxes untuk options (stratified, backup, validation)
            - Input fields untuk paths dan settings
            - Save/reset buttons
            - Form container yang mengelompokkan semua components
            
        Components Created:
            - train_slider: Slider untuk training ratio
            - valid_slider: Slider untuk validation ratio  
            - test_slider: Slider untuk test ratio
            - total_label: Label showing total ratio
            - stratified_checkbox: Option untuk stratified split
            - random_seed: Input untuk random seed
            - backup_checkbox: Option untuk backup dataset
            - validation_enabled: Option untuk enable validation
            - Etc...
        """
        try:
            # Create form components menggunakan local modules
            form_components = create_split_form(config)
            
            # Add logger bridge reference untuk use di event handlers
            form_components['logger_bridge'] = self._logger_bridge
            
            # Create layout yang organize form components
            layout_components = create_split_layout(form_components)
            
            # Return HANYA child components, TIDAK ada parent components!
            return {
                # Form container sebagai main child component
                'container': layout_components['container'],
                
                # Individual form components untuk programmatic access
                **form_components,
                
                # Layout reference
                'layout': layout_components
            }
            
        except Exception as e:
            self._logger.error(f"❌ Failed to create child components: {str(e)}", exc_info=True)
            raise
            
    def get_info_content(self) -> widgets.Widget:
        """📚 Provide content untuk info accordion.
        
        OPTIONAL IMPLEMENTATION dari parent method.
        
        Returns:
            HTML widget dengan informasi tentang dataset split
            
        Note:
            Parent class akan create info accordion dan insert content ini.
            Kita HANYA provide content, TIDAK create accordion sendiri!
        """
        return widgets.HTML("""
            <div style='padding: 10px; font-size: 14px; line-height: 1.6;'>
                <h4 style='margin-top: 0;'>📊 Tentang Dataset Split</h4>
                <p>Modul ini membagi dataset menjadi 3 subset untuk training model:</p>
                <ul style='margin: 10px 0;'>
                    <li><b>Training Set (70%):</b> Data untuk melatih model</li>
                    <li><b>Validation Set (15%):</b> Data untuk tuning hyperparameter dan monitoring overfitting</li>
                    <li><b>Test Set (15%):</b> Data untuk evaluasi final performa model</li>
                </ul>
                
                <h4>🔧 Opsi Konfigurasi</h4>
                <ul style='margin: 10px 0;'>
                    <li><b>Stratified Split:</b> Mempertahankan distribusi kelas di setiap subset</li>
                    <li><b>Random Seed:</b> Nilai untuk reproducible random split</li>
                    <li><b>Backup Dataset:</b> Simpan copy dataset asli sebelum split</li>
                    <li><b>Validation:</b> Periksa integritas data dan fix issues</li>
                </ul>
                
                <h4>💡 Tips</h4>
                <ul style='margin: 10px 0;'>
                    <li>Gunakan stratified split untuk dataset dengan imbalanced classes</li>
                    <li>Set random seed yang sama untuk reproducible experiments</li>
                    <li>Enable validation untuk detect corrupted images</li>
                    <li>Backup dataset sebelum split untuk safety</li>
                </ul>
                
                <p style='color: #666; font-size: 0.9em; margin-top: 15px;'>
                    <i>Note: Total ratio harus = 1.0 (100%). Slider akan auto-normalize jika total ≠ 1.0</i>
                </p>
            </div>
        """)
        
    def setup_handlers(self) -> None:
        """⚡ Setup event handlers untuk form components.
        
        OPTIONAL OVERRIDE dari parent method.
        
        CRITICAL: HARUS call super().setup_handlers() FIRST!
        
        Handlers yang di-setup:
        1. Slider handlers untuk auto-normalize ratios
        2. Checkbox handlers untuk enable/disable dependent fields
        3. Save/reset button handlers (sudah dari parent, kita extend)
        4. Custom validation dan feedback
        """
        # WAJIB: Call parent implementation first
        super().setup_handlers()
        
        try:
            # Import local event handler setup
            from smartcash.ui.dataset.split.handlers.event_handlers import setup_event_handlers
            
            # Setup handlers dengan reference ke self untuk access parent methods
            setup_event_handlers(self, self.ui_components)
            
            # Update status untuk indicate ready
            self.update_status("Split configuration ready", "success")
            
            self._logger.debug("Custom event handlers setup completed")
            
        except Exception as e:
            self._logger.error(f"❌ Failed to setup custom handlers: {str(e)}", exc_info=True)
            self.update_status(f"❌ Error: {str(e)}", "error")
            # Don't raise - UI masih bisa digunakan tanpa custom handlers


def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """🎯 Create dan display dataset split configuration UI.
    
    PUBLIC API FUNCTION - Main entry point untuk users.
    
    Function ini:
    1. Create instance SplitConfigInitializer
    2. Initialize UI (parent + child components)
    3. Display di Jupyter notebook
    
    Args:
        config: Optional configuration dictionary dengan keys:
            - split_ratios: Dict dengan train/valid/test ratios
            - stratified_split: Boolean untuk stratified sampling
            - random_seed: Integer untuk reproducibility
            - backup_enabled: Boolean untuk backup dataset
            - validation_enabled: Boolean untuk data validation
            - paths: Dict dengan directory paths
        **kwargs: Additional arguments untuk initializer
        
    Example:
        ```python
        # Default configuration
        create_split_config_cell()
        
        # Custom configuration
        create_split_config_cell(config={
            'split_ratios': {
                'train': 0.8,
                'valid': 0.1,
                'test': 0.1
            },
            'stratified_split': True,
            'random_seed': 42
        })
        ```
        
    Note:
        UI akan include:
        - Header dengan title dan description
        - Status panel untuk operation feedback
        - Form untuk split configuration
        - Log accordion untuk messages
        - Info accordion dengan documentation
    """
    try:
        # Create initializer instance
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Initialize UI - ini create semua components (parent + child)
        container = initializer.initialize()
        
        # Display di notebook
        display(container)
        
        logger.info("✅ Split config UI displayed successfully")
        
    except Exception as e:
        error_msg = f"❌ Failed to create split config UI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Display error menggunakan parent's error handler
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        error_widget = create_error_response(
            error_message=error_msg,
            error=e,
            title="🚨 Error in Dataset Split Configuration",
            include_traceback=True
        )
        display(error_widget)


def get_split_config_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """📦 Create split config UI dan return components untuk programmatic access.
    
    PUBLIC API FUNCTION - Untuk programmatic access ke components.
    
    Same as create_split_config_cell() tapi return components dictionary
    instead of displaying UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments untuk initializer
               
    Returns:
        Dictionary berisi semua UI components:
        {
            # References
            'initializer': SplitConfigInitializer instance,
            'handler': SplitConfigHandler instance,
            'logger_bridge': UILoggerBridge instance,
            
            # Containers
            'container': Root container widget,
            
            # Parent components (dari ConfigCellInitializer)
            'header': Header widget,
            'status_panel': Status panel widget,
            'log_accordion': Log accordion widget,
            'info_accordion': Info accordion widget,
            
            # Child components (dari create_child_components)
            'train_slider': Training ratio slider,
            'valid_slider': Validation ratio slider,
            'test_slider': Test ratio slider,
            ... (semua form components)
        }
        
    Example:
        ```python
        # Get components untuk programmatic manipulation
        components = get_split_config_components()
        
        # Access specific components
        components['train_slider'].value = 0.8
        components['status_panel'].value = "Custom status"
        
        # Display manually
        display(components['container'])
        ```
        
    Raises:
        Exception: Jika UI creation gagal
    """
    try:
        # Create initializer
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Initialize tanpa display
        initializer.initialize()
        
        # Return semua components untuk programmatic access
        return {
            # Core references
            'initializer': initializer,
            'handler': initializer.handler,
            'logger_bridge': initializer._logger_bridge,
            
            # Main container
            'container': initializer.get_container(),
            
            # All UI components (parent + child)
            **initializer.ui_components
        }
        
    except Exception as e:
        error_msg = f"❌ Failed to create split config components: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise