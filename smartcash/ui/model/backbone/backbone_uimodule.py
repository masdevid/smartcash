"""
File: smartcash/ui/model/backbone/backbone_uimodule.py
Main UIModule implementation for backbone module following new UIModule pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule, ModuleStatus
from smartcash.ui.logger import get_module_logger
from .configs.backbone_config_handler import BackboneConfigHandler
from .configs.backbone_defaults import get_default_backbone_config
from .operations.backbone_operation_manager import BackboneOperationManager
from datetime import datetime


class BackboneUIModule(UIModule):
    """
    Implementasi UIModule untuk konfigurasi backbone model.
    
    Fitur:
    - 🧬 Pemilihan dan konfigurasi backbone model
    - 🏗️ Integrasi pipeline pelatihan dini
    - 📊 Tampilan ringkasan model terkini  
    - 🔧 Manajemen dan validasi konfigurasi
    - 🎯 Integrasi dengan backend model builder
    - 📋 Panel ringkasan konfigurasi di summary_container
    - 🔄 Pelacakan progres untuk semua operasi
    """
    
    def __init__(self):
        """Initialize backbone UI module."""
        super().__init__(
            module_name='backbone',
            parent_module='model'
        )
        
        self.logger = get_module_logger("smartcash.ui.model.backbone")
        
        # Initialize components
        self._config_handler = None
        self._operation_manager = None
        self._ui_components = None
        
        self.logger.debug("✅ BackboneUIModule diinisialisasi")
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler."""
        try:
            self._config_handler = BackboneConfigHandler()
            
            # Set initial configuration
            if config:
                merged_config = self._config_handler.merge_config(
                    get_default_backbone_config(), config
                )
            else:
                merged_config = get_default_backbone_config()
            
            # Update config using keyword arguments
            self.update_config(**merged_config)
            self.logger.debug("✅ Handler konfigurasi diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"Gagal menginisialisasi handler konfigurasi: {e}")
            raise
    
    def _initialize_operation_manager(self) -> None:
        """Initialize operation manager."""
        try:
            if not self._ui_components:
                raise RuntimeError("Komponen UI harus dibuat terlebih dahulu sebelum operation manager")
            
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                raise RuntimeError("Container operasi tidak ditemukan dalam komponen UI")
            
            self._operation_manager = BackboneOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            self._operation_manager.initialize()
            
            # Setup UI logging bridge to capture backend service logs
            self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress tracker display
            self._initialize_progress_display()
            
            self.logger.debug("✅ Manajer operasi diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"Gagal menginisialisasi manajer operasi: {e}")
            raise
    
    def _setup_button_handlers(self) -> None:
        """Setup button click handlers for UI operations."""
        try:
            if not self._ui_components or not self._operation_manager:
                self.logger.warning("Tidak dapat menyiapkan tombol - komponen belum diinisialisasi")
                return
            
            action_container = self._ui_components.get('containers', {}).get('action')
            if not action_container:
                self.logger.warning("Container aksi tidak ditemukan untuk pengaturan tombol")
                return
            
            # Setup button click handlers with synchronous wrappers
            buttons = action_container.get('buttons', {})
            
            if 'validate' in buttons:
                buttons['validate'].on_click(self._handle_validate)
                
            if 'build' in buttons:
                buttons['build'].on_click(self._handle_build)
            
            # Setup save/reset button handlers
            if 'save' in buttons:
                buttons['save'].on_click(self._handle_save_config)
                
            if 'reset' in buttons:
                buttons['reset'].on_click(self._handle_reset_config)
            
            self.logger.debug("✅ Pengaturan penangan tombol selesai")
            
        except Exception as e:
            self.logger.error(f"Gagal menyiapkan penangan tombol: {e}")
    
    
    def _handle_validate(self, button) -> None:
        """Synchronous handler for validate button."""
        try:
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            # Disable only validate button during operation
            self._disable_validate_button()
            
            # Clear UI state
            self._clear_ui_state()
            
            # Check prerequisites before validation
            prereq_check = self._check_data_prerequisites_sync()
            if not prereq_check['success']:
                self._operation_manager.log(f"⚠️ Pemeriksaan prasyarat: {prereq_check['message']}", 'warning')
                # Continue with validation even if data is missing - just warn user
            
            # Get current configuration from UI
            current_config = self._get_current_ui_config()
            
            # Execute validation
            result = self._operation_manager.execute_validate(current_config)
            
            # Update summary with validation results if successful
            if result.get('success'):
                self._update_summary_display_sync()
                
        except Exception as e:
            self.logger.error(f"Kesalahan pada penangan validasi: {e}")
            if self._operation_manager:
                self._operation_manager.log(f"❌ Kesalahan validasi: {e}", 'error')
        finally:
            # Re-enable validate button
            self._enable_validate_button()
    
    def _handle_build(self, button) -> None:
        """Synchronous handler for build button."""
        try:
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            # Disable only build button during operation
            self._disable_build_button()
            
            # Clear UI state
            self._clear_ui_state()
            
            # Check prerequisites before build
            prereq_check = self._check_data_prerequisites_sync()
            if not prereq_check['success']:
                self._operation_manager.log(f"❌ Prerequisites missing: {prereq_check['message']}", 'error')
                return
            
            # Get current configuration from UI
            current_config = self._get_current_ui_config()
            
            # Execute build
            result = self._operation_manager.execute_build(current_config)
            
            # Update summary with build results if successful
            if result.get('success'):
                self._update_summary_display_sync()
                
        except Exception as e:
            self.logger.error(f"Build handler error: {e}")
            if self._operation_manager:
                self._operation_manager.log(f"❌ Build error: {e}", 'error')
        finally:
            # Re-enable build button
            self._enable_build_button()
    
    def _clear_ui_state(self) -> None:
        """Clear UI state before operations."""
        try:
            if self._operation_manager:
                self._operation_manager.clear_logs()
                self._operation_manager.update_progress(0, "Initializing...")
                
        except Exception as e:
            self.logger.error(f"Error clearing UI state: {e}")
    
    def _get_current_ui_config(self) -> Dict[str, Any]:
        """Get current configuration from UI widgets."""
        try:
            current_config = self.get_config().copy()
            
            # Get widget values if available
            widgets = self._ui_components.get('widgets', {})
            if widgets:
                backbone_config = current_config.setdefault('backbone', {})
                
                # Update backbone config from widgets
                if 'backbone_dropdown' in widgets:
                    backbone_config['model_type'] = widgets['backbone_dropdown'].value
                if 'pretrained_checkbox' in widgets:
                    backbone_config['pretrained'] = widgets['pretrained_checkbox'].value
                if 'feature_opt_checkbox' in widgets:
                    backbone_config['feature_optimization'] = widgets['feature_opt_checkbox'].value
                if 'mixed_precision_checkbox' in widgets:
                    backbone_config['mixed_precision'] = widgets['mixed_precision_checkbox'].value
                if 'input_size_slider' in widgets:
                    backbone_config['input_size'] = widgets['input_size_slider'].value
                if 'num_classes_input' in widgets:
                    backbone_config['num_classes'] = widgets['num_classes_input'].value
            
            return current_config
            
        except Exception as e:
            self.logger.error(f"Error getting UI config: {e}")
            return self.get_config().copy()
    
    def _update_summary_display_sync(self) -> None:
        """Update summary container with current model info (synchronous)."""
        try:
            if not self._operation_manager:
                return
                
            # Get current model summary
            model_info = self._operation_manager.get_current_model_summary()
            
            # Update summary container
            summary_container = self._ui_components.get('summary_container')
            if summary_container and model_info:
                from .components.backbone_ui import update_model_summary
                update_model_summary(summary_container, model_info)
                
        except Exception as e:
            self.logger.error(f"Error updating summary display: {e}")
    
    def _handle_save_config(self, button) -> None:
        """Handle save config button click."""
        try:
            if not self._operation_manager:
                self.logger.warning("Operation manager not available for logging")
                return
            
            # Get current UI configuration
            current_config = self._get_current_ui_config()
            
            # Update the module configuration using individual keys
            for key, value in current_config.items():
                self._config[key] = value
            
            # Log success message
            if self._operation_manager:
                self._operation_manager.log("💾 Configuration saved successfully", 'success')
            
            # Status updates are now handled via logging
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {e}"
            self.logger.error(error_msg)
            if self._operation_manager:
                self._operation_manager.log(f"❌ {error_msg}", 'error')
            self._update_header_status("Save failed", "error")
    
    def _handle_reset_config(self, button) -> None:
        """Handle reset config button click."""
        try:
            if not self._operation_manager:
                self.logger.warning("Operation manager not available for logging")
                return
            
            # Reset to default configuration
            from .configs.backbone_defaults import get_default_backbone_config
            default_config = get_default_backbone_config()
            
            # Update module configuration using individual keys
            for key, value in default_config.items():
                self._config[key] = value
            
            # Update UI widgets with default values
            self._update_ui_widgets_from_config(default_config)
            
            # Log success message
            if self._operation_manager:
                self._operation_manager.log("🔄 Configuration reset to defaults", 'info')
            
            # Status updates are now handled via logging
            
            self.logger.info("Configuration reset to defaults")
            
        except Exception as e:
            error_msg = f"Failed to reset configuration: {e}"
            self.logger.error(error_msg)
            if self._operation_manager:
                self._operation_manager.log(f"❌ {error_msg}", 'error')
            self._update_header_status("Reset failed", "error")
    
    def _check_data_prerequisites_sync(self) -> Dict[str, Any]:
        """Check if all required data is available (synchronous)."""
        try:
            missing = []
            warnings = []
            
            # Check pretrained models
            pretrained_check = self._check_pretrained_models()
            if not pretrained_check['available']:
                warnings.append("Pretrained models not found")
            
            # Check raw data using preprocessor API
            raw_data_check = self._check_raw_data_sync()
            if not raw_data_check['available']:
                missing.append("Raw data not found")
            
            # Check preprocessed data
            preprocessed_check = self._check_preprocessed_data_sync()
            if not preprocessed_check['available']:
                warnings.append("Preprocessed data not found")
            
            # Check augmented data
            augmented_check = self._check_augmented_data_sync()
            if not augmented_check['available']:
                warnings.append("Augmented data not found")
            
            if missing:
                return {
                    'success': False,
                    'message': f"Missing required data: {', '.join(missing)}",
                    'missing': missing,
                    'warnings': warnings
                }
            elif warnings:
                return {
                    'success': True,
                    'message': f"Some optional data missing: {', '.join(warnings)}",
                    'missing': [],
                    'warnings': warnings
                }
            else:
                return {
                    'success': True,
                    'message': "All data prerequisites available",
                    'missing': [],
                    'warnings': []
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error checking prerequisites: {e}",
                'missing': [],
                'warnings': []
            }
    
    def _check_pretrained_models(self) -> Dict[str, Any]:
        """Check if pretrained models are available."""
        try:
            from pathlib import Path
            
            pretrained_dir = Path('/data/pretrained')
            backbone_config = self.get_config().get('backbone', {})
            backbone_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Check for backbone-specific pretrained models
            if backbone_type == 'efficientnet_b4':
                efficientnet_dir = pretrained_dir / 'efficientnet'
                if efficientnet_dir.exists():
                    model_files = list(efficientnet_dir.glob('*.pth')) + list(efficientnet_dir.glob('*.pt'))
                    if model_files:
                        return {'available': True, 'path': str(efficientnet_dir), 'files': len(model_files)}
            elif backbone_type == 'cspdarknet':
                yolov5_dir = pretrained_dir / 'yolov5'
                if yolov5_dir.exists():
                    model_files = list(yolov5_dir.glob('*.pt'))
                    if model_files:
                        return {'available': True, 'path': str(yolov5_dir), 'files': len(model_files)}
            
            return {'available': False, 'path': str(pretrained_dir), 'files': 0}
            
        except Exception as e:
            self.logger.error(f"Error checking pretrained models: {e}")
            return {'available': False, 'path': '', 'files': 0}
    
    def _check_raw_data_sync(self) -> Dict[str, Any]:
        """Check raw data using preprocessor API (synchronous)."""
        try:
            # Use preprocessor API to check raw data
            from smartcash.dataset.preprocessor.api import get_preprocessing_status
            
            status = get_preprocessing_status()
            if status.get('success') and status.get('service_ready'):
                file_stats = status.get('file_statistics', {})
                
                # Check if any split has raw images
                total_raw = sum(
                    split_data.get('raw_images', 0) 
                    for split_data in file_stats.values()
                )
                
                return {
                    'available': total_raw > 0,
                    'total_files': total_raw,
                    'details': file_stats
                }
            
            return {'available': False, 'total_files': 0, 'details': {}}
            
        except Exception as e:
            self.logger.error(f"Error checking raw data: {e}")
            return {'available': False, 'total_files': 0, 'details': {}}
    
    def _check_preprocessed_data_sync(self) -> Dict[str, Any]:
        """Check preprocessed data using preprocessor API (synchronous)."""
        try:
            # Use preprocessor API to check preprocessed data
            from smartcash.dataset.preprocessor.api import get_preprocessing_status
            
            status = get_preprocessing_status()
            if status.get('success') and status.get('service_ready'):
                file_stats = status.get('file_statistics', {})
                
                # Check if any split has preprocessed files
                total_preprocessed = sum(
                    split_data.get('preprocessed_files', 0) 
                    for split_data in file_stats.values()
                )
                
                return {
                    'available': total_preprocessed > 0,
                    'total_files': total_preprocessed,
                    'details': file_stats
                }
            
            return {'available': False, 'total_files': 0, 'details': {}}
            
        except Exception as e:
            self.logger.error(f"Error checking preprocessed data: {e}")
            return {'available': False, 'total_files': 0, 'details': {}}
    
    def _check_augmented_data_sync(self) -> Dict[str, Any]:
        """Check augmented data using augmentor API (synchronous)."""
        try:
            # Use augmentor API to check augmented data
            from smartcash.dataset.augmentor import get_augmentation_status
            
            status = get_augmentation_status({})
            if status.get('service_ready'):
                # Check for augmented files across splits
                total_augmented = 0
                for split in ['train', 'valid', 'test']:
                    total_augmented += status.get(f'{split}_augmented', 0)
                
                return {
                    'available': total_augmented > 0,
                    'total_files': total_augmented,
                    'details': status
                }
            
            return {'available': False, 'total_files': 0, 'details': {}}
            
        except Exception as e:
            self.logger.error(f"Error checking augmented data: {e}")
            return {'available': False, 'total_files': 0, 'details': {}}
    
    def _disable_validate_button(self) -> None:
        """Disable only validate button during validation."""
        try:
            action_container = self._ui_components.get('containers', {}).get('action')
            if action_container:
                buttons = action_container.get('buttons', {})
                
                if 'validate' in buttons:
                    buttons['validate'].disabled = True
                    buttons['validate'].description = '⏳ Validating...'
                    
        except Exception as e:
            self.logger.error(f"Error disabling validate button: {e}")
    
    def _enable_validate_button(self) -> None:
        """Re-enable validate button after validation."""
        try:
            action_container = self._ui_components.get('containers', {}).get('action')
            if action_container:
                buttons = action_container.get('buttons', {})
                
                if 'validate' in buttons:
                    buttons['validate'].disabled = False
                    buttons['validate'].description = '🔍 Validate'
                    
        except Exception as e:
            self.logger.error(f"Error enabling validate button: {e}")
    
    def _disable_build_button(self) -> None:
        """Disable only build button during build."""
        try:
            action_container = self._ui_components.get('containers', {}).get('action')
            if action_container:
                buttons = action_container.get('buttons', {})
                
                if 'build' in buttons:
                    buttons['build'].disabled = True
                    buttons['build'].description = '⏳ Building...'
                    
        except Exception as e:
            self.logger.error(f"Error disabling build button: {e}")
    
    def _enable_build_button(self) -> None:
        """Re-enable build button after build."""
        try:
            action_container = self._ui_components.get('containers', {}).get('action')
            if action_container:
                buttons = action_container.get('buttons', {})
                
                if 'build' in buttons:
                    buttons['build'].disabled = False
                    buttons['build'].description = '🏗️ Build Model'
                    
        except Exception as e:
            self.logger.error(f"Error enabling build button: {e}")
    
    # Header status panel functionality has been removed
    # Use self.logger or self._operation_manager.log for status updates instead
    
    def _update_ui_widgets_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI widgets with values from config."""
        try:
            widgets = self._ui_components.get('widgets', {})
            backbone_config = config.get('backbone', {})
            
            if 'backbone_dropdown' in widgets:
                widgets['backbone_dropdown'].value = backbone_config.get('model_type', 'efficientnet_b4')
            if 'pretrained_checkbox' in widgets:
                widgets['pretrained_checkbox'].value = backbone_config.get('pretrained', True)
            if 'feature_opt_checkbox' in widgets:
                widgets['feature_opt_checkbox'].value = backbone_config.get('feature_optimization', True)
            if 'mixed_precision_checkbox' in widgets:
                widgets['mixed_precision_checkbox'].value = backbone_config.get('mixed_precision', True)
            if 'input_size_slider' in widgets:
                widgets['input_size_slider'].value = backbone_config.get('input_size', 640)
            if 'num_classes_input' in widgets:
                widgets['num_classes_input'].value = backbone_config.get('num_classes', 7)
                
        except Exception as e:
            self.logger.error(f"Error updating UI widgets: {e}")
            # Also log to operation container if available
            if self._operation_manager:
                self._operation_manager.log(f"⚠️ Warning: Could not update UI widgets - {e}", 'warning')
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """Setup UI logging bridge to capture backend service logs."""
        try:
            import logging
            from smartcash.ui.core.logging.ui_logging_manager import setup_ui_logging
            
            # Get log message function from operation container
            log_message_func = None
            if isinstance(operation_container, dict) and 'log_message' in operation_container:
                log_message_func = operation_container['log_message']
            elif hasattr(operation_container, 'log_message'):
                log_message_func = operation_container.log_message
            
            if not log_message_func:
                self.logger.warning("⚠️ Could not setup UI logging bridge - log_message function not found")
                return
            
            # Setup basic UI logging for the module
            setup_ui_logging(
                module_name='model.backbone',
                log_message_func=log_message_func
            )
            
            # Create custom handler for backend services
            class BackendUILogHandler(logging.Handler):
                """Custom handler to route backend service logs to UI."""
                
                def __init__(self, log_func: callable):
                    super().__init__()
                    self.log_func = log_func
                    self.setLevel(logging.INFO)
                    formatter = logging.Formatter('%(name)s: %(message)s')
                    self.setFormatter(formatter)
                
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        level = 'debug' if record.levelno == logging.DEBUG else \
                               'info' if record.levelno == logging.INFO else \
                               'warning' if record.levelno == logging.WARNING else \
                               'error'
                        self.log_func(msg, level)
                    except Exception:
                        pass  # Silently handle logging errors
            
            # Create handler for backend services
            backend_handler = BackendUILogHandler(log_message_func)
            
            # Configure backend service loggers
            backend_namespaces = [
                'smartcash.model',
                'smartcash.dataset.preprocessor', 
                'smartcash.dataset.augmentor',
                'smartcash.common'
            ]
            
            for namespace in backend_namespaces:
                logger = logging.getLogger(namespace)
                
                # Remove existing console handlers to prevent duplicate output
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        import sys
                        if hasattr(handler, 'stream') and handler.stream in (sys.stdout, sys.stderr):
                            logger.removeHandler(handler)
                
                # Add UI handler
                logger.addHandler(backend_handler)
                logger.setLevel(logging.INFO)
            
            self.logger.debug("✅ UI logging bridge setup completed for backend services")
                
        except ImportError as e:
            self.logger.warning(f"⚠️ UI logging manager not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to setup UI logging bridge: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress tracker display to show by default."""
        try:
            if not self._operation_manager:
                return
            
            # Initialize progress tracker with default state
            self._operation_manager.update_progress(0, "Ready - No operation running")
            
            # Log initial status to show the operation container is working
            self._operation_manager.log("🧬 Backbone module ready", 'info')
            self._operation_manager.log("📋 Progress tracker initialized", 'debug')
            
            self.logger.debug("✅ Progress display initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing progress display: {e}")
    
    def _create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components."""
        try:
            from .components.backbone_ui import create_backbone_ui
            
            self.logger.debug("Creating backbone UI components...")
            ui_components = create_backbone_ui(config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize the backbone module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
        """
        try:
            self.logger.info("🧬 Initializing backbone models module")
            
            # Initialize configuration handler
            self._initialize_config_handler(config)
            
            # Create UI components
            self._ui_components = self._create_ui_components(self.get_config())
            
            # Initialize operation manager
            self._initialize_operation_manager()
            
            # Setup button click handlers
            self._setup_button_handlers()
            
            # Register shared methods for cross-module integration
            self._register_shared_methods()
            
            # Set status to READY to indicate successful initialization
            self._status = ModuleStatus.READY
            self._initialized_at = datetime.now()
            self.logger.info("✅ Backbone models module initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backbone module: {e}")
            raise RuntimeError("Failed to create UI components")
    
    def _register_shared_methods(self) -> None:
        """Register shared methods for cross-module integration."""
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Register backbone operations
            SharedMethodRegistry.register_method(
                'backbone.execute_validate',
                self.execute_validate,
                description='Validate backbone configuration'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.execute_build', 
                self.execute_build,
                description='Build backbone model'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.get_model_summary',
                self.get_model_summary,
                description='Get current model summary'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.get_config',
                self.get_config,
                description='Get backbone configuration'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.update_config',
                self.update_config,
                description='Update backbone configuration'
            )
            
            self.logger.debug("✅ Shared methods registered")
            
        except Exception as e:
            self.logger.warning(f"Failed to register shared methods: {e}")
    
    # ==================== OPERATION METHODS ====================
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Validation result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_validate(config)
            
        except Exception as e:
            error_msg = f"Validation execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_build(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone build operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Build result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_build(config)
            
        except Exception as e:
            error_msg = f"Build execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get current model summary.
        
        Returns:
            Current model summary dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.get_current_model_summary()
            
        except Exception as e:
            error_msg = f"Get summary failed: {e}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    # ==================== STATUS AND INFO METHODS ====================
    
    def get_backbone_status(self) -> Dict[str, Any]:
        """
        Get current backbone module status.
        
        Returns:
            Status information dictionary
        """
        try:
            base_status = {
                'initialized': self.is_initialized(),
                'module_name': self.module_name,
                'parent_module': self.parent_module,
                'config_available': self._config_handler is not None,
                'operations_available': self._operation_manager is not None
            }
            
            if self._operation_manager:
                operation_status = self._operation_manager.get_status()
                base_status.update(operation_status)
            
            return base_status
            
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return {'initialized': False, 'error': str(e)}
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            UI components dictionary
        """
        return self._ui_components or {}
        
    def is_initialized(self) -> bool:
        """
        Check if the module is initialized.
        
        Returns:
            bool: True if the module is initialized and ready
        """
        return self._status == ModuleStatus.READY
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration.
        
        Returns:
            Save operation result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Sync config from UI if available
            if self._ui_components:
                ui_config = self._config_handler.sync_from_ui(self._ui_components)
                if ui_config:
                    self.update_config(ui_config)
            
            # Save configuration (implementation depends on storage strategy)
            current_config = self.get_config()
            
            self.logger.info("📋 Configuration saved successfully")
            return {
                'success': True,
                'message': 'Configuration saved successfully',
                'config': current_config
            }
            
        except Exception as e:
            error_msg = f"Failed to save config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Reset operation result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Reset to default configuration
            default_config = get_default_backbone_config()
            self.update_config(default_config)
            
            # Sync to UI if available
            if self._ui_components:
                self._config_handler.sync_to_ui(self._ui_components, default_config)
            
            self.logger.info("🔄 Configuration reset to defaults")
            return {
                'success': True,
                'message': 'Configuration reset to defaults',
                'config': default_config
            }
            
        except Exception as e:
            error_msg = f"Failed to reset config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            if self._operation_manager:
                self._operation_manager.cleanup()
            
            # Cleanup UI logging bridge
            self._cleanup_ui_logging_bridge()
            
            # Clear references
            self._config_handler = None
            self._operation_manager = None
            self._ui_components = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _cleanup_ui_logging_bridge(self) -> None:
        """Cleanup UI logging bridge handlers."""
        try:
            import logging
            from smartcash.ui.core.logging.ui_logging_manager import cleanup_ui_logging
            
            # Cleanup UI logging for this module
            cleanup_ui_logging('model.backbone')
            
            # Remove custom handlers from backend services
            backend_namespaces = [
                'smartcash.model',
                'smartcash.dataset.preprocessor', 
                'smartcash.dataset.augmentor',
                'smartcash.common'
            ]
            
            for namespace in backend_namespaces:
                logger = logging.getLogger(namespace)
                # Remove all handlers that were added by this module
                for handler in logger.handlers[:]:
                    if hasattr(handler, 'log_func'):  # Our custom handler
                        logger.removeHandler(handler)
            
            self.logger.debug("✅ UI logging bridge cleanup completed")
                
        except Exception as e:
            self.logger.debug(f"Error during logging cleanup: {e}")


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_backbone_uimodule_instance: Optional[BackboneUIModule] = None


def create_backbone_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> BackboneUIModule:
    """
    Create a new backbone UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        BackboneUIModule instance
    """
    module = BackboneUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_backbone_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> BackboneUIModule:
    """
    Get or create backbone UIModule singleton instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize if not exists
        **kwargs: Additional arguments
        
    Returns:
        BackboneUIModule singleton instance
    """
    global _backbone_uimodule_instance
    
    if _backbone_uimodule_instance is None:
        _backbone_uimodule_instance = create_backbone_uimodule(
            config=config,
            auto_initialize=auto_initialize,
            **kwargs
        )
    
    return _backbone_uimodule_instance


def reset_backbone_uimodule() -> None:
    """Reset the backbone UIModule singleton instance."""
    global _backbone_uimodule_instance
    
    if _backbone_uimodule_instance:
        _backbone_uimodule_instance.cleanup()
        _backbone_uimodule_instance = None


# ==================== CONVENIENCE FUNCTIONS ====================

def initialize_backbone_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Initialize and display the backbone UI.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately
        **kwargs: Additional arguments
        
    Returns:
        If display=True: Returns None (displays UI directly)
        If display=False: Returns a dictionary with UI components and status
    """
    try:
        # Get the module and UI components
        module = get_backbone_uimodule(config=config, **kwargs)
        ui_components = module.get_ui_components()
        
        # Prepare the result dictionary
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'status': module.get_backbone_status()
        }
        
        # Display the UI if requested
        if display and ui_components:
            from IPython import get_ipython
            from IPython.display import display as ipython_display, clear_output
            
            # Clear any existing output
            if get_ipython() is not None:
                clear_output(wait=True)
            
            # Get the main UI container and display it
            main_ui = ui_components.get('main_container')
            if main_ui is not None:
                try:
                    # Get the widget using the show() method
                    ui_widget = main_ui.show()
                    # Display the widget
                    ipython_display(ui_widget)
                except Exception as e:
                    # Fallback to simple display if anything goes wrong
                    logger = get_module_logger("smartcash.ui.model.backbone")
                    logger.error(f"Error displaying UI: {str(e)}")
                    ipython_display(main_ui)
                return None  # Don't return data when display=True
        
        return result
        
    except Exception as e:
        # Always return a dictionary, even on error
        return {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'status': {}
        }


def get_backbone_components() -> Dict[str, Any]:
    """
    Get backbone UI components from singleton instance.
    
    Returns:
        UI components dictionary
    """
    try:
        module = get_backbone_uimodule(auto_initialize=False)
        return module.get_ui_components()
    except:
        return {}


# ==================== TEMPLATE REGISTRATION ====================

def register_backbone_shared_methods() -> None:
    """Register backbone shared methods for cross-module access."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register module factory functions
        SharedMethodRegistry.register_method(
            'backbone.create_module',
            create_backbone_uimodule,
            description='Create backbone UIModule instance'
        )
        
        SharedMethodRegistry.register_method(
            'backbone.get_module',
            get_backbone_uimodule,
            description='Get backbone UIModule singleton'
        )
        
        SharedMethodRegistry.register_method(
            'backbone.reset_module',
            reset_backbone_uimodule,
            description='Reset backbone UIModule singleton'
        )
        
    except Exception as e:
        # Silently fail if shared methods not available
        pass


def register_backbone_template() -> None:
    """Register backbone module template."""
    try:
        from smartcash.ui.core.template_registry import register_template
        
        template_info = {
            'name': 'backbone',
            'title': '🧬 Backbone Models',
            'description': 'Backbone model configuration with early training pipeline',
            'category': 'model',
            'factory_function': create_backbone_uimodule,
            'config_function': get_default_backbone_config,
            'singleton_function': get_backbone_uimodule,
            'reset_function': reset_backbone_uimodule
        }
        
        register_template('backbone', template_info)
        
    except Exception as e:
        # Silently fail if template registry not available
        pass


# Auto-register shared methods and template
register_backbone_shared_methods()
register_backbone_template()