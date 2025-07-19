# -*- coding: utf-8 -*-
"""
File: smartcash/ui/model/backbone/backbone_uimodule.py
Description: Refactored implementation of the Backbone Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Optional

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

from smartcash.ui.core.decorators import suppress_ui_init_logs
from .components.backbone_ui import create_backbone_ui
from .configs.backbone_config_handler import BackboneConfigHandler
from .configs.backbone_defaults import get_default_backbone_config


class BackboneUIModule(BaseUIModule):
    """
    Backbone UI Module.
    """
    # Define required UI components at class level
    _required_components = [
        'main_container',
        'header_container',
        'form_container',
        'action_container',
        'operation_container'
    ]

    def __init__(self, enable_environment: bool = False):
        """
        Initialize the Backbone UI module.
        
        Args:
            enable_environment: Whether to enable environment management features
        """
        # Call parent initializer with required parameters
        super().__init__(
            module_name='backbone',
            parent_module='model',
            enable_environment=enable_environment
        )
        
        # Initialize log buffer for pre-operation-container logs
        self._log_buffer = []
        
        # Operation container reference for logging
        self._operation_container = None
        
        self.logger.debug("BackboneUIModule initialized.")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_backbone_config()

    def create_config_handler(self, config: Dict[str, Any]) -> BackboneConfigHandler:
        """Creates a configuration handler instance."""
        return BackboneConfigHandler(config)

    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the UI components for the module."""
        return create_backbone_ui(config=config)

    def _register_default_operations(self) -> None:
        """Register default operation handlers including backbone-specific operations."""
        # Call parent method to register base operations
        super()._register_default_operations()
        
        # Note: Dynamic button handler registration is now handled by BaseUIModule
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Backbone module-specific button handlers."""
        # Start with base handlers (save, reset)
        handlers = {}
        
        # Add Backbone-specific handlers 
        backbone_handlers = {
            'validate': self._operation_validate,
            'build': self._operation_build,
            'save': self._handle_save_config,
            'reset': self._handle_reset_config,
        }
        
        handlers.update(backbone_handlers)
        return handlers
    
    def _register_module_button_handlers(self) -> None:
        """Register module-specific button handlers."""
        try:
            # Get module-specific handlers
            module_handlers = self._get_module_button_handlers()
            
            # Register each handler
            for button_id, handler in module_handlers.items():
                self.register_button_handler(button_id, handler)
                self.logger.debug(f"✅ Registered backbone button handler: {button_id}")
            
            # Setup button handlers after registering them
            self._setup_button_handlers()
            
            self.logger.info(f"🎯 Registered {len(module_handlers)} backbone button handlers")
            
        except Exception as e:
            self.logger.error(f"Failed to register module button handlers: {e}", exc_info=True)
        
    def _setup_operation_container(self) -> bool:
        """
        Set up the operation container for the module.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Get the operation container from UI components
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self.logger.error("UI components not available for operation container setup")
                return False
                
            # Store reference to operation container
            self._operation_container = self._ui_components.get('operation_container')
            
            if not self._operation_container:
                self.logger.warning("Operation container not found in UI components")
                return False
                
            # Flush any buffered logs to the operation container
            self._flush_log_buffer()
            
            self.logger.debug("✅ Operation container setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup operation container: {e}", exc_info=True)
            return False

    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Backbone module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success:
                # Set UI components in config handler for extraction
                if self._config_handler and hasattr(self._config_handler, 'set_ui_components'):
                    self._config_handler.set_ui_components(self._ui_components)
                
                # Setup operation container reference for logging
                self._setup_operation_container()
                
                # Register module-specific button handlers
                self._register_module_button_handlers()
                
                # Flush any buffered logs to operation container
                self._flush_log_buffer()
                
                # Log initialization completion (Operation Checklist 3.2)
                self.log("🧬 Backbone module siap digunakan", 'info')
                self.log("✅ Semua fitur backbone tersedia", 'success')
                
                # Log module readiness
                self.log("🏗️ Modul backbone siap digunakan", 'info')
                
                self.logger.debug("✅ Backbone module initialization completed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Backbone module: {e}")
            return False

    def _operation_validate(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle validation operation with backend integration."""
        def validate_data():
            # Ensure UI components are ready first
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            # Check data prerequisites
            prereq_check = self._check_data_prerequisites()
            if not prereq_check['success']:
                return {'valid': False, 'message': f"Prerequisites missing: {prereq_check['message']}"}
            
            return {'valid': True}
        
        def execute_validate():
            self.log("🔍 Memulai validasi konfigurasi backbone...", 'info')
            return self._execute_validate_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Validasi Backbone",
            operation_func=execute_validate,
            button=button,
            validation_func=validate_data,
            success_message="Validasi backbone berhasil diselesaikan",
            error_message="Kesalahan validasi backbone"
        )

    def _operation_build(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle build operation with backend integration."""
        def validate_build():
            # Ensure UI components are ready first
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            # Check data prerequisites
            prereq_check = self._check_data_prerequisites()
            if not prereq_check['success']:
                return {'valid': False, 'message': f"Prerequisites missing: {prereq_check['message']}"}
            
            return {'valid': True}
        
        def execute_build():
            self.log("🏗️ Memulai pembangunan model backbone...", 'info')
            return self._execute_build_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Pembangunan Model",
            operation_func=execute_build,
            button=button,
            validation_func=validate_build,
            success_message="Pembangunan model berhasil diselesaikan",
            error_message="Kesalahan pembangunan model"
        )

    def _check_data_prerequisites(self) -> Dict[str, Any]:
        """Check if all required data is available."""
        try:
            missing = []
            warnings = []
            
            # Check pretrained models
            pretrained_check = self._check_pretrained_models()
            if not pretrained_check['available']:
                warnings.append("Pretrained models not found")
            
            # Check raw data using preprocessor API
            raw_data_check = self._check_raw_data()
            if not raw_data_check['available']:
                missing.append("Raw data not found")
            
            # Check preprocessed data
            preprocessed_check = self._check_preprocessed_data()
            if not preprocessed_check['available']:
                warnings.append("Preprocessed data not found")
            
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
            backbone_config = self.get_current_config().get('backbone', {})
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

    def _check_raw_data(self) -> Dict[str, Any]:
        """Check raw data using preprocessor API."""
        try:
            # Use preprocessor API to check raw data
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            status = get_preprocessing_status(config=self.get_current_config())
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

    def _check_preprocessed_data(self) -> Dict[str, Any]:
        """Check preprocessed data using preprocessor API."""
        try:
            # Use preprocessor API to check preprocessed data
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            status = get_preprocessing_status(config=self.get_current_config())
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

    # ==================== OPERATION EXECUTION METHODS ====================

    def _execute_validate_operation(self) -> Dict[str, Any]:
        """Execute the validation operation using operation handler."""
        try:
            from .operations.backbone_validate_operation import BackboneValidateOperationHandler
            
            # Create handler with current UI components and config
            handler = BackboneValidateOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Validasi berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Validasi gagal') if result else 'Validasi gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in validation operation: {e}"}

    def _execute_build_operation(self) -> Dict[str, Any]:
        """Execute the build operation using operation handler."""
        try:
            from .operations.backbone_build_operation import BackboneBuildOperationHandler
            
            # Create handler with current UI components and config
            handler = BackboneBuildOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Pembangunan model berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Pembangunan model gagal') if result else 'Pembangunan model gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in build operation: {e}"}

    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        updater = self.get_component('operation_summary_updater')
        if updater and callable(updater):
            self.log(f"Memperbarui ringkasan operasi.", 'debug')
            updater(content)
        else:
            self.log("Komponen updater ringkasan operasi tidak ditemukan atau tidak dapat dipanggil.", 'warning')

    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not hasattr(self, '_log_buffer') or not self._log_buffer:
                return
                
            # Ensure operation container is available
            if not hasattr(self, '_operation_container') or not self._operation_container:
                self.logger.warning("⚠️ Operation container not available for log buffer flush")
                return
            
            # Flush all buffered logs
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer after flushing
            buffered_logs = len(self._log_buffer)
            self._log_buffer.clear()
            self.logger.debug(f"✅ Flushed {buffered_logs} logs to operation container")
            
        except Exception as e:
            self.logger.error(f"Failed to flush log buffer: {e}")

    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            UI components dictionary
        """
        return self._ui_components or {}

    def _handle_save_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle save configuration button click."""
        try:
            self.log("💾 Menyimpan konfigurasi backbone...", 'info')
            result = self.save_config()
            if result.get('success', True):
                self.log("✅ Konfigurasi backbone berhasil disimpan", 'success')
                return {'success': True, 'message': 'Configuration saved successfully'}
            else:
                self.log("❌ Gagal menyimpan konfigurasi backbone", 'error')
                return {'success': False, 'message': result.get('message', 'Save failed')}
        except Exception as e:
            self.log(f"❌ Error menyimpan konfigurasi: {e}", 'error')
            return {'success': False, 'message': str(e)}

    def _handle_reset_config(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle reset configuration button click."""
        try:
            self.log("🔄 Reset konfigurasi backbone ke default...", 'info')
            result = self.reset_config()
            if result.get('success', True):
                self.log("✅ Konfigurasi backbone berhasil direset", 'success')
                return {'success': True, 'message': 'Configuration reset successfully'}
            else:
                self.log("❌ Gagal reset konfigurasi backbone", 'error')
                return {'success': False, 'message': result.get('message', 'Reset failed')}
        except Exception as e:
            self.log(f"❌ Error reset konfigurasi: {e}", 'error')
            return {'success': False, 'message': str(e)}
