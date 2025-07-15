"""
File: smartcash/ui/setup/colab/colab_uimodule.py
Description: Colab Module implementation using BaseUIModule pattern with operation checklist compliance.
"""

from typing import Dict, Any, Optional
import os
import sys

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.logger import get_module_logger

# Environment management imports
from smartcash.common.environment import get_environment_manager, EnvironmentManager
from smartcash.common.constants.paths import get_paths_for_environment

# Colab module imports
from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
from smartcash.ui.setup.colab.configs.colab_config_handler import ColabConfigHandler
from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config
from smartcash.ui.setup.colab.operations.operation_manager import ColabOperationManager


class ColabUIModule(BaseUIModule):
    """
    Colab Module implementation using BaseUIModule pattern.
    
    Features:
    - 🌟 Google Colab environment detection and setup
    - 🔧 Sequential operations: INIT → DRIVE → SYMLINK → FOLDERS → CONFIG → ENV → VERIFY
    - 📊 Real-time progress tracking and UI-integrated logging
    - 💾 No-persistence configuration (Colab-specific requirement)
    - 🔄 Enhanced factory-based initialization functions
    - ✅ Full compliance with OPERATION_CHECKLISTS.md requirements
    """
    
    def __init__(self):
        """Initialize Colab UI module."""
        super().__init__(
            module_name='colab',
            parent_module='setup'
        )
        
        # Set required components for validation (Operation Checklist 1.2)
        self._required_components = [
            'main_container',
            'header_container', 
            'action_container',
            'operation_container'
        ]
        
        # Colab-specific attributes
        self._operation_manager: Optional[ColabOperationManager] = None
        self._environment_manager: Optional[EnvironmentManager] = None
        self._environment_detected = False
        self._is_colab_environment = False
        self._environment_paths = {}
        
        # Initialize log buffer for pre-operation-container logs
        self._log_buffer = []
        
        self.logger.debug("✅ ColabUIModule initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Colab module (BaseUIModule requirement)."""
        return get_default_colab_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> ColabConfigHandler:
        """Create config handler instance for Colab module (BaseUIModule requirement)."""
        handler = ColabConfigHandler(logger=self.logger)
        if config:
            handler.update_config(config)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Colab module (BaseUIModule requirement)."""
        try:
            self.logger.debug("Creating Colab UI components...")
            ui_components = create_colab_ui(config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Colab module with environment detection.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Detect environment first (Operation Checklist 8.3)
            self._detect_environment()
            
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success:
                # Setup operation manager after UI components are created
                self._setup_operation_manager()
                
                # Flush any buffered logs to operation container
                self._flush_log_buffer()
                
                # Log environment detection results (Operation Checklist 3.2)
                env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                
                if self._is_colab_environment:
                    self.log("✅ Berjalan di Google Colab - semua fitur tersedia", 'success')
                else:
                    self.log("⚠️ Tidak berjalan di Google Colab - beberapa fitur mungkin terbatas", 'warning')
                
                # Update status panel (Operation Checklist 7.1)
                self.update_operation_status("Siap untuk pengaturan lingkungan Colab", "info")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Colab module: {e}")
            return False
    
    def _detect_environment(self) -> None:
        """Detect environment using standardized EnvironmentManager (Operation Checklist 8.3)."""
        try:
            # Use standardized environment manager
            self._environment_manager = get_environment_manager(logger=self.logger)
            self._is_colab_environment = self._environment_manager.is_colab
            self._environment_detected = True
            
            # Get appropriate paths for current environment
            self._environment_paths = get_paths_for_environment(
                is_colab=self._is_colab_environment,
                is_drive_mounted=self._environment_manager.is_drive_mounted if self._is_colab_environment else False
            )
            
            env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
            self.logger.debug(f"✅ Environment detected: {env_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to detect environment: {e}")
            # Fallback to simple detection
            try:
                import google.colab
                self._is_colab_environment = True
            except ImportError:
                self._is_colab_environment = False
            self._environment_detected = True
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager with UI integration."""
        try:
            if not self._ui_components:
                raise RuntimeError("UI components must be created before operation manager")
            
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                raise RuntimeError("Operation container not found in UI components")
            
            self._operation_manager = ColabOperationManager(
                config=self.get_current_config(),
                operation_container=operation_container
            )
            
            # Set Colab environment flag
            if hasattr(self._operation_manager, 'set_colab_environment'):
                self._operation_manager.set_colab_environment(self._is_colab_environment)
            
            # Initialize operation manager
            self._operation_manager.initialize()
            
            self.logger.debug("✅ Operation manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}")
            raise
    
    def _register_default_operations(self) -> None:
        """Register default operations for Colab module (Operation Checklist 9.1)."""
        # Call parent method first
        super()._register_default_operations()
        
        # Register Colab-specific operations (Operation Checklist 8.3)
        self.register_operation_handler('full_setup', self._handle_full_setup)
        self.register_operation_handler('init_environment', self._handle_init_environment)
        self.register_operation_handler('mount_drive', self._handle_mount_drive)
        self.register_operation_handler('verify_setup', self._handle_verify_setup)
        self.register_operation_handler('detect_environment', self._handle_detect_environment)
        
        # Register button handlers (Operation Checklist 2.2)
        # Only register handlers for buttons that actually exist in the UI
        self.register_button_handler('colab_setup', self._handle_full_setup)
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not self._log_buffer:
                return
                
            # Display all buffered logs to operation container
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer
            self._log_buffer.clear()
            
        except Exception as e:
            self.logger.debug(f"Failed to flush log buffer: {e}")
    
    # ==================== OPERATION HANDLERS ====================
    
    def _handle_full_setup(self, button=None) -> Dict[str, Any]:
        """Handle full Colab setup operation (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Pengaturan Lengkap Colab")
            self.update_operation_status("Memulai pengaturan lengkap Colab...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            # Execute full setup with progress tracking (Operation Checklist 3.1)
            result = self._operation_manager.execute_full_setup()
            
            if result.get('success'):
                self.log_operation_complete("Pengaturan Lengkap Colab")
                self.update_operation_status("Pengaturan Colab berhasil diselesaikan", "info")
                self.log("✅ Pengaturan lengkap lingkungan Colab selesai", 'success')
            else:
                error_msg = result.get('message', 'Pengaturan gagal')
                self.log_operation_error("Pengaturan Lengkap Colab", error_msg)
                self.update_operation_status(f"Pengaturan gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan pengaturan lengkap: {e}"
            self.log_operation_error("Pengaturan Lengkap Colab", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_init_environment(self, button=None) -> Dict[str, Any]:
        """Handle environment initialization (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Inisialisasi Lingkungan")
            self.update_operation_status("Menginisialisasi lingkungan...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            result = self._operation_manager.execute_init()
            
            if result.get('success'):
                self.log_operation_complete("Inisialisasi Lingkungan")
                self.update_operation_status("Lingkungan berhasil diinisialisasi", "info")
            else:
                error_msg = result.get('message', 'Inisialisasi gagal')
                self.log_operation_error("Inisialisasi Lingkungan", error_msg)
                self.update_operation_status(f"Inisialisasi gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan inisialisasi lingkungan: {e}"
            self.log_operation_error("Inisialisasi Lingkungan", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_mount_drive(self, button=None) -> Dict[str, Any]:
        """Handle Google Drive mounting using EnvironmentManager (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Mount Google Drive")
            self.update_operation_status("Memasang Google Drive...", "info")
            
            if not self._is_colab_environment:
                warning_msg = "Mount Google Drive hanya tersedia di lingkungan Colab"
                self.log(f"⚠️ {warning_msg}", 'warning')
                self.update_operation_status(warning_msg, "warning")
                return {'success': False, 'message': warning_msg}
            
            if not self._environment_manager:
                self._detect_environment()  # Ensure environment manager is initialized
            
            # Use EnvironmentManager to mount drive
            success, message = self._environment_manager.mount_drive()
            
            if success:
                self.log_operation_complete("Mount Google Drive")
                self.update_operation_status("Google Drive berhasil dipasang", "info")
                drive_path = self._environment_manager.drive_path
                self.log(f"✅ Google Drive dipasang di: {drive_path}", 'success')
                
                # Update paths after successful mount
                self._environment_paths = get_paths_for_environment(
                    is_colab=True,
                    is_drive_mounted=True
                )
                
                return {
                    'success': True, 
                    'message': message,
                    'path': str(drive_path) if drive_path else '/content/drive',
                    'paths': self._environment_paths
                }
            else:
                self.log_operation_error("Mount Google Drive", message)
                self.update_operation_status(f"Mount drive gagal: {message}", "error")
                return {'success': False, 'message': message}
                
        except Exception as e:
            error_msg = f"Kesalahan mount drive: {e}"
            self.log_operation_error("Mount Google Drive", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_verify_setup(self, button=None) -> Dict[str, Any]:
        """Handle setup verification (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Verifikasi Pengaturan")
            self.update_operation_status("Memverifikasi pengaturan...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            result = self._operation_manager.execute_verify()
            
            if result.get('success'):
                self.log_operation_complete("Verifikasi Pengaturan")
                self.update_operation_status("Verifikasi pengaturan selesai", "info")
                self.log("✅ Pengaturan lingkungan Colab telah diverifikasi", 'success')
            else:
                error_msg = result.get('message', 'Verifikasi gagal')
                self.log_operation_error("Verifikasi Pengaturan", error_msg)
                self.update_operation_status(f"Verifikasi gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan verifikasi: {e}"
            self.log_operation_error("Verifikasi Pengaturan", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_detect_environment(self, button=None) -> Dict[str, Any]:
        """Handle environment detection operation."""
        try:
            self.log_operation_start("Deteksi Lingkungan")
            self.update_operation_status("Mendeteksi lingkungan...", "info")
            
            self._detect_environment()
            
            env_info = {
                'is_colab': self._is_colab_environment,
                'runtime_type': 'colab' if self._is_colab_environment else 'local',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            }
            
            env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
            self.log(f"🌍 Lingkungan: {env_type}", 'info')
            self.log(f"🐍 Python: {env_info['python_version']}", 'info')
            self.log(f"💻 Platform: {env_info['platform']}", 'info')
            
            self.log_operation_complete("Deteksi Lingkungan")
            self.update_operation_status(f"Lingkungan terdeteksi: {env_type}", "info")
            
            return {
                'success': True,
                'message': f'Lingkungan terdeteksi: {env_type}',
                'environment': env_info
            }
            
        except Exception as e:
            error_msg = f"Kesalahan deteksi lingkungan: {e}"
            self.log_operation_error("Deteksi Lingkungan", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    # ==================== COLAB-SPECIFIC METHODS ====================
    
    def get_colab_status(self) -> Dict[str, Any]:
        """
        Get current Colab environment status (Operation Checklist 9.2).
        
        Returns:
            Status information dictionary
        """
        try:
            status = {
                'initialized': self._is_initialized,
                'module_name': self.module_name,
                'environment_detected': self._environment_detected,
                'is_colab': self._is_colab_environment,
                'config_loaded': self._config_handler is not None,
                'operation_manager_ready': self._operation_manager is not None,
                'ui_created': bool(self._ui_components)
            }
            
            # Add environment-specific information
            if self._is_colab_environment and self._environment_manager:
                try:
                    # Use EnvironmentManager for drive status
                    status['drive_mounted'] = self._environment_manager.is_drive_mounted
                    status['drive_path'] = str(self._environment_manager.drive_path) if self._environment_manager.drive_path else None
                    status['base_directory'] = str(self._environment_manager.base_dir)
                    status['data_directory'] = str(self._environment_manager.get_dataset_path())
                except Exception as e:
                    self.logger.warning(f"Failed to get drive status: {e}")
                    status['drive_mounted'] = False
            
            return status
            
        except Exception as e:
            return {'error': f'Pemeriksaan status gagal: {str(e)}'}
    
    def is_colab_environment(self) -> bool:
        """Check if running in Google Colab environment."""
        return self._is_colab_environment
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get detailed environment information using EnvironmentManager."""
        base_info = {
            'is_colab': self._is_colab_environment,
            'runtime_type': 'colab' if self._is_colab_environment else 'local',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'environment_detected': self._environment_detected,
            'paths': self._environment_paths
        }
        
        # Add EnvironmentManager system info if available
        if self._environment_manager:
            try:
                system_info = self._environment_manager.get_system_info()
                base_info.update({
                    'base_directory': system_info.get('base_directory'),
                    'data_directory': system_info.get('data_directory'),
                    'drive_mounted': system_info.get('drive_mounted'),
                    'drive_path': system_info.get('drive_path'),
                    'cuda_available': system_info.get('cuda_available'),
                    'total_memory_gb': system_info.get('total_memory_gb'),
                    'available_memory_gb': system_info.get('available_memory_gb')
                })
            except Exception as e:
                self.logger.warning(f"Failed to get system info from EnvironmentManager: {e}")
        
        return base_info


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

def initialize_colab_ui(config: Optional[Dict[str, Any]] = None, 
                       show_display: bool = True, 
                       **kwargs) -> Optional[Dict[str, Any]]:
    """Initialize and optionally display the Colab UI module."""
    # Filter out conflicting display-related parameters from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['display', 'show_display']}
    
    # Determine final display value - prioritize explicit 'display' parameter
    if 'display' in kwargs:
        final_display = kwargs['display']
    else:
        final_display = show_display
    
    return EnhancedUIModuleFactory.create_and_display(
        module_class=ColabUIModule,
        config=config,
        display=final_display,
        **filtered_kwargs
    )

def get_colab_components(config: Optional[Dict[str, Any]] = None, 
                        **kwargs) -> Optional[Dict[str, Any]]:
    """Get Colab UI components without displaying."""
    return initialize_colab_ui(config=config, show_display=False, **kwargs)


# ==================== CONVENIENCE FUNCTIONS ====================

# Global module instance for singleton pattern
_colab_module_instance: Optional[ColabUIModule] = None

def create_colab_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> ColabUIModule:
    """
    Create a new Colab UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        ColabUIModule instance
    """
    module = ColabUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_colab_uimodule() -> Optional[ColabUIModule]:
    """Get the current Colab UIModule instance."""
    global _colab_module_instance
    return _colab_module_instance


def reset_colab_uimodule() -> None:
    """Reset the global Colab UIModule instance."""
    global _colab_module_instance
    if _colab_module_instance:
        try:
            _colab_module_instance.cleanup()
        except:
            pass
    _colab_module_instance = None


def display_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Display Colab UI and return components."""
    return initialize_colab_ui(config=config, display=True, **kwargs)


def detect_colab_environment() -> Dict[str, Any]:
    """Detect if running in Google Colab environment."""
    try:
        import google.colab
        return {"is_colab": True, "runtime_type": "colab"}
    except ImportError:
        return {"is_colab": False, "runtime_type": "local"}


def mount_google_drive(drive_path: str = "/content/drive") -> Dict[str, Any]:
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount(drive_path)
        return {"success": True, "path": drive_path}
    except ImportError:
        return {"success": False, "error": "Not running in Google Colab"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== SHARED METHODS REGISTRATION ====================

def register_colab_shared_methods() -> None:
    """Register shared methods for Colab module (Operation Checklist 9.1)."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register Colab-specific shared methods
        SharedMethodRegistry.register_method(
            'colab.detect_environment',
            detect_colab_environment,
            description='Detect Colab environment'
        )
        
        SharedMethodRegistry.register_method(
            'colab.mount_drive',
            mount_google_drive,
            description='Mount Google Drive'
        )
        
        SharedMethodRegistry.register_method(
            'colab.get_status',
            lambda: create_colab_uimodule().get_colab_status(),
            description='Get Colab environment status'
        )
        
        logger = get_module_logger("smartcash.ui.setup.colab.shared")
        logger.debug("📋 Registered Colab shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.setup.colab.shared")
        logger.error(f"Failed to register shared methods: {e}")


# Auto-register when module is imported
try:
    register_colab_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")