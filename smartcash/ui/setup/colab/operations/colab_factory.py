"""
Colab Factory (Optimized) - Enhanced Mixin Integration
Factory for creating and managing Colab operations with shared methods.
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.components.operation_container import OperationContainer
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from .base_colab_operation import BaseColabOperation
from .init_operation import InitOperation
from .drive_mount_operation import DriveMountOperation
from .symlink_operation import SymlinkOperation
from .folders_operation import FoldersOperation
from .config_sync_operation import ConfigSyncOperation
from .env_setup_operation import EnvSetupOperation
from .verify_operation import VerifyOperation


class ColabOperationFactory(LoggingMixin):
    """Optimized factory with enhanced mixin integration."""
    
    def __init__(self, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None):
        super().__init__()
        self.config = config
        self.operation_container = operation_container
        self._operations_cache = {}
        self._operation_classes = {
            'init': InitOperation, 'drive_mount': DriveMountOperation, 'symlink': SymlinkOperation,
            'folders': FoldersOperation, 'config_sync': ConfigSyncOperation, 'env_setup': EnvSetupOperation, 'verify': VerifyOperation
        }
    
    def get_operation(self, operation_type: str) -> BaseColabOperation:
        if operation_type not in self._operations_cache:
            self._operations_cache[operation_type] = self._create_operation(operation_type)
        return self._operations_cache[operation_type]
    
    def _create_operation(self, operation_type: str) -> BaseColabOperation:
        operation_class = self._operation_classes.get(operation_type)
        if not operation_class:
            raise ValueError(f"Unknown operation type: {operation_type}")
        return operation_class(operation_type, self.config, operation_container=self.operation_container)
    
    def execute_operation(self, operation_type: str, method_name: str, **kwargs) -> Dict[str, Any]:
        try:
            operation = self.get_operation(operation_type)
            method = getattr(operation, method_name)
            return method(**kwargs)
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            error_msg = f"Failed to execute {operation_type}.{method_name}: {e}"
            self.log_error(f"{error_msg}\n\nTraceback:\n{error_traceback}")
            return {'success': False, 'error': str(e), 'traceback': error_traceback, 'operation': operation_type, 'method': method_name}


# Streamlined operation functions
def init_environment(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, 
                     progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('init', 'execute_init', progress_callback=progress_callback)

def mount_drive(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('drive_mount', 'execute_mount_drive', progress_callback=progress_callback)

def create_symlinks(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                    progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('symlink', 'execute_create_symlinks', progress_callback=progress_callback)

def create_folders(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                   progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('folders', 'execute_create_folders', progress_callback=progress_callback)

def sync_config(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('config_sync', 'execute_sync_configs', progress_callback=progress_callback)

def setup_environment(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                      progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('env_setup', 'execute_setup_environment', progress_callback=progress_callback)

def verify_setup(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                 progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('verify', 'execute_verify_setup', progress_callback=progress_callback)

def execute_full_setup(config: Dict[str, Any], operation_container: Optional[OperationContainer] = None,
                       progress_callback: Optional[Callable] = None, logger=None) -> Dict[str, Any]:  # noqa: ARG001
    """Execute full Colab setup sequence with enhanced error handling."""
    factory = ColabOperationFactory(config, operation_container)
    
    # Reset progress tracker if available
    if operation_container and hasattr(operation_container, 'get') and callable(operation_container.get):
        progress_tracker = operation_container.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            try:
                progress_tracker.reset()
                factory.log_debug("Progress tracker reset successfully")
            except Exception as e:
                factory.log_warning(f"Failed to reset progress tracker: {e}")
    
    try:
        factory.log_info("🚀 Starting full setup sequence...")
        
        # Detect environment to determine which operations to run
        env_info = detect_environment(config)
        is_colab = env_info.get('is_colab', False)
        
        if is_colab:
            factory.log_info("📱 Colab environment detected - running full sequence")
            # Define full colab setup sequence
            setup_sequence = [
                ('init', init_environment, "🔧 Initializing environment..."),
                ('drive_mount', mount_drive, "📁 Mounting Google Drive..."),
                ('symlink', create_symlinks, "🔗 Creating symbolic links..."),
                ('folders', create_folders, "📂 Creating folders..."),
                ('config_sync', sync_config, "⚙️ Syncing configuration..."),
                ('env_setup', setup_environment, "🌍 Setting up environment..."),
                ('verify', verify_setup, "🔍 Verifying setup...")
            ]
        else:
            factory.log_info("🏠 Local environment detected - skipping drive mount and symlinks")
            # Define local setup sequence (skip drive mount and symlinks)
            setup_sequence = [
                ('init', init_environment, "🔧 Initializing environment..."),
                ('folders', create_folders, "📂 Creating folders..."),
                ('config_sync', sync_config, "⚙️ Syncing configuration..."),
                ('env_setup', setup_environment, "🌍 Setting up environment..."),
                ('verify', verify_setup, "🔍 Verifying setup...")
            ]
        
        stage_results = {}
        total_stages = len(setup_sequence)
        
        for i, (stage_name, stage_func, stage_message) in enumerate(setup_sequence):
            base_progress = (i / total_stages) * 100
            
            if progress_callback:
                try:
                    progress_callback(base_progress, stage_message)
                except Exception as e:
                    factory.log_warning(f"Error updating progress: {e}")
            
            factory.log_info(f"Executing stage {i+1}/{total_stages}: {stage_name}")
            
            # Create stage-specific progress callback
            def stage_progress(progress, message, stage_name=stage_name):  # Capture stage_name in closure
                if progress_callback:
                    try:
                        stage_weight = 100 / total_stages
                        overall_progress = base_progress + (progress / 100) * stage_weight
                        progress_callback(overall_progress, f"{stage_name.upper()}: {message}")
                    except Exception as e:
                        factory.log_warning(f"Error in stage progress callback: {e}")
            
            try:
                # Pass None for logger as individual operations will create their own loggers
                result = stage_func(config, operation_container, stage_progress, None)
                stage_results[stage_name] = result
                
                if not result.get('success', False):
                    error_msg = f"Setup failed at stage '{stage_name}': {result.get('error', 'Unknown error')}"
                    
                    # Include traceback in logs if available
                    if 'traceback' in result:
                        factory.log_error(f"{error_msg}\n\nTraceback:\n{result['traceback']}")
                    else:
                        factory.log_error(error_msg)
                        
                    # Update progress tracker with error state
                    if operation_container and hasattr(operation_container, 'get'):
                        progress_tracker = operation_container.get('progress_tracker')
                        if progress_tracker and hasattr(progress_tracker, 'set_all_error'):
                            progress_tracker.set_all_error(error_msg)
                    
                    return {'success': False, 'failed_stage': stage_name, 'stage_results': stage_results, 'error': error_msg}
                    
            except Exception as e:
                import traceback
                error_msg = f"Unexpected error in stage '{stage_name}': {str(e)}"
                tb = traceback.format_exc()
                factory.log_error(f"{error_msg}\n\nTraceback:\n{tb}")
                
                # Update progress tracker with error state
                if operation_container and hasattr(operation_container, 'get'):
                    progress_tracker = operation_container.get('progress_tracker')
                    if progress_tracker and hasattr(progress_tracker, 'set_all_error'):
                        progress_tracker.set_all_error(error_msg)
                
                return {'success': False, 'failed_stage': stage_name, 'stage_results': stage_results, 'error': error_msg, 'traceback': tb}
        
        if progress_callback:
            progress_callback(100, "🎉 Setup completed successfully!")
        
        factory.log_info("✅ Full Colab setup completed successfully")
        return {'success': True, 'stage_results': stage_results, 'message': 'Full Colab setup completed successfully'}
        
    except Exception as e:
        error_msg = f"Full setup failed: {str(e)}"
        factory.log_error(error_msg)
        return {'success': False, 'error': error_msg}

def detect_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # noqa: ARG001
    """Detect Colab environment with enhanced detection."""
    try:
        import google.colab  # noqa: F401
        return {"is_colab": True, "runtime_type": "colab", "features": ["drive_mount", "gpu_access", "tpu_access"]}
    except ImportError:
        import os
        # Alternative detection methods
        if 'COLAB_GPU' in os.environ or '/content' in os.getcwd():
            return {"is_colab": True, "runtime_type": "colab_alternative", "features": ["basic_colab"]}
        return {"is_colab": False, "runtime_type": "local", "features": ["local_development"]}

def get_available_operations() -> List[str]:
    """Get list of available operations."""
    return ['init', 'drive_mount', 'symlink', 'folders', 'config_sync', 'env_setup', 'verify']

def get_operation_info(operation_type: str) -> Dict[str, Any]:
    """Get comprehensive information about operations."""
    operation_info = {
        'init': {'name': 'Environment Initialization', 'description': 'Initialize Colab environment and detect system', 'phase': 'initialization', 'dependencies': [], 'duration': 'short'},
        'drive_mount': {'name': 'Google Drive Mount', 'description': 'Mount Google Drive for data access', 'phase': 'mounting', 'dependencies': ['init'], 'duration': 'medium'},
        'symlink': {'name': 'Symbolic Links', 'description': 'Create symbolic links for project structure', 'phase': 'linking', 'dependencies': ['drive_mount'], 'duration': 'short'},
        'folders': {'name': 'Folder Creation', 'description': 'Create required project folders', 'phase': 'structure', 'dependencies': ['symlink'], 'duration': 'short'},
        'config_sync': {'name': 'Configuration Sync', 'description': 'Synchronize project configuration', 'phase': 'configuration', 'dependencies': ['folders'], 'duration': 'medium'},
        'env_setup': {'name': 'Environment Setup', 'description': 'Setup environment variables and paths', 'phase': 'environment', 'dependencies': ['config_sync'], 'duration': 'medium'},
        'verify': {'name': 'Setup Verification', 'description': 'Verify complete setup integrity', 'phase': 'verification', 'dependencies': ['env_setup'], 'duration': 'short'}
    }
    return operation_info.get(operation_type, {'name': 'Unknown', 'description': 'Unknown operation', 'phase': 'unknown', 'dependencies': [], 'duration': 'unknown'})