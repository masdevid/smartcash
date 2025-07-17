"""
Colab Factory for creating and managing shared operations without operation manager.
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.operation_container import OperationContainer

from .base_colab_operation import BaseColabOperation
from .init_operation import InitOperation
from .drive_mount_operation import DriveMountOperation
from .symlink_operation import SymlinkOperation
from .folders_operation import FoldersOperation
from .config_sync_operation import ConfigSyncOperation
from .env_setup_operation import EnvSetupOperation
from .verify_operation import VerifyOperation


class ColabOperationFactory:
    """Factory for creating and managing Colab operations with shared methods."""
    
    def __init__(self, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None):
        """Initialize the factory.
        
        Args:
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
        """
        self.logger = get_module_logger("smartcash.ui.setup.colab.operations.factory")
        self.config = config
        self.operation_container = operation_container
        self._operations_cache = {}
    
    def get_operation(self, operation_type: str) -> BaseColabOperation:
        """Get or create an operation instance.
        
        Args:
            operation_type: Type of operation to get
            
        Returns:
            Operation instance
        """
        if operation_type not in self._operations_cache:
            self._operations_cache[operation_type] = self._create_operation(operation_type)
        return self._operations_cache[operation_type]
    
    def _create_operation(self, operation_type: str) -> BaseColabOperation:
        """Create an operation instance.
        
        Args:
            operation_type: Type of operation to create
            
        Returns:
            Operation instance
        """
        operation_classes = {
            'init': InitOperation,
            'drive_mount': DriveMountOperation,
            'symlink': SymlinkOperation,
            'folders': FoldersOperation,
            'config_sync': ConfigSyncOperation,
            'env_setup': EnvSetupOperation,
            'verify': VerifyOperation
        }
        
        operation_class = operation_classes.get(operation_type)
        if not operation_class:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        return operation_class(
            operation_type,
            self.config,
            operation_container=self.operation_container
        )
    
    def execute_operation(self, operation_type: str, method_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific operation method.
        
        Args:
            operation_type: Type of operation
            method_name: Method to execute
            **kwargs: Arguments for the method
            
        Returns:
            Operation result
        """
        try:
            operation = self.get_operation(operation_type)
            method = getattr(operation, method_name)
            return method(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to execute {operation_type}.{method_name}: {e}")
            return {'success': False, 'error': str(e)}


# Shared methods for common operations
def init_environment(config: Dict[str, Any], 
                     operation_container: Optional[OperationContainer] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Initialize Colab environment.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('init', 'execute_init', progress_callback=progress_callback)


def mount_drive(config: Dict[str, Any],
                operation_container: Optional[OperationContainer] = None,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Mount Google Drive.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('drive_mount', 'execute_mount_drive', progress_callback=progress_callback)


def create_symlinks(config: Dict[str, Any],
                    operation_container: Optional[OperationContainer] = None,
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Create symbolic links.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('symlink', 'execute_create_symlinks', progress_callback=progress_callback)


def create_folders(config: Dict[str, Any],
                   operation_container: Optional[OperationContainer] = None,
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Create required folders.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('folders', 'execute_create_folders', progress_callback=progress_callback)


def sync_config(config: Dict[str, Any],
                operation_container: Optional[OperationContainer] = None,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Sync configuration.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('config_sync', 'execute_sync_configs', progress_callback=progress_callback)


def setup_environment(config: Dict[str, Any],
                      operation_container: Optional[OperationContainer] = None,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Setup environment variables.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('env_setup', 'execute_setup_environment', progress_callback=progress_callback)


def verify_setup(config: Dict[str, Any],
                 operation_container: Optional[OperationContainer] = None,
                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Verify setup integrity.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    factory = ColabOperationFactory(config, operation_container)
    return factory.execute_operation('verify', 'execute_verify_setup', progress_callback=progress_callback)


def execute_full_setup(config: Dict[str, Any],
                       operation_container: Optional[OperationContainer] = None,
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Execute full Colab setup sequence.
    
    Args:
        config: Configuration dictionary
        operation_container: Optional operation container
        progress_callback: Optional progress callback
        
    Returns:
        Operation result
    """
    logger = get_module_logger("smartcash.ui.setup.colab.operations.factory.full_setup")
    
    try:
        logger.info("🚀 Starting full Colab setup sequence...")
        
        # Define setup sequence
        setup_sequence = [
            ('init', init_environment, "🔧 Initializing environment..."),
            ('drive_mount', mount_drive, "📁 Mounting Google Drive..."),
            ('symlink', create_symlinks, "🔗 Creating symbolic links..."),
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
                progress_callback(base_progress, stage_message)
            
            logger.info(f"Executing stage {i+1}/{total_stages}: {stage_name}")
            
            # Create stage-specific progress callback
            def stage_progress(progress, message):
                if progress_callback:
                    stage_weight = 100 / total_stages
                    overall_progress = base_progress + (progress / 100) * stage_weight
                    progress_callback(overall_progress, f"{stage_name.upper()}: {message}")
            
            result = stage_func(config, operation_container, stage_progress)
            stage_results[stage_name] = result
            
            if not result.get('success', False):
                error_msg = f"Setup failed at stage '{stage_name}': {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'failed_stage': stage_name,
                    'stage_results': stage_results,
                    'error': error_msg
                }
        
        if progress_callback:
            progress_callback(100, "🎉 Setup completed successfully!")
        
        logger.info("✅ Full Colab setup completed successfully")
        
        return {
            'success': True,
            'stage_results': stage_results,
            'message': 'Full Colab setup completed successfully'
        }
        
    except Exception as e:
        error_msg = f"Full setup failed: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}


def detect_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # noqa: ARG001
    """Detect Colab environment.
    
    Args:
        config: Optional configuration
        
    Returns:
        Environment detection result
    """
    try:
        import google.colab  # noqa: F401
        return {"is_colab": True, "runtime_type": "colab"}
    except ImportError:
        return {"is_colab": False, "runtime_type": "local"}


def get_available_operations() -> List[str]:
    """Get list of available operations.
    
    Returns:
        List of operation names
    """
    return [
        'init',
        'drive_mount', 
        'symlink',
        'folders',
        'config_sync',
        'env_setup',
        'verify'
    ]


def get_operation_info(operation_type: str) -> Dict[str, Any]:
    """Get information about a specific operation.
    
    Args:
        operation_type: Type of operation
        
    Returns:
        Operation information
    """
    operation_info = {
        'init': {
            'name': 'Environment Initialization',
            'description': 'Initialize Colab environment and detect system',
            'phase': 'initialization'
        },
        'drive_mount': {
            'name': 'Google Drive Mount',
            'description': 'Mount Google Drive for data access',
            'phase': 'mounting'
        },
        'symlink': {
            'name': 'Symbolic Links',
            'description': 'Create symbolic links for project structure',
            'phase': 'linking'
        },
        'folders': {
            'name': 'Folder Creation',
            'description': 'Create required project folders',
            'phase': 'structure'
        },
        'config_sync': {
            'name': 'Configuration Sync',
            'description': 'Synchronize project configuration',
            'phase': 'configuration'
        },
        'env_setup': {
            'name': 'Environment Setup',
            'description': 'Setup environment variables and paths',
            'phase': 'environment'
        },
        'verify': {
            'name': 'Setup Verification',
            'description': 'Verify complete setup integrity',
            'phase': 'verification'
        }
    }
    
    return operation_info.get(operation_type, {'name': 'Unknown', 'description': 'Unknown operation'})