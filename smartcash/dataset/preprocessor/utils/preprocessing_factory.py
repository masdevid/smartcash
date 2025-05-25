"""
File: smartcash/dataset/preprocessor/utils/preprocessing_factory.py
Deskripsi: Factory untuk creating preprocessing services dengan proper dependency injection
"""

from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor


class PreprocessingFactory:
    """Factory untuk creating preprocessing services dengan unified configuration."""
    
    @staticmethod
    def create_preprocessing_manager(config: Dict[str, Any], logger=None, 
                                   progress_callback: Optional[Callable] = None) -> PreprocessingManager:
        """
        Create PreprocessingManager dengan full dependency injection.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            progress_callback: Progress callback untuk UI notifications
            
        Returns:
            Configured PreprocessingManager instance
        """
        logger = logger or get_logger("PreprocessingManager")
        
        # Create manager dengan dependencies
        manager = PreprocessingManager(config, logger)
        
        # Register progress callback jika disediakan
        if progress_callback:
            manager.register_progress_callback(progress_callback)
        
        logger.debug("🏭 PreprocessingManager created via factory")
        return manager
    
    @staticmethod
    def create_dataset_checker(config: Dict[str, Any], logger=None) -> DatasetChecker:
        """
        Create DatasetChecker dengan configuration.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            
        Returns:
            Configured DatasetChecker instance
        """
        logger = logger or get_logger("DatasetChecker")
        
        checker = DatasetChecker(config, logger)
        
        logger.debug("🏭 DatasetChecker created via factory")
        return checker
    
    @staticmethod
    def create_cleanup_executor(config: Dict[str, Any], logger=None,
                              progress_callback: Optional[Callable] = None) -> CleanupExecutor:
        """
        Create CleanupExecutor dengan configuration dan progress callback.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            progress_callback: Progress callback untuk cleanup updates
            
        Returns:
            Configured CleanupExecutor instance
        """
        logger = logger or get_logger("CleanupExecutor")
        
        executor = CleanupExecutor(config, logger)
        
        # Register progress callback jika disediakan
        if progress_callback:
            executor.register_progress_callback(progress_callback)
        
        logger.debug("🏭 CleanupExecutor created via factory")
        return executor
    
    @staticmethod
    def create_service_bundle(config: Dict[str, Any], logger=None,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Create complete service bundle untuk preprocessing operations.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            progress_callback: Progress callback untuk UI notifications
            
        Returns:
            Dictionary berisi semua preprocessing services
        """
        logger = logger or get_logger("PreprocessingFactory")
        
        # Create all services dengan shared configuration
        services = {
            'preprocessing_manager': PreprocessingFactory.create_preprocessing_manager(
                config, logger, progress_callback
            ),
            'dataset_checker': PreprocessingFactory.create_dataset_checker(config, logger),
            'cleanup_executor': PreprocessingFactory.create_cleanup_executor(
                config, logger, progress_callback
            )
        }
        
        # Add metadata
        services['_factory_metadata'] = {
            'created_services': list(services.keys()),
            'config_provided': bool(config),
            'progress_callback_registered': progress_callback is not None,
            'bundle_complete': True
        }
        
        logger.success(f"🏭 Service bundle created: {len(services)-1} services")
        return services
    
    @staticmethod
    def validate_service_dependencies(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dependencies untuk service creation.
        
        Args:
            config: Configuration dictionary untuk validation
            
        Returns:
            Dictionary validation result
        """
        validation = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'config_completeness': {}
        }
        
        # Check required config sections
        required_sections = ['data', 'preprocessing']
        for section in required_sections:
            if section not in config:
                validation['issues'].append(f"Missing config section: {section}")
                validation['valid'] = False
            else:
                validation['config_completeness'][section] = True
        
        # Check data directory
        data_dir = config.get('data', {}).get('dir')
        if not data_dir:
            validation['issues'].append("Missing data directory in config")
            validation['valid'] = False
        
        # Check preprocessing output directory
        output_dir = config.get('preprocessing', {}).get('output_dir')
        if not output_dir:
            validation['warnings'].append("No preprocessing output_dir specified, using default")
        
        # Validate paths if provided
        from pathlib import Path
        if data_dir and not Path(data_dir).exists():
            validation['warnings'].append(f"Data directory does not exist: {data_dir}")
        
        return validation