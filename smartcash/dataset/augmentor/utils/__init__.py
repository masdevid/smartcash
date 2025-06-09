"""
File: smartcash/dataset/augmentor/utils/__init__.py
Deskripsi: Utils module exports dengan one-liner utilities dan dataset detection
"""

from smartcash.dataset.augmentor.utils.config_validator import (
    validate_augmentation_config,
    get_default_augmentation_config,
    load_config_from_file,
    reload_default_config
)

from smartcash.dataset.augmentor.utils.progress_bridge import (
    ProgressBridge,
    create_progress_bridge,
    make_progress_callback
)

from smartcash.dataset.augmentor.utils.file_processor import FileProcessor
from smartcash.dataset.augmentor.utils.file_scanner import FileScanner
from smartcash.dataset.augmentor.utils.filename_manager import FilenameManager
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver
from smartcash.dataset.augmentor.utils.balance_calculator import BalanceCalculator
from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
from smartcash.dataset.augmentor.utils.symlink_manager import SymlinkManager

# One-liner utilities
validate_config = lambda config: validate_augmentation_config(config)
get_default_config = lambda: get_default_augmentation_config()
create_file_processor = lambda config: FileProcessor(config)
create_file_scanner = lambda: FileScanner()
create_filename_manager = lambda: FilenameManager()
create_path_resolver = lambda config: PathResolver(config)
create_balance_calculator = lambda config: BalanceCalculator(config)
create_cleanup_manager = lambda config, progress=None: CleanupManager(config, progress)
create_symlink_manager = lambda config: SymlinkManager(config)

__all__ = [
    # Config validation
    'validate_augmentation_config',
    'get_default_augmentation_config', 
    'load_config_from_file',
    'reload_default_config',
    
    # Progress utilities
    'ProgressBridge',
    'create_progress_bridge',
    'make_progress_callback',
    
    # File utilities
    'FileProcessor',
    'FileScanner',
    'FilenameManager',
    'PathResolver',
    'BalanceCalculator',
    'CleanupManager',
    'SymlinkManager',
    
    # One-liner factories
    'validate_config',
    'get_default_config',
    'create_file_processor',
    'create_file_scanner',
    'create_filename_manager',
    'create_path_resolver',
    'create_balance_calculator',
    'create_cleanup_manager',
    'create_symlink_manager'
]