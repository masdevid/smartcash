"""
File: smartcash/ui/setup/env_config/handlers/environment_checker.py
Deskripsi: Handler untuk pengecekan status environment - SRP untuk checking functionality
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.common.environment import get_environment_manager

class EnvironmentChecker:
    """Handler untuk pengecekan status environment - fokus hanya pada checking"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi checker dengan UI components
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "env_checker")
        self.env_manager = get_environment_manager()
    
    def check_colab_environment(self) -> bool:
        """Check apakah berjalan di Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def check_drive_connection(self) -> Dict[str, Any]:
        """
        Check status koneksi Google Drive
        
        Returns:
            Dictionary status drive
        """
        if not self.check_colab_environment():
            return {'connected': True, 'path': None, 'message': 'Local environment'}
            
        try:
            is_mounted = self.env_manager.is_drive_mounted
            drive_path = self.env_manager.drive_path if is_mounted else None
            
            return {
                'connected': is_mounted,
                'path': str(drive_path) if drive_path else None,
                'message': '✅ Drive terhubung' if is_mounted else '❌ Drive tidak terhubung'
            }
        except Exception as e:
            return {'connected': False, 'path': None, 'message': f'Error checking drive: {str(e)}'}
    
    def check_required_directories(self) -> Dict[str, bool]:
        """
        Check direktori yang diperlukan
        
        Returns:
            Dictionary status direktori
        """
        required_dirs = ['data', 'exports', 'logs', 'models', 'output', 'configs']
        status = {}
        
        if self.check_colab_environment():
            base_path = Path("/content")
        else:
            base_path = self.env_manager.base_dir
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            status[dir_name] = dir_path.exists()
        
        return status
    
    def check_config_files(self) -> Dict[str, bool]:
        """
        Check file konfigurasi yang diperlukan
        
        Returns:
            Dictionary status config files
        """
        config_files = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml'
        ]
        status = {}
        
        if self.check_colab_environment():
            config_path = Path("/content/configs")
        else:
            config_path = self.env_manager.base_dir / "configs"
        
        for config_file in config_files:
            file_path = config_path / config_file
            status[config_file] = file_path.exists()
        
        return status
    
    def get_environment_status(self) -> Dict[str, Any]:
        """
        Get status lengkap environment
        
        Returns:
            Dictionary status environment
        """
        self.logger.info("🔍 Memeriksa status environment...")
        
        # Check basic environment
        is_colab = self.check_colab_environment()
        
        # Check drive connection
        drive_status = self.check_drive_connection()
        
        # Check directories
        dir_status = self.check_required_directories()
        missing_dirs = [name for name, exists in dir_status.items() if not exists]
        
        # Check config files
        config_status = self.check_config_files()
        missing_configs = [name for name, exists in config_status.items() if not exists]
        
        # Determine if ready
        is_ready = (
            drive_status['connected'] and 
            len(missing_dirs) == 0 and 
            len(missing_configs) <= 2  # Allow some missing configs
        )
        
        status = {
            'is_colab': is_colab,
            'drive_connected': drive_status['connected'],
            'drive_path': drive_status['path'],
            'directories_exist': dir_status,
            'missing_dirs': missing_dirs,
            'config_files_exist': config_status,
            'missing_configs': missing_configs,
            'ready': is_ready,
            'base_dir': str(self.env_manager.base_dir)
        }
        
        # Log status
        self._log_status(status)
        
        return status
    
    def _log_status(self, status: Dict[str, Any]):
        """Log status environment"""
        env_type = "🔬 Google Colab" if status['is_colab'] else "🏠 Local"
        self.logger.info(f"Environment: {env_type}")
        
        if status['is_colab']:
            drive_icon = "✅" if status['drive_connected'] else "❌"
            self.logger.info(f"Google Drive: {drive_icon} {status.get('drive_path', 'Tidak terhubung')}")
        
        # Log missing directories
        if status['missing_dirs']:
            self.logger.warning(f"📁 Direktori belum ada: {', '.join(status['missing_dirs'])}")
        else:
            self.logger.success("📁 Semua direktori tersedia")
        
        # Log missing configs
        if status['missing_configs']:
            self.logger.info(f"📝 Config akan dibuat: {len(status['missing_configs'])} file")
        else:
            self.logger.success("📝 Semua config tersedia")
        
        # Overall status
        if status['ready']:
            self.logger.success("🎉 Environment siap digunakan!")
        else:
            self.logger.info("🔧 Environment perlu dikonfigurasi")