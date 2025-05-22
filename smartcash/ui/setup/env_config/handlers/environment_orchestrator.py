"""
File: smartcash/ui/setup/env_config/handlers/environment_orchestrator.py
Deskripsi: Orchestrator untuk koordinasi setup environment - SRP untuk coordination
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.handlers.environment_checker import EnvironmentChecker
from smartcash.ui.setup.env_config.handlers.drive_connector import DriveConnector
from smartcash.ui.setup.env_config.handlers.directory_manager import DirectoryManager
from smartcash.ui.setup.env_config.handlers.config_manager import ConfigFileManager

class EnvironmentOrchestrator:
    """
    Orchestrator untuk koordinasi semua tahap setup environment
    Mengelola urutan dan dependensi antar handler
    """
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi orchestrator dengan UI components
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "env_orchestrator")
        
        # Initialize handlers
        self.checker = EnvironmentChecker(ui_components)
        self.drive_connector = DriveConnector(ui_components)
        self.directory_manager = DirectoryManager(ui_components)
        self.config_manager = ConfigFileManager(ui_components)
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check status environment lengkap
        
        Returns:
            Dictionary status environment
        """
        return self.checker.get_environment_status()
    
    def perform_setup(self) -> bool:
        """
        Lakukan setup environment lengkap dengan progress tracking
        
        Returns:
            Success status
        """
        self.logger.info("🚀 Memulai setup environment SmartCash...")
        self._update_progress(0.1, "Memulai setup...")
        
        try:
            # Step 1: Check current status
            self._update_progress(0.2, "🔍 Checking environment...")
            status = self.checker.get_environment_status()
            
            # Skip jika sudah ready (kecuali ada missing directories)
            if status['ready'] and not status['missing_dirs']:
                self.logger.success("✅ Environment sudah terkonfigurasi!")
                self._update_progress(1.0, "Setup selesai")
                return True
            
            # Step 2: Connect to Drive (jika Colab)
            self._update_progress(0.3, "📱 Menghubungkan Google Drive...")
            if status['is_colab']:
                success, message = self.drive_connector.ensure_drive_mounted()
                if not success:
                    self.logger.error(f"❌ Gagal setup: {message}")
                    return False
                
                # Create drive directories
                if not self.drive_connector.create_drive_directories():
                    self.logger.error("❌ Gagal membuat direktori Drive")
                    return False
            
            # Step 3: Setup directories
            self._update_progress(0.5, "📁 Membuat direktori...")
            if not self.directory_manager.create_local_directories():
                self.logger.error("❌ Gagal membuat direktori lokal")
                return False
            
            # Step 4: Create symlinks (jika Colab)
            self._update_progress(0.6, "🔗 Membuat symlinks...")
            if status['is_colab']:
                drive_paths = self.drive_connector.get_drive_paths()
                if not self.directory_manager.create_symlinks(drive_paths):
                    self.logger.warning("⚠️ Beberapa symlink gagal dibuat")
            
            # Step 5: Setup config directory
            self._update_progress(0.7, "📝 Setup konfigurasi...")
            drive_config_path = None
            if status['is_colab']:
                drive_paths = self.drive_connector.get_drive_paths()
                drive_config_path = drive_paths.get('configs')
            
            if not self.config_manager.setup_config_directory(drive_config_path):
                self.logger.error("❌ Gagal setup direktori config")
                return False
            
            # Step 6: Copy default configs
            self._update_progress(0.8, "📋 Menyalin konfigurasi...")
            if not self.config_manager.copy_default_configs():
                self.logger.warning("⚠️ Beberapa config gagal disalin")
            
            # Step 7: Initialize config manager
            self._update_progress(0.9, "🔧 Inisialisasi config manager...")
            config_manager_instance = self.config_manager.initialize_config_manager()
            if config_manager_instance:
                self.ui_components['config_manager'] = config_manager_instance
            
            # Step 8: Final verification
            self._update_progress(0.95, "✅ Verifikasi setup...")
            final_status = self.checker.get_environment_status()
            
            if final_status['ready'] or not final_status['missing_dirs']:
                self.logger.success("🎉 Setup environment berhasil!")
                self._update_progress(1.0, "Setup selesai")
                return True
            else:
                self.logger.warning("⚠️ Setup selesai dengan beberapa issue")
                self._update_progress(1.0, "Setup selesai (dengan warning)")
                return True  # Return True agar button tidak disabled
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup: {str(e)}")
            self._update_progress(0.0, "Setup gagal")
            return False
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress UI"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.components.progress_tracking import update_progress
                update_progress(self.ui_components, int(value * 100), 100, message)
            except ImportError:
                pass