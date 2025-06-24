# File: smartcash/ui/setup/env_config/handlers/setup_handler.py
# Deskripsi: Handler untuk setup workflow dan status management

import os
import time
from typing import Dict, Any
from pathlib import Path
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
from smartcash.ui.setup.env_config.utils.progress_updater import ProgressUpdater

class SetupHandler:
    """🚀 Handler untuk environment setup workflow"""
    
    def __init__(self):
        self.status_handler = StatusHandler()
        self.drive_handler = DriveHandler()
        self.folder_handler = FolderHandler()
        self.config_handler = ConfigHandler()
        self.progress_updater = ProgressUpdater()
        
    def perform_initial_status_check(self, ui_components: Dict[str, Any], logger):
        """Lakukan pengecekan status awal environment"""
        logger.info("🔍 Memeriksa status environment saat ini...")
        
        # Check environment status
        env_status = self.status_handler.check_environment_status()
        
        # Update status panel
        self._update_status_panel(ui_components, env_status)
        
        # Log hasil check
        if env_status.get('ready', False):
            logger.success("✅ Environment sudah terkonfigurasi dengan baik")
        else:
            logger.warning("🔧 Environment perlu dikonfigurasi")
            missing_items = env_status.get('missing_items', [])
            if missing_items:
                logger.info(f"📋 Item yang perlu dikonfigurasi: {', '.join(missing_items)}")
    
    def handle_setup_click(self, ui_components: Dict[str, Any], logger):
        """Handle klik tombol setup dengan workflow lengkap"""
        logger.info("🚀 Memulai konfigurasi environment...")
        
        # Disable button selama proses
        self._disable_setup_button(ui_components)
        
        try:
            # Setup workflow steps
            if self._execute_setup_workflow(ui_components, logger):
                logger.success("🎉 Environment berhasil dikonfigurasi!")
                self._update_status_panel(ui_components, {'ready': True})
            else:
                logger.error("❌ Setup gagal - silakan coba lagi")
                
        except Exception as e:
            logger.error(f"❌ Error selama setup: {str(e)}")
        finally:
            self._enable_setup_button(ui_components)
    
    def _execute_setup_workflow(self, ui_components: Dict[str, Any], logger) -> bool:
        """Execute setup workflow dengan progress tracking"""
        # Step 1: Mount Google Drive
        self.progress_updater.update_progress(ui_components, 15, "📱 Menghubungkan Google Drive...")
        if not self.drive_handler.mount_drive(logger):
            return False
        
        # Step 2: Create folder structures
        self.progress_updater.update_progress(ui_components, 40, "📁 Membuat struktur folder...")
        self.folder_handler.create_folder_structures(logger)
        
        # Step 3: Setup configurations
        self.progress_updater.update_progress(ui_components, 70, "📋 Menyalin konfigurasi...")
        self.config_handler.setup_configurations(logger)
        
        # Step 4: Final validation
        self.progress_updater.update_progress(ui_components, 90, "✅ Memvalidasi setup...")
        is_valid = self._validate_setup(logger)
        
        # Complete
        self.progress_updater.update_progress(ui_components, 100, "🎉 Setup selesai!")
        return is_valid
    
    def _validate_setup(self, logger) -> bool:
        """Validasi hasil setup"""
        try:
            key_paths = [
                "/content/drive/MyDrive/SmartCash/data",
                "/content/drive/MyDrive/SmartCash/configs"
            ]
            
            all_valid = True
            for path in key_paths:
                if os.path.exists(path):
                    logger.success(f"✅ Validasi OK: {path}")
                else:
                    logger.warning(f"⚠️ Path tidak ditemukan: {path}")
                    all_valid = False
            
            return all_valid
        except Exception as e:
            logger.warning(f"⚠️ Warning validasi: {str(e)}")
            return False
    
    def _update_status_panel(self, ui_components: Dict[str, Any], status: Dict[str, Any]):
        """Update panel status berdasarkan hasil check"""
        if 'status_panel' not in ui_components:
            return
            
        panel = ui_components['status_panel']
        is_ready = status.get('ready', False)
        
        if is_ready:
            panel.value = """
            <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">
                ✅ Environment sudah terkonfigurasi dengan baik
            </div>
            """
        else:
            panel.value = """
            <div style="background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;">
                🔧 Environment perlu dikonfigurasi - Klik tombol setup untuk memulai
            </div>
            """
    
    def _disable_setup_button(self, ui_components: Dict[str, Any]):
        """Disable setup button selama proses"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = True
            ui_components['setup_button'].description = "⏳ Setting up..."
    
    def _enable_setup_button(self, ui_components: Dict[str, Any]):
        """Re-enable setup button setelah proses"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "🚀 Setup Environment"