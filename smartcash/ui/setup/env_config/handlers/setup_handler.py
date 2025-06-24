"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Fixed setup handler dengan proper UI state management dan workflow
"""

import time
from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.constants import PROGRESS_STEPS, STATUS_MESSAGES

class SetupHandler:
    """🚀 Fixed setup handler dengan comprehensive workflow"""
    
    def __init__(self):
        self.status_handler = StatusHandler()
        self.drive_handler = DriveHandler()
        self.folder_handler = FolderHandler()
        self.config_handler = ConfigHandler()
    
    def handle_setup_click(self, ui_components: Dict[str, Any], logger):
        """Handle setup button click dengan comprehensive workflow"""
        logger.info("🚀 Memulai konfigurasi environment...")
        
        # Set UI to setup state
        self._set_setup_running_state(ui_components)
        
        try:
            # Execute setup workflow
            success = self._execute_comprehensive_setup(ui_components, logger)
            
            if success:
                # Set UI to ready state
                self._set_setup_complete_state(ui_components, logger)
            else:
                # Reset UI to setup needed state
                self._set_setup_failed_state(ui_components, logger)
                
        except Exception as e:
            logger.error(f"❌ Setup failed: {str(e)}")
            self._set_setup_failed_state(ui_components, logger, str(e))
    
    def _execute_comprehensive_setup(self, ui_components: Dict[str, Any], logger) -> bool:
        """Execute comprehensive setup workflow dengan progress tracking"""
        
        # Step 1: Environment Analysis
        self._update_progress(ui_components, 'analysis', logger)
        if not self._analyze_environment(logger):
            return False
        
        # Step 2: Drive Connection
        self._update_progress(ui_components, 'drive_mount', logger)
        if not self._ensure_drive_connection(logger):
            return False
        
        # Step 3: Folder Structure
        self._update_progress(ui_components, 'folders', logger)
        if not self._create_folder_structure(logger):
            return False
        
        # Step 4: Configuration Setup
        self._update_progress(ui_components, 'configs', logger)
        if not self._setup_configurations(logger):
            return False
        
        # Step 5: Symlink Creation
        self._update_progress(ui_components, 'symlinks', logger)
        if not self._create_symlinks(logger):
            return False
        
        # Step 6: Final Validation
        self._update_progress(ui_components, 'validation', logger)
        if not self._validate_complete_setup(logger):
            return False
        
        # Step 7: Complete
        self._update_progress(ui_components, 'complete', logger)
        return True
    
    def _analyze_environment(self, logger) -> bool:
        """Analyze current environment"""
        try:
            logger.info("🔍 Menganalisis environment saat ini...")
            
            # Basic environment checks
            import os, sys
            import platform
            
            logger.info(f"📊 Python version: {sys.version.split()[0]}")
            logger.info(f"📊 Platform: {platform.system()}")
            
            # Check if we're in Colab
            try:
                import google.colab
                logger.info("📱 Running in Google Colab")
                return True
            except ImportError:
                logger.warning("⚠️ Not running in Google Colab")
                return True  # Continue anyway
                
        except Exception as e:
            logger.error(f"❌ Environment analysis failed: {str(e)}")
            return False
    
    def _ensure_drive_connection(self, logger) -> bool:
        """Ensure Google Drive is connected"""
        try:
            return self.drive_handler.ensure_drive_mounted(logger)
        except Exception as e:
            logger.error(f"❌ Drive connection failed: {str(e)}")
            return False
    
    def _create_folder_structure(self, logger) -> bool:
        """Create required folder structure"""
        try:
            return self.folder_handler.create_smartcash_structure(logger)
        except Exception as e:
            logger.error(f"❌ Folder creation failed: {str(e)}")
            return False
    
    def _setup_configurations(self, logger) -> bool:
        """Setup configuration files"""
        try:
            return self.config_handler.setup_essential_configs(logger)
        except Exception as e:
            logger.error(f"❌ Configuration setup failed: {str(e)}")
            return False
    
    def _create_symlinks(self, logger) -> bool:
        """Create symbolic links"""
        try:
            import os
            from pathlib import Path
            
            logger.info("🔗 Membuat symbolic links...")
            
            # Create symlinks dari Drive ke local paths
            drive_path = '/content/drive/MyDrive/SmartCash'
            local_path = '/content/smartcash_data'
            
            if os.path.exists(drive_path) and not os.path.exists(local_path):
                os.symlink(drive_path, local_path)
                logger.success(f"✅ Symlink created: {local_path} -> {drive_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Symlink creation failed: {str(e)}")
            return False
    
    def _validate_complete_setup(self, logger) -> bool:
        """Final validation of complete setup"""
        try:
            logger.info("✅ Memvalidasi setup lengkap...")
            
            # Get fresh status check
            final_status = self.status_handler.get_comprehensive_status()
            
            if final_status.get('ready', False):
                logger.success("🎉 Setup validation passed!")
                return True
            else:
                missing = final_status.get('missing_items', [])
                logger.warning(f"⚠️ Setup incomplete: {', '.join(missing)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Setup validation failed: {str(e)}")
            return False
    
    def _update_progress(self, ui_components: Dict[str, Any], step: str, logger):
        """Update progress bar dan text"""
        if step not in PROGRESS_STEPS:
            return
        
        step_info = PROGRESS_STEPS[step]
        progress_value = step_info['range'][1]  # Take max value
        progress_label = step_info['label']
        
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = progress_value
        
        # Update progress text
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                f"<span style='color: #007bff;'>{progress_label}</span>"
            )
        
        logger.info(f"📊 Progress: {progress_value}% - {progress_label}")
        time.sleep(0.5)  # Small delay untuk visual feedback
    
    def _set_setup_running_state(self, ui_components: Dict[str, Any]):
        """Set UI state saat setup sedang berjalan"""
        # Disable button
        if 'setup_button' in ui_components:
            setup_button = ui_components['setup_button']
            setup_button.disabled = True
            setup_button.description = "⚙️ Setting Up..."
            setup_button.button_style = 'warning'
        
        # Update status panel
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #856404; padding: 10px; margin: 5px 0; "
                "background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;'>"
                "⚙️ Sedang mengkonfigurasi environment - mohon tunggu...</p>"
            )
    
    def _set_setup_complete_state(self, ui_components: Dict[str, Any], logger):
        """Set UI state setelah setup berhasil"""
        # Update button to success state (disabled)
        if 'setup_button' in ui_components:
            setup_button = ui_components['setup_button']
            setup_button.disabled = True
            setup_button.description = "✅ Environment Ready"
            setup_button.button_style = 'success'
        
        # Update status panel
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #155724; padding: 10px; margin: 5px 0; "
                "background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px;'>"
                "🎉 Environment berhasil dikonfigurasi dan siap digunakan!</p>"
            )
        
        # Set progress to 100%
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #28a745; font-weight: bold;'>🎉 Setup selesai - Environment siap digunakan!</span>"
            )
        
        logger.success("🎉 Environment berhasil dikonfigurasi!")
    
    def _set_setup_failed_state(self, ui_components: Dict[str, Any], logger, error_msg: str = None):
        """Set UI state jika setup gagal"""
        # Re-enable button untuk retry
        if 'setup_button' in ui_components:
            setup_button = ui_components['setup_button']
            setup_button.disabled = False
            setup_button.description = "🔄 Retry Setup"
            setup_button.button_style = 'danger'
        
        # Update status panel dengan error
        error_text = f" - {error_msg}" if error_msg else ""
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                f"<p style='color: #721c24; padding: 10px; margin: 5px 0; "
                f"background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px;'>"
                f"❌ Setup gagal{error_text}. Silakan coba lagi.</p>"
            )
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #dc3545;'>❌ Setup gagal - Silakan coba lagi</span>"
            )
        
        logger.error(f"❌ Setup gagal{error_text}")