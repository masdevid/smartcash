"""
File: smartcash/ui/setup/env_config/handlers/drive_setup_handler.py
Deskripsi: Drive setup dengan proper mount detection dan state management yang robust
"""

import os
import shutil
import time
from typing import Dict, Any, Tuple
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class DriveSetupHandler:
    """Drive setup dengan proper mount detection dan state management"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = None  # Will be initialized when needed
        self.required_folders = ['data', 'configs', 'exports', 'logs', 'models', 'output']
        self.config_templates = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
    
    def _ensure_logger(self):
        """Ensure logger is initialized"""
        if self.logger is None:
            self.logger = create_ui_logger_bridge(self.ui_components, "drive_setup")
    
    def ensure_drive_mounted(self) -> Tuple[bool, str]:
        """Mount Drive dengan proper state management dan wait mechanism"""
        self._ensure_logger()
        
        if not self._is_colab():
            return True, "Local environment"
        
        # Check existing mount dengan comprehensive validation
        if self._check_drive_ready_comprehensive():
            return True, "Drive sudah terhubung dan siap"
        
        try:
            from google.colab import drive
            self.logger.info("📱 Mounting Google Drive...")
            drive.mount('/content/drive')
            
            # Wait untuk mount completion dengan timeout dan state validation
            return self._wait_for_drive_ready_comprehensive(max_wait=45)
            
        except Exception as e:
            return False, f"Mount error: {str(e)}"
    
    def _check_drive_ready_comprehensive(self) -> bool:
        """Comprehensive check apakah Drive benar-benar ready dengan multiple validations"""
        drive_path = Path('/content/drive/MyDrive')
        
        # Basic existence check
        if not drive_path.exists():
            return False
        
        # Directory accessibility check
        try:
            # Test basic listing
            list(drive_path.iterdir())
        except Exception:
            return False
        
        # Write access test dengan retry
        for attempt in range(3):
            try:
                test_file = drive_path / f'.smartcash_ready_test_{attempt}'
                test_content = f'ready_test_{int(time.time())}'
                
                # Write test
                test_file.write_text(test_content)
                
                # Immediate read test
                read_content = test_file.read_text()
                
                # Cleanup
                test_file.unlink()
                
                # Validate content
                if read_content == test_content:
                    return True
                    
            except Exception:
                if attempt < 2:
                    time.sleep(0.5)  # Brief delay before retry
                continue
        
        return False
    
    def _wait_for_drive_ready_comprehensive(self, max_wait: int = 45) -> Tuple[bool, str]:
        """Wait untuk Drive ready dengan comprehensive validation dan progress updates"""
        self.logger.info("⏳ Menunggu Drive mount completion...")
        
        for i in range(max_wait):
            if self._check_drive_ready_comprehensive():
                self.logger.success(f"✅ Drive ready dalam {i+1} detik")
                
                # Additional settling time untuk stability
                time.sleep(1)
                
                # Final validation
                if self._check_drive_ready_comprehensive():
                    return True, "Drive siap digunakan dan telah divalidasi"
                
            # Progress updates setiap 5 detik
            if i % 5 == 0 and i > 0:
                self.logger.info(f"⏳ Validating Drive state... ({i}s)")
            
            time.sleep(1)
        
        # Final attempt dengan extended validation
        if self._check_drive_ready_comprehensive():
            return True, "Drive ready setelah timeout (late detection)"
        
        return False, f"Timeout setelah {max_wait}s - Drive mungkin terpasang tapi tidak dapat divalidasi"
    
    def create_drive_folders(self, drive_base_path: str) -> Dict[str, bool]:
        """Create folders di Drive dengan comprehensive error handling"""
        self._ensure_logger()
        results = {}
        drive_base = Path(drive_base_path)
        
        # Ensure base path exists dengan retry
        for attempt in range(3):
            try:
                drive_base.mkdir(parents=True, exist_ok=True)
                break
            except Exception as e:
                if attempt == 2:
                    self.logger.error(f"❌ Gagal create base path: {str(e)}")
                    return {folder: False for folder in self.required_folders}
                time.sleep(1)
        
        # Create each folder dengan individual error handling
        for folder in self.required_folders:
            folder_path = drive_base / folder
            success = False
            
            for attempt in range(3):
                try:
                    if not folder_path.exists():
                        folder_path.mkdir(parents=True, exist_ok=True)
                        # Verify creation dengan delay
                        time.sleep(0.2)
                        success = folder_path.exists() and folder_path.is_dir()
                    else:
                        success = folder_path.is_dir()
                    
                    if success:
                        break
                        
                except Exception as e:
                    if attempt == 2:
                        self.logger.error(f"❌ Folder {folder}: {str(e)}")
                    else:
                        time.sleep(0.5)
            
            results[folder] = success
        
        success_count = sum(results.values())
        if success_count > 0:
            self.logger.success(f"📁 Setup {success_count}/{len(self.required_folders)} folder di Drive")
        
        return results
    
    def clone_config_templates(self, drive_base_path: str) -> Dict[str, bool]:
        """Clone configs dengan comprehensive validation"""
        self._ensure_logger()
        results = {}
        repo_config_path = Path('/content/smartcash/configs')
        drive_config_path = Path(drive_base_path) / 'configs'
        
        # Ensure destination exists
        try:
            drive_config_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"❌ Gagal create config directory: {str(e)}")
            return {config: False for config in self.config_templates}
        
        cloned_count = 0
        for config_file in self.config_templates:
            src_file = repo_config_path / config_file
            dst_file = drive_config_path / config_file
            success = False
            
            try:
                if src_file.exists():
                    if not dst_file.exists():
                        # Copy dengan validation
                        shutil.copy2(src_file, dst_file)
                        time.sleep(0.1)  # Brief settling time
                        
                        # Verify copy dengan content validation
                        if dst_file.exists() and dst_file.stat().st_size > 0:
                            success = True
                            cloned_count += 1
                    else:
                        # File sudah ada, verify integrity
                        success = dst_file.exists() and dst_file.stat().st_size > 0
                else:
                    self.logger.warning(f"⚠️ Source config tidak ditemukan: {config_file}")
                    
            except Exception as e:
                self.logger.error(f"❌ Config {config_file}: {str(e)}")
            
            results[config_file] = success
        
        if cloned_count > 0:
            self.logger.success(f"📋 Clone {cloned_count} config template")
        
        return results
    
    def create_symlinks(self, drive_base_path: str) -> Dict[str, bool]:
        """Create symlinks dengan comprehensive validation dan cleanup"""
        self._ensure_logger()
        
        if not self._is_colab():
            return {folder: True for folder in self.required_folders}
        
        results = {}
        drive_base = Path(drive_base_path)
        
        for folder in self.required_folders:
            local_path = Path(f'/content/{folder}')
            drive_path = drive_base / folder
            success = False
            
            try:
                # Check jika symlink valid sudah ada
                if (local_path.is_symlink() and 
                    local_path.exists() and 
                    local_path.resolve() == drive_path.resolve()):
                    success = True
                    results[folder] = success
                    continue
                
                # Backup existing local directory jika ada
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = Path(f'/content/{folder}_backup_{int(time.time())}')
                    try:
                        shutil.move(local_path, backup_path)
                        self.logger.info(f"📦 Backup {folder} ke {backup_path.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal backup {folder}: {str(e)}")
                        # Force remove jika backup gagal
                        if local_path.is_dir():
                            shutil.rmtree(local_path)
                        else:
                            local_path.unlink()
                
                # Remove broken symlink jika ada
                if local_path.is_symlink():
                    local_path.unlink()
                
                # Ensure drive path exists
                if not drive_path.exists():
                    drive_path.mkdir(parents=True, exist_ok=True)
                
                # Create symlink dengan retry
                for attempt in range(3):
                    try:
                        local_path.symlink_to(drive_path)
                        time.sleep(0.2)  # Brief settling time
                        
                        # Comprehensive validation
                        if (local_path.exists() and 
                            local_path.is_symlink() and 
                            local_path.resolve() == drive_path.resolve()):
                            success = True
                            break
                            
                    except Exception as e:
                        if attempt == 2:
                            self.logger.error(f"❌ Symlink {folder} (attempt {attempt+1}): {str(e)}")
                        else:
                            time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"❌ Symlink {folder}: {str(e)}")
            
            results[folder] = success
        
        success_count = sum(results.values())
        if success_count > 0:
            self.logger.success(f"🔗 Setup {success_count}/{len(self.required_folders)} symlink")
        
        return results
    
    def perform_complete_setup(self, drive_base_path: str) -> Dict[str, Any]:
        """Complete setup dengan proper sequencing dan state management"""
        self._ensure_logger()
        
        # Pre-setup validation
        if not self._check_drive_ready_comprehensive():
            return {'success': False, 'error': 'Drive not ready for setup'}
        
        self.logger.info(f"🚀 Memulai setup lengkap di: {drive_base_path}")
        
        # Step 1: Create Drive folders
        self._update_progress(0.3, "📁 Creating Drive folders...")
        folder_results = self.create_drive_folders(drive_base_path)
        
        # Validate folder creation
        folder_success_rate = sum(folder_results.values()) / len(folder_results)
        if folder_success_rate < 0.5:
            self.logger.warning("⚠️ Sebagian besar folder gagal dibuat")
        
        # Step 2: Clone configs
        self._update_progress(0.5, "📋 Cloning configs...")
        config_results = self.clone_config_templates(drive_base_path)
        
        # Validate config cloning
        config_success_count = sum(config_results.values())
        if config_success_count < 3:
            self.logger.warning("⚠️ Config templates tidak lengkap")
        
        # Step 3: Create symlinks (most critical step)
        self._update_progress(0.7, "🔗 Creating symlinks...")
        
        # Additional delay untuk memastikan Drive state stable
        time.sleep(1)
        
        symlink_results = self.create_symlinks(drive_base_path)
        
        # Comprehensive success evaluation
        folder_success = sum(folder_results.values()) >= len(folder_results) * 0.8
        config_success = config_success_count >= 3  # Minimal essential configs
        symlink_success = sum(symlink_results.values()) >= len(symlink_results) * 0.8
        
        overall_success = folder_success and config_success and symlink_success
        
        # Final validation setelah setup
        self._update_progress(0.9, "✅ Validating setup...")
        if overall_success:
            # Quick validation test
            validation_passed = self._validate_setup_integrity(drive_base_path)
            if not validation_passed:
                self.logger.warning("⚠️ Setup validation menunjukkan beberapa issue")
                overall_success = False
        
        return {
            'folders': folder_results,
            'configs': config_results,
            'symlinks': symlink_results,
            'success': overall_success,
            'validation': {
                'folder_success_rate': folder_success_rate,
                'config_count': config_success_count,
                'symlink_success_rate': sum(symlink_results.values()) / len(symlink_results)
            }
        }
    
    def _validate_setup_integrity(self, drive_base_path: str) -> bool:
        """Quick validation untuk memastikan setup integrity"""
        try:
            drive_base = Path(drive_base_path)
            
            # Check critical folders
            critical_folders = ['data', 'configs']
            for folder in critical_folders:
                folder_path = drive_base / folder
                if not folder_path.exists() or not folder_path.is_dir():
                    return False
            
            # Check symlinks
            critical_symlinks = ['data', 'configs']
            for folder in critical_symlinks:
                local_path = Path(f'/content/{folder}')
                if not (local_path.exists() and local_path.is_symlink()):
                    return False
            
            # Test write access
            test_file = drive_base / 'data' / '.setup_validation_test'
            test_file.write_text('validation')
            content = test_file.read_text()
            test_file.unlink()
            
            return content == 'validation'
            
        except Exception:
            return False
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress dengan error handling"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import update_progress
                update_progress(self.ui_components, int(value * 100), 100, message)
            except ImportError:
                pass
    
    def _is_colab(self) -> bool:
        """Check apakah berjalan di Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False