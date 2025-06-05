"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI untuk environment config dengan Environment Summary yang informatif dan UI yang stabil
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_config_orchestrator import EnvironmentConfigOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class EnvConfigComponent:
    """Component UI untuk environment config dengan Environment Summary yang informatif"""
    
    def __init__(self):
        """Inisialisasi component dengan UI yang stabile dan Environment Summary"""
        # Setup UI components first tanpa logging
        self.ui_components = UIFactory.create_ui_components()
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Initialize orchestrator tanpa logging dulu
        self.orchestrator = EnvironmentConfigOrchestrator(self.ui_components)
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
        self.setup_completed = False
        
        # Initialize environment manager untuk summary (silent mode)
        self._init_environment_manager_silent()
    
    def _init_environment_manager_silent(self):
        """Initialize environment manager tanpa logging untuk menghindari output sebelum UI"""
        try:
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            self.env_manager = get_silent_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception:
            # Silent initialization, tidak ada log sebelum UI ready
            pass
    
    def _init_environment_manager(self):
        """Initialize environment manager dan tampilkan summary"""
        try:
            from smartcash.common.environment import get_environment_manager
            self.env_manager = get_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
            
            # Display environment summary
            self._display_environment_summary()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal inisialisasi environment manager: {str(e)}")
    
    def _display_environment_summary(self):
        """Display comprehensive environment summary dengan silent system info gathering"""
        try:
            from smartcash.ui.setup.env_config.helpers.system_info_helper import SystemInfoHelper
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            
            # Use silent environment manager untuk prevent log leakage
            silent_env_manager = get_silent_environment_manager()
            
            # Get enhanced system info dengan silent mode
            enhanced_info = SystemInfoHelper.get_system_info()
            
            # Get system info dari silent environment manager
            env_system_info = silent_env_manager.get_system_info()
            
            # Merge information
            merged_info = {**enhanced_info, **env_system_info}
            
            # Format summary lines
            summary_lines = SystemInfoHelper.format_system_summary(merged_info)
            
            # Get recommendations
            recommendations = SystemInfoHelper.get_system_recommendations(merged_info)
            
            # Build HTML content
            summary_html_parts = [
                "🌍 <strong>Environment Summary:</strong>"
            ]
            
            # Add indented summary lines
            for line in summary_lines:
                summary_html_parts.append(f"   • {line}")
            
            # Add recommendations jika ada
            if recommendations:
                summary_html_parts.append("<br><strong>💡 Recommendations:</strong>")
                for emoji, recommendation in recommendations[:3]:  # Limit ke 3 recommendations
                    summary_html_parts.append(f"   • {emoji} {recommendation}")
            
            # Join dan update panel
            summary_html = "<br>".join(summary_html_parts)
            
            if 'env_summary_panel' in self.ui_components:
                self.ui_components['env_summary_panel'].value = f"""
                <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                           border-left: 4px solid #17a2b8; border-radius: 4px; margin: 10px 0;
                           font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.4;">
                    {summary_html}
                </div>
                """
            
            # Log summary info yang penting tanpa detail berlebihan
            self._log_summary_highlights_minimal(merged_info)
            
        except Exception as e:
            # Fallback ke basic summary jika enhanced gagal
            self._display_basic_environment_summary()
    
    def _log_summary_highlights_minimal(self, enhanced_info: Dict[str, Any]):
        """Log highlight minimal dari system info"""
        try:
            if not hasattr(self, 'logger') or self.logger is None:
                return
            
            # Hanya log info critical yang user perlu tahu
            env_type = enhanced_info.get('environment', 'Unknown')
            
            # Single summary log
            summary_parts = [env_type]
            
            if enhanced_info.get('cuda_available'):
                summary_parts.append("GPU✅")
            else:
                summary_parts.append("CPU-only")
            
            if 'available_memory_gb' in enhanced_info:
                memory_gb = enhanced_info['available_memory_gb']
                summary_parts.append(f"{memory_gb:.1f}GB RAM")
            
            self.logger.info(f"📊 {' | '.join(summary_parts)}")
            
        except Exception:
            # Completely silent jika error
            pass
    
    def _display_basic_environment_summary(self):
        """Fallback basic environment summary jika enhanced gagal"""
        try:
            system_info = self.env_manager.get_system_info() if hasattr(self, 'env_manager') else {}
            
            basic_summary = [
                "🌍 <strong>Environment Summary (Basic):</strong>",
                f"   • 🏠 Platform: {system_info.get('environment', 'Unknown')}",
                f"   • 🐍 Python: {system_info.get('python_version', 'N/A')}",
                f"   • 🎮 GPU: {'✅ Available' if system_info.get('cuda_available') else '❌ CPU Only'}"
            ]
            
            basic_html = "<br>".join(basic_summary)
            
            if 'env_summary_panel' in self.ui_components:
                self.ui_components['env_summary_panel'].value = f"""
                <div style="padding: 12px; background-color: #fff3cd; color: #856404; 
                           border-left: 4px solid #ffc107; border-radius: 4px; margin: 10px 0;">
                    {basic_html}
                    <br><br><em>⚠️ Detailed system info tidak tersedia</em>
                </div>
                """
                
        except Exception as e:
            self.logger.warning(f"⚠️ Basic summary error: {str(e)}")
    
    def _log_summary_highlights(self, enhanced_info: Dict[str, Any]):
        """Log highlight dari system info yang penting"""
        try:
            # Log environment type
            env_type = enhanced_info.get('environment', 'Unknown')
            self.logger.info(f"🌍 Environment: {env_type}")
            
            # Log critical resources
            if 'available_memory_gb' in enhanced_info:
                memory_gb = enhanced_info['available_memory_gb']
                memory_status = "🟢 Optimal" if memory_gb > 8 else "🟡 Limited" if memory_gb > 4 else "🔴 Low"
                self.logger.info(f"🧠 Memory: {memory_status} ({memory_gb:.1f}GB)")
            
            # Log GPU status
            if enhanced_info.get('cuda_available'):
                gpu_name = enhanced_info.get('cuda_device_name', 'GPU')[:25]
                self.logger.info(f"🎮 GPU: ✅ {gpu_name}")
            else:
                self.logger.info("🎮 GPU: ❌ CPU-only mode")
            
            # Log storage status
            if 'disk_free_gb' in enhanced_info:
                free_gb = enhanced_info['disk_free_gb']
                storage_status = "🟢 Plenty" if free_gb > 20 else "🟡 Limited" if free_gb > 10 else "🔴 Low"
                self.logger.info(f"💾 Storage: {storage_status} ({free_gb:.1f}GB free)")
            
            # Log critical missing packages
            missing_packages = enhanced_info.get('missing_packages', [])
            if missing_packages:
                critical_missing = [pkg for pkg in missing_packages if pkg in ['PyTorch', 'YOLO', 'OpenCV']]
                if critical_missing:
                    self.logger.warning(f"📦 Missing critical: {', '.join(critical_missing)}")
            
        except Exception as e:
            self.logger.debug(f"🔍 Log highlights error: {str(e)}")
    
    def _get_requirements_status(self) -> list:
        """Get status requirements dan dependencies yang penting"""
        status_parts = []
        
        try:
            # Check critical packages
            critical_packages = {
                'torch': '🔥 PyTorch',
                'torchvision': '👁️ TorchVision', 
                'ultralytics': '🎯 YOLO',
                'albumentations': '🎨 Augmentation',
                'roboflow': '📊 Dataset',
                'yaml': '⚙️ Config',
                'tqdm': '📊 Progress'
            }
            
            available_packages = []
            missing_packages = []
            
            for package, display_name in critical_packages.items():
                try:
                    __import__(package)
                    available_packages.append(display_name)
                except ImportError:
                    missing_packages.append(display_name)
            
            # Display available packages
            if available_packages:
                packages_str = ", ".join(available_packages[:4])  # Limit to 4 untuk readability
                if len(available_packages) > 4:
                    packages_str += f" (+{len(available_packages)-4} lainnya)"
                status_parts.append(f"   • 📦 <span style='color:#4CAF50'><strong>Packages:</strong></span> {packages_str}")
            
            # Display missing packages jika ada
            if missing_packages:
                missing_str = ", ".join(missing_packages[:3])
                status_parts.append(f"   • ⚠️ <span style='color:#FF9800'><strong>Missing:</strong></span> {missing_str}")
            
        except Exception:
            status_parts.append("   • 📦 <span style='color:#6c757d'><strong>Packages:</strong></span> Status tidak dapat diperiksa")
        
        # Check disk space jika bisa
        try:
            import shutil
            if self.env_manager.is_colab:
                free_space = shutil.disk_usage('/content').free / (1024**3)  # GB
                space_color = "#4CAF50" if free_space > 10 else "#FF9800" if free_space > 5 else "#F44336"
                status_parts.append(f"   • 💾 <span style='color:{space_color}'><strong>Disk Space:</strong></span> {free_space:.1f}GB available")
        except Exception:
            pass
        
        return status_parts
    
    def _handle_setup_click(self, button):
        """Handle setup button dengan proper state management dan Drive refresh"""
        button.disabled = True
        
        try:
            self._reset_ui_state()
            
            # Initialize orchestrator logger jika belum
            if hasattr(self.orchestrator, 'init_logger'):
                self.orchestrator.init_logger()
            
            # Perform setup dengan proper state management
            success = self.orchestrator.perform_environment_setup()
            
            if success:
                self.setup_completed = True
                self._update_status("✅ Environment siap digunakan", "success")
                
                # Refresh environment summary setelah setup berhasil
                self._display_environment_summary()
                
                # Button tetap disabled, progress tersembunyi
            else:
                button.disabled = False
                self._update_status("❌ Setup gagal - Coba lagi", "error")
                self._show_progress()
                
        except Exception as e:
            self.logger.error(f"❌ Error setup: {str(e)}")
            button.disabled = False
            self._update_status(f"❌ Error: {str(e)}", "error")
            self._show_progress()
    
    def _reset_ui_state(self):
        """Reset UI state sebelum setup"""
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
        self._show_progress()
    
    def _show_progress(self):
        """Tampilkan progress bar"""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'visible'
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def display(self):
        """Display UI dengan delayed initialization untuk menghindari premature logs"""
        # Display UI terlebih dahulu tanpa logging
        display(self.ui_components['ui_layout'])
        
        # Setup logger SETELAH UI displayed
        self.logger = setup_ipython_logging(
            self.ui_components,
            ENV_CONFIG_LOGGER_NAMESPACE,
            redirect_all_logs=False
        )
        
        # Display environment summary setelah UI ready
        self._display_environment_summary()
        
        try:
            # Check status dengan retry mechanism untuk Drive state
            env_status = self._check_environment_status_with_retry()
            
            if env_status.get('ready', False):
                self.setup_completed = True
                self.ui_components['setup_button'].disabled = True
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
                # Sembunyikan progress jika sudah ready
                if 'progress_container' in self.ui_components:
                    self.ui_components['progress_container'].layout.visibility = 'hidden'
            else:
                # Tampilkan apa yang perlu di-setup dengan prioritas
                missing_items = self._get_prioritized_missing_items(env_status)
                
                if missing_items:
                    items_str = ', '.join(missing_items[:3])  # Limit display
                    self._update_status(f"🔧 Perlu setup: {items_str}...", "warning")
                else:
                    self._update_status("🔧 Environment perlu dikonfigurasi", "info")
                
                self.ui_components['setup_button'].disabled = False
                self._show_progress()
            
        except Exception as e:
            # Jangan crash UI, tapi beri info
            self._update_status("⚠️ Status check error - Silakan setup", "warning")
            self.ui_components['setup_button'].disabled = False
            self._show_progress()
    
    def _check_environment_status_with_retry(self, max_retries: int = 3) -> Dict[str, Any]:
        """Check environment status dengan retry untuk memastikan Drive state accuracy"""
        import time
        
        for attempt in range(max_retries):
            try:
                # Refresh environment manager state sebelum check
                if hasattr(self, 'env_manager'):
                    self.env_manager.refresh_drive_status()
                
                # Small delay untuk memastikan state terupdate
                if attempt > 0:
                    time.sleep(1)
                
                # Check status dari orchestrator
                env_status = self.orchestrator.check_environment_status()
                
                # Jika Drive mounted tapi status checker belum detect, retry
                if (hasattr(self, 'env_manager') and 
                    self.env_manager.is_drive_mounted and 
                    not env_status.get('drive', {}).get('mounted', False) and 
                    attempt < max_retries - 1):
                    continue
                
                return env_status
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt, return basic status
                    return {'ready': False, 'error': str(e)}
                continue
        
        return {'ready': False, 'error': 'Status check failed after retries'}
    
    def _get_prioritized_missing_items(self, env_status: Dict[str, Any]) -> list:
        """Get missing items dengan prioritas untuk display yang informatif"""
        missing_items = []
        
        # Prioritas 1: Drive connection
        if not env_status.get('drive', {}).get('mounted', False):
            missing_items.append("Google Drive")
        
        # Prioritas 2: Essential folders (limit ke 2)
        missing_folders = env_status.get('missing_drive_folders', [])[:2]
        missing_items.extend([f"folder {folder}" for folder in missing_folders])
        
        # Prioritas 3: Essential configs (limit ke 2)  
        missing_configs = env_status.get('missing_drive_configs', [])[:2]
        missing_items.extend([f"config {config.replace('_config.yaml', '')}" for config in missing_configs])
        
        return missing_items


def create_env_config_component() -> EnvConfigComponent:
    """Factory function untuk membuat component dengan environment summary"""
    return EnvConfigComponent()