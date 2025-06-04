"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Complete downloader initializer dengan proper config inheritance dan error handling
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.components.main_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_download_parameters
from smartcash.ui.dataset.downloader.handlers.progress_handler import setup_progress_handlers

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan CommonInitializer pattern dan config inheritance."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, parent_module='dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan responsive layout."""
        try:
            return create_downloader_ui(config, env)
        except Exception as e:
            self.logger.error(f"❌ UI creation error: {str(e)}")
            # Fallback UI
            import ipywidgets as widgets
            return {
                'ui': widgets.VBox([
                    widgets.HTML(f"<h3>⚠️ Downloader (Error Mode)</h3>"),
                    widgets.HTML(f"<p style='color:red;'>Error: {str(e)}</p>")
                ])
            }
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan progress callback integration."""
        results = {}
        
        # Validasi komponen yang diperlukan
        required_components = ['download_button', 'validate_button', 'workspace_field', 'project_field', 'version_field', 'api_key_field']
        missing_components = [comp for comp in required_components if comp not in ui_components]
        
        if missing_components:
            error_msg = f"Komponen tidak ditemukan: {', '.join(missing_components)}"
            self.logger.error(f"❌ {error_msg}")
            # Tidak perlu fallback, jika komponen tidak ada maka handler tidak akan disetup
            # Ini akan mencegah error di kemudian hari
            return {'error': error_msg, 'missing_components': missing_components}
        
        # Pastikan logger tersedia untuk digunakan oleh handler
        if 'logger' not in ui_components:
            ui_components['logger'] = self.logger
        
        # Setup progress handlers dengan callback
        try:
            progress_result = setup_progress_handlers(ui_components)
            if 'error' in progress_result:
                self.logger.warning(f"⚠️ Progress handler setup warning: {progress_result['error']}")
            else:
                results.update(progress_result)
                self.logger.info("✅ Progress handlers berhasil diinisialisasi")
        except Exception as e:
            self.logger.warning(f"⚠️ Progress handler setup warning: {str(e)}")
        
        # Setup download handlers dengan validation dan progress
        try:
            download_result = setup_download_handlers(ui_components, env, config)
            if 'error' in download_result:
                self.logger.warning(f"⚠️ Download handler setup warning: {download_result['error']}")
            else:
                results.update(download_result)
                self.logger.info("✅ Download handlers berhasil diinisialisasi")
        except Exception as e:
            self.logger.warning(f"⚠️ Download handler setup warning: {str(e)}")
        
        # Setup initial validation jika diminta
        if config.get('auto_validate', False):
            try:
                validation_result = validate_download_parameters(ui_components, include_api_test=False)
                if validation_result['valid']:
                    self.logger.info("✅ Initial validation passed")
                    results['validation_result'] = validation_result
                else:
                    self.logger.warning(f"⚠️ Initial validation issues: {validation_result.get('message', '')}")
            except Exception as e:
                self.logger.warning(f"⚠️ Initial validation error: {str(e)}")
        
        if 'error' not in results:
            self.logger.success("✅ Dataset downloader handlers berhasil diinisialisasi")
        
        return results
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py dengan inheritance support."""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
            return DEFAULT_CONFIG.copy()
        except ImportError:
            self.logger.warning("⚠️ defaults.py tidak ditemukan, menggunakan fallback config")
            return self._fallback_config()
    
    def _fallback_config(self) -> Dict[str, Any]:
        """Fallback config jika defaults.py tidak ada."""
        return {
            '_base_': ['base_config'],  # Inherit dari base_config.yaml
            
            # Dataset identification
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'api_key': '',
            
            # Download options
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False,
            
            # Progress options
            'progress_enabled': True,
            'show_detailed_progress': False,
            
            # Performance options
            'retry_attempts': 3,
            'timeout_seconds': 30,
            'chunk_size_kb': 8,
            
            # Metadata
            'module_name': 'downloader',
            'version': '1.0.0',
            'created_by': 'SmartCash Dataset Downloader'
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada."""
        return [
            'ui', 'main_container', 'header'
        ]
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation dengan downloader-specific checks."""
        # Base validation
        base_result = super()._validate_setup(ui_components)
        
        if not base_result['valid']:
            return base_result
        
        # Downloader-specific validation
        required_components = {
            'form': ['workspace_field', 'project_field', 'version_field', 'api_key_field'],
            'buttons': ['download_button', 'validate_button'],
            'progress': ['progress_bar', 'status_panel']
        }
        
        missing_components = {}
        for category, components in required_components.items():
            missing = [comp for comp in components if comp not in ui_components]
            if missing:
                missing_components[category] = missing
        
        if missing_components:
            # Jika komponen form yang hilang, ini adalah error kritis
            if 'form' in missing_components and len(missing_components['form']) > 0:
                critical_missing = missing_components['form']
                error_msg = f"Komponen form kritis tidak ditemukan: {', '.join(critical_missing)}"
                self.logger.error(f"❌ {error_msg}")
                return {'valid': False, 'message': error_msg, 'missing_components': missing_components}
            
            # Jika komponen button yang hilang, ini adalah error kritis
            if 'buttons' in missing_components and 'download_button' in missing_components['buttons']:
                error_msg = "Tombol download tidak ditemukan, UI tidak akan berfungsi dengan benar"
                self.logger.error(f"❌ {error_msg}")
                return {'valid': False, 'message': error_msg, 'missing_components': missing_components}
            
            # Untuk komponen progress, hanya berikan warning
            if 'progress' in missing_components:
                self.logger.warning(f"⚠️ Komponen progress tidak ditemukan: {', '.join(missing_components['progress'])}. UI mungkin tidak menampilkan progress dengan benar.")
        
        # Pastikan logger tersedia untuk semua handler
        if 'logger' not in ui_components:
            ui_components['logger'] = self.logger
            self.logger.info("📝 Logger ditambahkan ke UI components")
        
        return {'valid': True, 'message': "Downloader validation passed"}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Enhanced finalize dengan downloader-specific setup."""
        super()._finalize_setup(ui_components, config)
        
        # Pastikan logger tersedia untuk semua handler
        if 'logger' not in ui_components:
            ui_components['logger'] = self.logger
        
        # Auto-detect API key jika tersedia
        try:
            from smartcash.ui.dataset.downloader.handlers.config_updater import DownloaderConfigUpdater
            
            # Validasi komponen yang diperlukan sebelum update
            required_fields = ['api_key_field']
            missing_fields = [field for field in required_fields if field not in ui_components]
            
            if not missing_fields:
                DownloaderConfigUpdater.update_ui_from_environment(ui_components)
                self.logger.info("🔑 API key berhasil dideteksi dari environment")
            else:
                self.logger.warning(f"⚠️ Tidak dapat mendeteksi API key: komponen {', '.join(missing_fields)} tidak ditemukan")
        except ImportError:
            self.logger.warning("⚠️ Module config_updater tidak ditemukan, melewati deteksi API key")
        except Exception as e:
            self.logger.warning(f"⚠️ Environment update warning: {str(e)}")
        
        # Set downloader-specific metadata
        ui_components.update({
            'downloader_version': '1.0.0',
            'supported_formats': ['yolov5pytorch', 'yolov8', 'coco', 'createml'],
            'api_provider': 'roboflow'
        })
        
        # Pastikan format dropdown memiliki nilai yang benar
        if 'format_dropdown' in ui_components and hasattr(ui_components['format_dropdown'], 'options'):
            try:
                ui_components['format_dropdown'].options = ui_components['supported_formats']
                self.logger.info("📝 Format dropdown diperbarui dengan format yang didukung")
            except Exception as e:
                self.logger.warning(f"⚠️ Tidak dapat memperbarui format dropdown: {str(e)}")
        
        self.logger.success("✅ Finalisasi setup downloader berhasil")

# Singleton instance dan public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Public API untuk initialize downloader UI dengan comprehensive error handling."""
    from smartcash.common.logger import get_logger
    logger = get_logger('downloader.initializer')
    
    try:
        # Validasi config jika disediakan
        if config is not None:
            from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
            validation = DownloaderConfigExtractor.validate_extracted_config(config)
            if not validation.get('valid', False):
                errors = validation.get('errors', [])
                error_msg = f"Konfigurasi tidak valid: {'; '.join(errors)}"
                logger.error(f"❌ {error_msg}")
                # Tetap lanjutkan dengan config default
                logger.info("💡 Menggunakan konfigurasi default sebagai fallback")
                config = None
        
        # Inisialisasi UI
        result = _downloader_initializer.initialize(env=env, config=config, **kwargs)
        
        # Validasi hasil inisialisasi
        if not isinstance(result, dict) or 'ui' not in result:
            logger.error("❌ Hasil inisialisasi tidak valid, tidak mengandung komponen UI")
            raise ValueError("Hasil inisialisasi tidak valid")
        
        # Tambahkan informasi versi
        result['version'] = result.get('downloader_version', '1.0.0')
        
        logger.success("✅ Downloader UI berhasil diinisialisasi")
        return result
    except ImportError as e:
        # Error import menandakan masalah dengan instalasi atau struktur kode
        logger.error(f"❌ Import error: {str(e)}")
        logger.error("❌ Pastikan semua modul yang diperlukan sudah diinstal dan struktur kode benar")
        
        import ipywidgets as widgets
        error_ui = widgets.VBox([
            widgets.HTML("<h3 style='color:red;'>⚠️ Downloader Initialization Failed</h3>"),
            widgets.HTML(f"<p><strong>Import Error:</strong> {str(e)}</p>"),
            widgets.HTML("<p><em>💡 Pastikan semua dependencies sudah diinstal</em></p>")
        ], layout=widgets.Layout(padding='20px', border='1px solid red', border_radius='5px'))
        
        return {'ui': error_ui, 'error': str(e), 'error_type': 'import_error'}
    except Exception as e:
        # Error umum lainnya
        logger.error(f"❌ Critical downloader initialization error: {str(e)}")
        
        import ipywidgets as widgets
        error_ui = widgets.VBox([
            widgets.HTML("<h3 style='color:red;'>⚠️ Downloader Initialization Failed</h3>"),
            widgets.HTML(f"<p><strong>Error:</strong> {str(e)}</p>"),
            widgets.HTML("<p><em>💡 Coba restart cell atau periksa log untuk detail lebih lanjut</em></p>")
        ], layout=widgets.Layout(padding='20px', border='1px solid red', border_radius='5px'))
        
        return {'ui': error_ui, 'error': str(e), 'error_type': 'initialization_error'}

def get_downloader_status():
    """Get downloader status untuk debugging."""
    try:
        return _downloader_initializer.get_module_status()
    except Exception as e:
        return {
            'module_name': 'downloader',
            'initialized': False,
            'error': str(e),
            'status': 'error'
        }

def validate_downloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate downloader config tanpa UI."""
    try:
        from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
        return DownloaderConfigExtractor.validate_extracted_config(config)
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Config validation error: {str(e)}"],
            'warnings': []
        }