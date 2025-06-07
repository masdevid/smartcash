"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Handler untuk download dataset yang fokus pada fungsionalitas download dengan delegasi ke handler lain
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.dialogs import confirm
from smartcash.dataset.downloader import get_downloader_instance, create_ui_compatible_config
from smartcash.ui.dataset.downloader.handlers.button_handler import ButtonHandler
from smartcash.ui.dataset.downloader.handlers.check_handler import DatasetCheckHandler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import DatasetCleanupHandler
from smartcash.common.environment import get_environment_manager


class DownloadHandler(ButtonHandler):
    """
    Handler untuk download dataset yang mewarisi dari ButtonHandler
    dengan fokus pada fungsionalitas download dataset
    """

    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger):
        super().__init__(ui_components, logger)
        self.config = config

        button_configs = [
            ('download_button', self._handle_download_click),
            ('check_button', self._handle_check_click),
            ('cleanup_button', self._handle_cleanup_click),
            ('save_button', self.handle_save_click),
            ('reset_button', self.handle_reset_click)
        ]
        self.register_button_configs(button_configs)

    def _handle_download_click(self, button) -> None:
        try:
            self._prepare_button_state(button)

            if self.progress_tracker:
                self.progress_tracker.show("Dataset Download")
                self.progress_tracker.update_overall(10, "🔧 Memvalidasi konfigurasi...")

            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self._handle_error("Config handler tidak tersedia", button)
                return

            ui_config = config_handler.extract_config_from_ui(self.ui_components)
            service_config = create_ui_compatible_config(ui_config)

            validation = config_handler.validate_config(ui_config)
            if not validation['valid']:
                error_msg = f"Config tidak valid: {'; '.join(validation['errors'])}"
                self._handle_error(error_msg, button)
                return

            if self.progress_tracker:
                self.progress_tracker.update_overall(30, "✅ Configuration valid")

            has_existing = self._check_existing_dataset_quick()

            if has_existing:
                self._show_download_confirmation(service_config, button)
            else:
                self._execute_download(service_config, button)

        except Exception as e:
            self._handle_error(f"Error download handler: {str(e)}", button)

    def _execute_download(self, service_config: Dict[str, Any], button) -> None:
        try:
            self.logger.info("🔧 Konfigurasi service download:")
            for key, value in service_config.items():
                if key == 'api_key':
                    masked_key = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '****'
                    self.logger.info(f"🔑 {key}: {masked_key}")
                else:
                    self.logger.info(f"🔧 {key}: {value}")

            required = ['workspace', 'project', 'version', 'api_key']
            missing = [f for f in required if not service_config.get(f, '').strip()]

            if missing:
                self._handle_error(f"Field wajib kosong: {', '.join(missing)}", button)
                return

            self.logger.info(f"🔍 Workspace: {service_config.get('workspace')}")
            self.logger.info(f"🔍 Project: {service_config.get('project')}")
            self.logger.info(f"🔍 Version: {service_config.get('version')}")
            self.logger.info(f"🔍 Output format: {service_config.get('output_format')}")

            if self.progress_tracker:
                self.progress_tracker.update_overall(40, "🏭 Creating download service...")

            self.logger.info(f"🔧 Konfigurasi: rename_files={service_config.get('rename_files')}, organize_dataset={service_config.get('organize_dataset')}, validate_download={service_config.get('validate_download')}")

            try:
                env_manager = get_environment_manager()
                self.logger.info(f"📂 Dataset path: {env_manager.get_dataset_path()}")
            except Exception as e:
                self.logger.warning(f"⚠️ Tidak dapat mengambil info environment: {str(e)}")

            self.logger.info("🔧 Membuat download service...")
            downloader = get_downloader_instance(service_config, self.logger)
            if not downloader:
                self._handle_error("Gagal membuat download service", button)
                return

            self.logger.info("✅ Download service berhasil dibuat")

            try:
                if hasattr(downloader, 'set_progress_callback'):
                    progress_callback = self._create_progress_callback()
                    downloader.set_progress_callback(progress_callback)
                    self.logger.info("✅ Progress callback berhasil diatur")
                else:
                    self.logger.warning("⚠️ Service tidak mendukung progress callback")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal mengatur progress callback: {str(e)}")

            if self.progress_tracker:
                self.progress_tracker.update_overall(40, "📥 Memulai proses download...")

            self.logger.info("🚀 Memulai proses download dataset...")
            download_exec_start = time.time()
            try:
                result = downloader.download_dataset()
                download_exec_time = time.time() - download_exec_start
                self.logger.info(f"⏱️ Waktu eksekusi download: {download_exec_time:.2f} detik")
            except Exception as e:
                download_exec_time = time.time() - download_exec_start
                error_msg = f"Error saat download: {str(e)} (setelah {download_exec_time:.2f} detik)"
                self.logger.error(f"❌ {error_msg}")
                if self.progress_tracker:
                    self.progress_tracker.error(error_msg)
                raise Exception(error_msg)

            # Log hasil download
            self.logger.info(f"📊 Download result keys: {list(result.keys()) if result else 'None'}")
            if result and 'status' in result:
                self.logger.info(f"📊 Download status: {result.get('status')}")
            if result and 'message' in result:
                self.logger.info(f"📊 Download message: {result.get('message')}")

            # Handle response
            if result and result.get('status') == 'success':
                stats = result.get('stats', {})
                total_images = stats.get('total_images', 0)
                success_msg = f"Dataset berhasil didownload: {total_images:,} gambar"

                # Log detail hasil download dengan format yang lebih jelas
                self.logger.info(f"📊 Total gambar: {total_images:,} gambar")
                self.logger.info(f"📊 Total label: {stats.get('total_labels', 0):,} label")

                # Log detail per split dengan statistik lebih lengkap
                splits = stats.get('splits', {})
                self.logger.info("📊 Detail per split:")
                for split_name, split_stats in splits.items():
                    img_count = split_stats.get('images', 0)
                    label_count = split_stats.get('labels', 0)
                    img_percent = (img_count / total_images * 100) if total_images > 0 else 0
                    self.logger.info(f"📊 {split_name}: {img_count} gambar ({img_percent:.1f}%), {label_count} label")

                # Log statistik classes jika tersedia
                if 'classes' in stats:
                    self.logger.info(f"📊 Classes detected: {stats.get('classes')}")

                if self.progress_tracker:
                    self.progress_tracker.complete(success_msg)

                show_status_safe(success_msg, "success", self.ui_components)
                self.logger.success(f"✅ {success_msg}")

                # Log additional stats jika tersedia dengan detail lebih lengkap
                if stats.get('uuid_renamed'):
                    naming_stats = stats.get('naming_stats', {})
                    if naming_stats:
                        total_renamed = naming_stats.get('total_files', 0)
                        self.logger.info(f"🔄 UUID renaming: {total_renamed} files processed")

                        # Log detail renaming stats jika tersedia
                        for key, value in naming_stats.items():
                            if key != 'total_files':
                                self.logger.info(f"🔄 {key}: {value}")

                # Log output directory dengan verifikasi
                output_dir = result.get('output_dir', '')
                if output_dir:
                    output_path = Path(output_dir)
                    exists = output_path.exists()
                    self.logger.info(f"📂 Output directory: {output_dir} ({'exists' if exists else 'not found'})")

                    # Log informasi output directory
                    if exists:
                        try:
                            files = list(output_path.glob('*'))
                            subdirs = [f for f in files if f.is_dir()]
                            self.logger.info(f"📂 Output berisi {len(files)} item, {len(subdirs)} direktori")

                            # Log beberapa subdirectory jika ada
                            if subdirs:
                                subdir_names = [d.name for d in subdirs[:5]]
                                self.logger.info(f"📂 Subdirektori: {', '.join(subdir_names)}{' dan lainnya...' if len(subdirs) > 5 else ''}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Tidak dapat scan output directory: {str(e)}")

                # Log total waktu eksekusi
                total_time = time.time() - download_start_time
                self.logger.info(f"⏱️ Total waktu eksekusi: {total_time:.2f} detik")

            else:
                error_msg = f"Download gagal: {result.get('message', 'Unknown error') if result else 'No response from service'}"
                self._handle_error(error_msg, button)

        except Exception as e:
            import traceback
            self.logger.error(f"❌ Error saat download: {str(e)}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self._handle_error(f"Error saat download: {str(e)}", button)
        finally:
            self._restore_button_state()

    def _show_download_confirmation(self, service_config: Dict[str, Any], button) -> None:
        """Show confirmation dialog"""
        workspace = service_config.get('workspace', '')
        project = service_config.get('project', '')
        version = service_config.get('version', '')

        dataset_id = f"{workspace}/{project}:v{version}"

        message = f"Dataset existing akan ditimpa!\n\n"
        message += f"🎯 Target: {dataset_id}\n"
        message += f"🔄 UUID Renaming: {'✅' if service_config.get('rename_files', True) else '❌'}\n"
        message += f"✅ Validasi: {'✅' if service_config.get('validate_download', True) else '❌'}\n"
        message += f"💾 Backup: {'✅' if service_config.get('backup_existing', False) else '❌'}\n\n"
        message += f"Lanjutkan download?"

        confirm(
            "Konfirmasi Download Dataset",
            message,
            on_yes=lambda btn: self._execute_download(service_config, button),
            on_no=lambda btn: (
                self.logger.info("🚫 Download dibatalkan"),
                self._restore_button_state()
            )
        )

    def _handle_check_click(self, button) -> None:
        """Handle check button click dengan delegasi ke DatasetCheckHandler"""
        try:
            self._prepare_button_state(button)
            
            # Ambil handler dari ui_components jika sudah ada
            check_handler = self.ui_components.get('check_handler')
            if not check_handler:
                check_handler = DatasetCheckHandler(self.ui_components, self.config, self.logger)
                
            check_handler.check_dataset(button)
            
        except Exception as e:
            self._handle_error(f"Error saat memeriksa dataset: {str(e)}", button)
        finally:
            self._restore_button_state()
            
    def _handle_cleanup_click(self, button) -> None:
        """Handle cleanup button click dengan delegasi ke DatasetCleanupHandler"""
        try:
            self._prepare_button_state(button)

            # Ambil handler dari ui_components jika sudah ada
            cleanup_handler = self.ui_components.get('cleanup_handler')
            if not cleanup_handler:
                cleanup_handler = DatasetCleanupHandler(self.ui_components, self.config, self.logger)

            # Gunakan metode helper dari ButtonHandler untuk menjalankan handler async
            self.run_async_handler(cleanup_handler.cleanup_dataset, button)

        except Exception as e:
            self._handle_error(f"Error saat cleanup: {str(e)}", button)
        finally:
            self._restore_button_state()

    def _check_existing_dataset_quick(self) -> bool:
        """Memeriksa apakah dataset sudah ada dengan cepat"""
        try:
            env_manager = get_environment_manager()
            dataset_path = env_manager.get_dataset_path()
            return dataset_path.exists() and any(dataset_path.iterdir())
        except Exception:
            return False

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan proper error handling"""
    logger = ui_components.get('logger')

    try:
        # Buat handler-handler yang diperlukan
        download_handler = DownloadHandler(ui_components, config, logger)
        check_handler = DatasetCheckHandler(ui_components, config, logger)
        cleanup_handler = DatasetCleanupHandler(ui_components, config, logger)

        # Setup handlers
        ui_components = download_handler.setup_handlers()

        # Tambahkan referensi handler ke ui_components
        ui_components.update({
            'download_handler': download_handler,
            'check_handler': check_handler,
            'cleanup_handler': cleanup_handler,
            'handler': download_handler  # Referensi umum
        })

        logger.success("✅ Download handlers berhasil dikonfigurasi")
        return ui_components

    except Exception as e:
        logger.error(f"❌ Error setup handlers: {str(e)}")
        return ui_components


# Export
__all__ = ['setup_download_handlers', 'DownloadHandler']