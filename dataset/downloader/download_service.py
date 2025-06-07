"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: FIXED download service dengan standardized response format dan progress callback
"""

import time
import requests
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from smartcash.dataset.downloader.base import (
    BaseDownloaderComponent, ValidationHelper, PathHelper, FileHelper
)
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.downloader.file_processor import create_file_processor
from smartcash.dataset.downloader.validators import create_dataset_validator


class DownloadService(BaseDownloaderComponent):
    """FIXED download service dengan UI compatibility dan standardized interface"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
    @property
    def roboflow_client(self):
        if not self._roboflow_client:
            self._roboflow_client = create_roboflow_client(self.config.get('api_key', ''), self.logger)
            if self._progress_callback:
                self._roboflow_client.set_progress_callback(self._create_standardized_callback())
        return self._roboflow_client
    
    @property
    def file_processor(self):
        if not self._file_processor:
            self._file_processor = create_file_processor(self.logger)
            if self._progress_callback:
                self._file_processor.set_progress_callback(self._create_standardized_callback())
        return self._file_processor
    
    @property
    def validator(self):
        if not self._validator:
            self._validator = create_dataset_validator(self.logger)
        return self._validator
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """
        FIXED: Set progress callback dengan signature yang diharapkan UI
        
        Args:
            callback: Function dengan signature (step, current, total, message)
        """
        self._progress_callback = callback
        
        # Update existing components jika sudah diinisialisasi
        if self._roboflow_client:
            self._roboflow_client.set_progress_callback(self._create_standardized_callback())
        if self._file_processor:
            self._file_processor.set_progress_callback(self._create_standardized_callback())
    
    def _create_standardized_callback(self) -> Callable:
        """Create standardized callback wrapper untuk internal components"""
        def standardized_callback(*args):
            if self._progress_callback:
                # Handle berbagai signature dari internal components
                if len(args) == 4:
                    # Already standardized: (step, current, total, message)
                    self._progress_callback(*args)
                elif len(args) == 3:
                    # Format: (step, progress_percent, message)
                    step, progress, message = args
                    self._progress_callback(step, progress, 100, message)
                elif len(args) == 2:
                    # Format: (progress_percent, message)
                    progress, message = args
                    self._progress_callback("progress", progress, 100, message)
        
        return standardized_callback
    
    def download_dataset(self) -> Dict[str, Any]:
        """
        FIXED: Download dataset dengan response format yang diharapkan UI
        
        Returns:
            Dictionary dengan format:
            {
                'status': 'success'|'error',
                'message': str,
                'stats': {...},
                'output_dir': str,
                'duration': float
            }
        """
        start_time = time.time()
        
        try:
            # Validate config
            validation = ValidationHelper.validate_config(
                self.config, ['workspace', 'project', 'version', 'api_key']
            )
            if not validation['valid']:
                return self._create_error_response(f"Config tidak valid: {'; '.join(validation['errors'])}")
            
            # Extract parameters
            params = self._extract_params()
            
            # Setup paths
            paths = PathHelper.setup_download_paths(
                params['workspace'], params['project'], params['version']
            )
            
            # Execute download flow dengan progress tracking
            self._notify_progress("validate", 0, 100, "🔧 Validating configuration...")
            
            metadata = self._get_metadata(params)
            
            if params['backup_existing']:
                self._handle_backup(paths)
            
            self._download_and_extract(metadata['download_url'], paths)
            stats = self._organize_dataset(paths, params)
            
            if params['validate_download']:
                self._validate_results(paths['final_dir'])
            
            # Cleanup
            FileHelper.cleanup_temp(paths['temp_dir'])
            
            # FIXED: Return format yang diharapkan UI
            return {
                'status': 'success',
                'message': f"Dataset berhasil didownload: {stats.get('total_images', 0):,} gambar",
                'stats': {
                    'total_images': stats.get('total_images', 0),
                    'total_labels': stats.get('total_labels', 0),
                    'splits': stats.get('splits', {}),
                    'uuid_renamed': params['rename_files'],
                    'naming_stats': stats.get('naming_stats', {}) if params['rename_files'] else None
                },
                'output_dir': str(paths['final_dir']),
                'duration': time.time() - start_time,
                'metadata': {
                    'workspace': params['workspace'],
                    'project': params['project'],
                    'version': params['version'],
                    'format': params['output_format']
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Download failed: {str(e)}")
            return self._create_error_response(f"Download error: {str(e)}")
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """FIXED: Create error response dengan format yang konsisten"""
        return {
            'status': 'error',
            'message': message,
            'stats': {'total_images': 0, 'total_labels': 0, 'splits': {}},
            'output_dir': '',
            'duration': 0.0
        }
    
    def _extract_params(self) -> Dict[str, Any]:
        """Extract parameters dengan proper defaults"""
        return {
            'workspace': self.config.get('workspace', ''),
            'project': self.config.get('project', ''),
            'version': self.config.get('version', ''),
            'api_key': self.config.get('api_key', ''),
            'output_format': self.config.get('output_format', 'yolov5pytorch'),
            'validate_download': self.config.get('validate_download', True),
            'backup_existing': self.config.get('backup_existing', False),
            'rename_files': self.config.get('rename_files', True)  # FIXED: Always boolean
        }
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata dengan progress notification"""
        self._notify_progress('metadata', 10, 100, "📊 Getting metadata...")
        
        result = self.roboflow_client.get_dataset_metadata(
            params['workspace'], params['project'], 
            params['version'], params['output_format']
        )
        
        if result['status'] != 'success':
            raise Exception(f"Metadata failed: {result['message']}")
        
        self._notify_progress('metadata', 30, 100, "✅ Metadata obtained")
        return result
    
    def _handle_backup(self, paths: Dict[str, Path]) -> None:
        """Handle backup dengan progress notification"""
        if self._has_existing_dataset(paths['final_dir']):
            self._notify_progress('backup', 35, 100, "💾 Creating backup...")
            self.logger.info("💾 Creating backup...")
            
            backup_result = FileHelper.backup_directory(paths['final_dir'])
            
            if backup_result['success']:
                self.logger.info(f"✅ Backup: {backup_result['backup_path']}")
            else:
                self.logger.warning(f"⚠️ Backup failed: {backup_result['message']}")
    
    def _download_and_extract(self, download_url: str, paths: Dict[str, Path]) -> None:
        """Download dan extract dengan progress tracking dan detailed logging per step"""
        # Log download start dengan detail
        self.logger.info(f"📥 Memulai download dataset dari URL: {download_url}")
        self.logger.info(f"📥 Target file: {paths['temp_zip']}")
        
        # Download (40-70%) dengan notifikasi per step
        self._notify_progress('download_start', 40, 100, "📥 Memulai proses download...")
        self.logger.info("📥 Menjalankan download via Roboflow client...")
        
        # Implementasi download dengan notifikasi per step
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            last_percent = 0
            
            paths['temp_zip'].parent.mkdir(parents=True, exist_ok=True)
            
            with open(paths['temp_zip'], 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            # Hitung progress dalam persen
                            current_percent = int((downloaded / total_size) * 100)
                            
                            # Notifikasi setiap 5% kemajuan
                            if current_percent >= last_percent + 5 or current_percent == 100:
                                last_percent = current_percent
                                progress_overall = 40 + int((current_percent / 100) * 30)  # Map ke 40-70% overall
                                self._notify_progress(
                                    'download_progress', 
                                    progress_overall, 
                                    100, 
                                    f"📥 Downloading: {current_percent}% ({downloaded/1048576:.1f}/{total_size/1048576:.1f} MB)"
                                )
            
            download_result = {'status': 'success'}
            file_size_mb = paths['temp_zip'].stat().st_size / (1024 * 1024)
            self.logger.success(f"✅ Download berhasil: {file_size_mb:.2f} MB")
            self._notify_progress('download_complete', 70, 100, f"✅ Download selesai: {file_size_mb:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"❌ Download gagal: {str(e)}")
            raise Exception(f"Download failed: {str(e)}")
        
        # Extract (70-80%) dengan notifikasi per step
        self.logger.info(f"📦 Mengekstrak file dari {paths['temp_zip']} ke {paths['extract_dir']}")
        self._notify_progress('extract_start', 70, 100, "📦 Memulai ekstraksi file...")
        
        try:
            import zipfile
            import os
            
            # Pastikan direktori ekstrak ada
            paths['extract_dir'].mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(paths['temp_zip'], 'r') as zip_ref:
                # Dapatkan total file untuk progress tracking
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                extracted_files = 0
                last_percent = 0
                
                # Ekstrak file satu per satu dengan progress tracking
                for file in file_list:
                    zip_ref.extract(file, paths['extract_dir'])
                    extracted_files += 1
                    
                    # Hitung progress dalam persen
                    current_percent = int((extracted_files / total_files) * 100)
                    
                    # Notifikasi setiap 10% kemajuan
                    if current_percent >= last_percent + 10 or current_percent == 100:
                        last_percent = current_percent
                        progress_overall = 70 + int((current_percent / 100) * 10)  # Map ke 70-80% overall
                        self._notify_progress(
                            'extract_progress', 
                            progress_overall, 
                            100, 
                            f"📦 Extracting: {current_percent}% ({extracted_files}/{total_files} files)"
                        )
            
            extract_result = {'status': 'success', 'file_count': total_files}
            self.logger.success(f"✅ Ekstraksi berhasil: {total_files} file")
            self._notify_progress('extract_complete', 80, 100, f"✅ Ekstraksi selesai: {total_files} file")
            
        except Exception as e:
            self.logger.error(f"❌ Ekstraksi gagal: {str(e)}")
            raise Exception(f"Extract failed: {str(e)}")
    
    def _organize_dataset(self, paths: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Organize dataset dengan progress tracking dan notifikasi per step"""
        # Notifikasi awal proses pengorganisasian
        self._notify_progress('organize_start', 80, 100, "🗂️ Memulai pengorganisasian dataset...")
        
        try:
            # Scan direktori ekstrak untuk mendapatkan informasi file
            self._notify_progress('organize_scan', 81, 100, "🔍 Scanning files...")
            
            # Hitung jumlah file di direktori ekstrak
            image_files = list(paths['extract_dir'].glob('**/*.jpg')) + list(paths['extract_dir'].glob('**/*.jpeg')) + \
                         list(paths['extract_dir'].glob('**/*.png'))
            label_files = list(paths['extract_dir'].glob('**/*.txt'))
            yaml_files = list(paths['extract_dir'].glob('**/*.yaml'))
            
            total_files = len(image_files) + len(label_files) + len(yaml_files)
            self._notify_progress(
                'organize_scan_complete', 
                82, 
                100, 
                f"🔍 Scan selesai: {len(image_files)} gambar, {len(label_files)} label, {len(yaml_files)} yaml"
            )
            
            # Notifikasi proses pengorganisasian
            if params['rename_files']:
                self._notify_progress('organize_rename', 83, 100, "🔄 Mengorganisasi dataset dengan UUID renaming...")
                self.logger.info(f"🔄 Mengorganisasi {total_files} file dengan UUID renaming...")
            else:
                self._notify_progress('organize_copy', 83, 100, "💾 Mengorganisasi dataset dengan simple copy...")
                self.logger.info(f"💾 Mengorganisasi {total_files} file dengan simple copy...")
            
            # Proses pengorganisasian dengan progress tracking
            processed_files = 0
            last_percent = 0
            
            # Definisikan callback untuk progress tracking
            def organize_progress_callback(file_path, is_copied):
                nonlocal processed_files, last_percent
                processed_files += 1
                
                # Hitung progress dalam persen
                if total_files > 0:
                    current_percent = int((processed_files / total_files) * 100)
                    
                    # Notifikasi setiap 10% kemajuan
                    if current_percent >= last_percent + 10 or current_percent == 100 or processed_files <= 5:
                        last_percent = current_percent
                        progress_overall = 83 + int((current_percent / 100) * 12)  # Map ke 83-95% overall
                        self._notify_progress(
                            'organize_progress', 
                            progress_overall, 
                            100, 
                            f"🗂️ Organizing: {current_percent}% ({processed_files}/{total_files} files)"
                        )
            
            # Set callback ke file processor jika memungkinkan
            if hasattr(self.file_processor, 'set_file_callback'):
                self.file_processor.set_file_callback(organize_progress_callback)
            
            # Jalankan pengorganisasian
            if params['rename_files']:
                result = self.file_processor.organize_dataset_with_renaming(
                    paths['extract_dir'], paths['final_dir']
                )
            else:
                result = self.file_processor.organize_dataset(
                    paths['extract_dir'], paths['final_dir']
                )
            
            # Reset callback
            if hasattr(self.file_processor, 'set_file_callback'):
                self.file_processor.set_file_callback(None)
            
            # Validasi hasil
            if result['status'] != 'success':
                self._notify_progress('organize_failed', 85, 100, f"❌ Pengorganisasian gagal: {result['message']}")
                raise Exception(f"Organization failed: {result['message']}")
            
            # Notifikasi selesai
            stats = {
                'total_images': result.get('total_images', 0),
                'total_labels': result.get('total_labels', 0),
                'splits': result.get('splits', {})
            }
            
            # Notifikasi statistik per split
            splits = stats.get('splits', {})
            for split_name, split_stats in splits.items():
                self._notify_progress(
                    f'organize_split_{split_name}', 
                    90, 
                    100, 
                    f"📁 {split_name}: {split_stats.get('images', 0)} gambar, {split_stats.get('labels', 0)} label"
                )
            
            # Notifikasi UUID renaming jika diaktifkan
            if params['rename_files']:
                naming_stats = self.file_processor.get_naming_statistics()
                stats['naming_stats'] = naming_stats
                renamed_files = naming_stats.get('total_files', 0)
                self.logger.info(f"🔄 UUID renaming: {renamed_files} files")
                self._notify_progress(
                    'organize_rename_complete', 
                    92, 
                    100, 
                    f"🔄 UUID renaming selesai: {renamed_files} files"
                )
            
            # Notifikasi pengorganisasian selesai
            self._notify_progress(
                'organize_complete', 
                95, 
                100, 
                f"✅ Pengorganisasian selesai: {stats['total_images']} gambar, {stats['total_labels']} label"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Pengorganisasian gagal: {str(e)}")
            self._notify_progress('organize_error', 85, 100, f"❌ Pengorganisasian gagal: {str(e)}")
            raise Exception(f"Organization failed: {str(e)}")
    
    def _validate_results(self, final_dir: Path) -> None:
        """Validate results dengan progress tracking dan notifikasi per step"""
        self._notify_progress('validate_start', 95, 100, "✅ Memulai validasi hasil...")
        self.logger.info(f"✅ Memvalidasi hasil di {final_dir}")
        
        try:
            # Notifikasi awal validasi
            self._notify_progress('validate_structure', 96, 100, "🔍 Memeriksa struktur dataset...")
            
            # Validasi struktur dataset
            validation_result = self.validator.validate_extracted_dataset(final_dir)
            
            # Notifikasi hasil validasi
            if validation_result['valid']:
                self._notify_progress(
                    'validate_success', 
                    98, 
                    100, 
                    f"✅ Validasi berhasil: {validation_result.get('splits', [])} splits"
                )
                self.logger.success(f"✅ Dataset valid dengan {len(validation_result.get('splits', []))} splits")
            else:
                issues = validation_result.get('issues', [])
                self._notify_progress(
                    'validate_issues', 
                    97, 
                    100, 
                    f"⚠️ Validasi menemukan {len(issues)} masalah"
                )
                
                # Notifikasi per masalah (maksimal 3)
                for i, issue in enumerate(issues[:3]):
                    self._notify_progress(
                        f'validate_issue_{i+1}', 
                        97, 
                        100, 
                        f"⚠️ Issue {i+1}: {issue}"
                    )
            
            # Notifikasi validasi selesai
            self._notify_progress('validate_complete', 99, 100, "✅ Validasi selesai")
            
            if not validation_result['valid'] and validation_result.get('issues'):
                issues = validation_result['issues'][:3]  # Limit issues shown
                self.logger.warning(f"⚠️ Validation issues: {', '.join(issues)}")
                
        except Exception as e:
            self.logger.error(f"❌ Validasi gagal: {str(e)}")
            self._notify_progress('validate_error', 97, 100, f"❌ Validasi gagal: {str(e)}")
            # Tidak raise exception karena validasi bukan langkah kritis
    
    def _has_existing_dataset(self, dataset_dir: Path) -> bool:
        """Check existing dataset"""
        return (dataset_dir.exists() and 
                any((dataset_dir / split).exists() and any((dataset_dir / split).iterdir())
                    for split in ['train', 'valid', 'test']))
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information untuk debugging"""
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        return {
            'has_progress_callback': self._progress_callback is not None,
            'callback_signature': 'step, current, total, message',
            'environment': {
                'is_colab': env_manager.is_colab,
                'drive_mounted': env_manager.is_drive_mounted
            },
            'components_loaded': {
                'roboflow_client': self._roboflow_client is not None,
                'file_processor': self._file_processor is not None,
                'validator': self._validator is not None
            },
            'config_keys': list(self.config.keys()),
            'uuid_support': True
        }


def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """
    FIXED: Factory dengan comprehensive validation dan UI compatibility
    """
    from smartcash.common.logger import get_logger
    logger = logger or get_logger('downloader.factory')
    
    try:
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Apply defaults untuk missing keys
        defaults = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'output_format': 'yolov5pytorch',
            'rename_files': True,
            'organize_dataset': True,
            'validate_download': True,
            'backup_existing': False,
            'retry_count': 3,
            'timeout': 30
        }
        
        # Merge dengan defaults
        merged_config = {**defaults, **config}
        
        # Validate required fields
        validation = ValidationHelper.validate_config(
            merged_config, ['workspace', 'project', 'version', 'api_key']
        )
        
        if not validation['valid']:
            logger.error(f"❌ Config validation failed: {'; '.join(validation['errors'])}")
            return None
        
        service = DownloadService(merged_config, logger)
        logger.success("✅ Download service created successfully")
        return service
        
    except Exception as e:
        logger.error(f"❌ Error creating download service: {str(e)}")
        return None