"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Fixed download service dengan proper imports dan Path handling
"""

import time
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.dataset.downloader.base import BaseDownloaderComponent, ValidationHelper, PathHelper, FileHelper, DirectoryManager
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.downloader.file_processor import create_file_processor
from smartcash.dataset.downloader.validators import create_dataset_validator
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage
from smartcash.common.environment import get_environment_manager

class DownloadService(BaseDownloaderComponent):
    """Fixed download service dengan proper imports dan enhanced directory management"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        self.env_manager = get_environment_manager()
        self.progress_tracker = None
        self.directory_manager = DirectoryManager()
        
        # Extract config dengan optimal workers
        self.max_workers = config.get('max_workers', self._get_default_workers())
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.chunk_size = config.get('chunk_size', 8192*4)
        self.parallel_downloads = config.get('parallel_downloads', True)
        
        self.logger.info(f"🔧 DownloadService initialized with {self.max_workers} workers")
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
    def _get_default_workers(self) -> int:
        """Get default optimal workers untuk download"""
        try:
            from smartcash.common.threadpools import get_download_workers
            return get_download_workers()
        except ImportError:
            return 4  # Safe fallback
    
    def set_progress_callback(self, callback) -> None:
        """Set progress callback dan create tracker"""
        super().set_progress_callback(callback)
        self.progress_tracker = DownloadProgressTracker(callback)
    
    @property
    def roboflow_client(self):
        if not self._roboflow_client:
            self._roboflow_client = create_roboflow_client(self.config.get('api_key', ''), self.logger)
        return self._roboflow_client
    
    @property
    def file_processor(self):
        if not self._file_processor:
            # Pass config workers ke file processor
            self._file_processor = create_file_processor(self.logger, self.max_workers)
        return self._file_processor
    
    @property
    def validator(self):
        if not self._validator:
            self._validator = create_dataset_validator(self.logger, self.max_workers)
        return self._validator
    
    def download_dataset(self) -> Dict[str, Any]:
        """Enhanced download dengan proper directory management dan error handling"""
        start_time = time.time()
        
        try:
            # Stage 1: Initialization
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.INIT, f"Memulai download dengan {self.max_workers} workers...")
            
            # Validate config
            validation = self._validate_config()
            if not validation['valid']:
                error_msg = f"Konfigurasi tidak valid: {'; '.join(validation['errors'])}"
                if self.progress_tracker:
                    self.progress_tracker.error(error_msg)
                return self._create_error_response(error_msg)
            
            if self.progress_tracker:
                self.progress_tracker.update_stage(50, "✅ Konfigurasi valid")
            
            # Extract parameters
            params = self._extract_params()
            
            # Setup paths using environment manager dengan enhanced validation
            dataset_path = self.env_manager.get_dataset_path()
            paths = self._setup_download_paths(dataset_path, params)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Inisialisasi selesai")
            
            # Stage 2: Get metadata
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.METADATA, "Mengambil metadata dataset...")
            
            metadata = self._get_metadata(params)
            download_url = metadata.get('download_url')
            if not download_url:
                error_msg = "URL download tidak ditemukan dalam metadata"
                if self.progress_tracker:
                    self.progress_tracker.error(error_msg)
                return self._create_error_response(error_msg)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Metadata berhasil diperoleh")
            
            # Stage 3: Backup if needed
            if params['backup_existing'] and self._has_existing_dataset(dataset_path):
                if self.progress_tracker:
                    self.progress_tracker.start_stage(DownloadStage.BACKUP, "Membuat backup...")
                
                self._handle_backup(dataset_path)
                
                if self.progress_tracker:
                    self.progress_tracker.complete_stage("✅ Backup selesai")
            
            # Stage 4: Download dengan config optimization
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.DOWNLOAD, f"Mengunduh dataset (chunk: {self.chunk_size})...")
            
            self._download_dataset_file(download_url, paths)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Download selesai")
            
            # Stage 5: Extract dengan parallelization
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.EXTRACT, "Mengekstrak dataset...")
            
            self._extract_dataset(paths)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Ekstraksi selesai")
            
            # Stage 6: Organize dengan optimal workers
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.ORGANIZE, f"Mengorganisasi dataset ({self.max_workers} workers)...")
            
            stats = self._organize_dataset(paths, params)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Organisasi selesai")
            
            # Stage 7: Validate dengan parallel processing
            if params['validate_download']:
                if self.progress_tracker:
                    self.progress_tracker.start_stage(DownloadStage.VALIDATE, "Memvalidasi hasil...")
                
                self._validate_results(dataset_path)
                
                if self.progress_tracker:
                    self.progress_tracker.complete_stage("✅ Validasi selesai")
            
            # Stage 8: Cleanup
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.CLEANUP, "Membersihkan file sementara...")
            
            FileHelper.cleanup_temp(paths['temp_dir'])
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Cleanup selesai")
            
            # Complete
            duration = time.time() - start_time
            success_msg = f"Dataset berhasil didownload: {stats.get('total_images', 0):,} gambar ({duration:.1f}s, {self.max_workers} workers)"
            
            if self.progress_tracker:
                self.progress_tracker.complete_all(success_msg)
            
            return {
                'status': 'success',
                'message': success_msg,
                'stats': {
                    'total_images': stats.get('total_images', 0),
                    'total_labels': stats.get('total_labels', 0),
                    'splits': stats.get('splits', {}),
                    'uuid_renamed': params['rename_files'],
                    'naming_stats': stats.get('naming_stats', {}) if params['rename_files'] else None,
                    'workers_used': self.max_workers,
                    'parallel_enabled': self.parallel_downloads
                },
                'output_dir': str(dataset_path),
                'duration': duration,
                'metadata': {
                    'workspace': params['workspace'],
                    'project': params['project'],
                    'version': params['version'],
                    'format': params['output_format']
                }
            }
            
        except Exception as e:
            error_msg = f"Error downloading dataset: {str(e)}"
            if self.progress_tracker:
                self.progress_tracker.error(error_msg)
            return self._create_error_response(error_msg)
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate config dengan comprehensive checks"""
        required_fields = ['workspace', 'project', 'version', 'api_key']
        return ValidationHelper.validate_config(self.config, required_fields)
    
    def _extract_params(self) -> Dict[str, Any]:
        """Extract parameters dengan optimal defaults"""
        return {
            'workspace': self.config.get('workspace', ''),
            'project': self.config.get('project', ''),
            'version': self.config.get('version', ''),
            'api_key': self.config.get('api_key', ''),
            'output_format': self.config.get('output_format', 'yolov5pytorch'),
            'validate_download': self.config.get('validate_download', True),
            'backup_existing': self.config.get('backup_existing', False),
            'rename_files': self.config.get('rename_files', True)
        }
    
    def _setup_download_paths(self, dataset_path: Path, params: Dict[str, Any]) -> Dict[str, Path]:
        """Enhanced setup download paths dengan directory validation"""
        dataset_name = f"{params['workspace']}_{params['project']}_v{params['version']}"
        
        paths = {
            'dataset_dir': dataset_path,
            'temp_dir': dataset_path / 'downloads' / f"{dataset_name}_temp",
            'temp_zip': dataset_path / 'downloads' / f"{dataset_name}_temp" / 'dataset.zip',
            'extract_dir': dataset_path / 'downloads' / f"{dataset_name}_temp" / 'extracted'
        }
        
        # Create temp directories dengan error handling
        try:
            FileHelper.ensure_directory(paths['temp_dir'])
            
            # Ensure dataset structure exists
            structure_result = self.directory_manager.ensure_dataset_structure(dataset_path)
            if structure_result['status'] == 'error':
                raise Exception(f"Failed to create dataset structure: {structure_result['message']}")
                
        except Exception as e:
            raise Exception(f"Failed to setup download paths: {str(e)}")
        
        return paths
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata dengan progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(20, "📊 Connecting to Roboflow...")
        
        result = self.roboflow_client.get_dataset_metadata(
            params['workspace'], params['project'], 
            params['version'], params['output_format']
        )
        
        if result['status'] != 'success':
            raise Exception(f"Metadata failed: {result['message']}")
        
        if self.progress_tracker:
            self.progress_tracker.update_stage(80, "📊 Metadata response received")
        
        return result
    
    def _download_dataset_file(self, download_url: str, paths: Dict[str, Path]) -> None:
        """Download dengan config timeout dan chunk_size"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, f"📥 Starting download (chunk: {self.chunk_size})...")
        
        import requests
        
        try:
            response = requests.get(download_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(paths['temp_zip'], 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and self.progress_tracker:
                            progress = int((downloaded / total_size) * 80) + 10  # 10-90%
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            self.progress_tracker.update_stage(
                                progress, 
                                f"📥 Downloaded: {size_mb:.1f}/{total_mb:.1f} MB"
                            )
            
            file_size_mb = paths['temp_zip'].stat().st_size / (1024 * 1024)
            self.logger.success(f"✅ Download selesai: {file_size_mb:.2f} MB")
            
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def _extract_dataset(self, paths: Dict[str, Path]) -> None:
        """Extract dataset dengan progress tracking"""
        import zipfile
        
        try:
            if self.progress_tracker:
                self.progress_tracker.update_stage(10, "📦 Opening archive...")
            
            with zipfile.ZipFile(paths['temp_zip'], 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                if self.progress_tracker:
                    self.progress_tracker.update_stage(20, f"📦 Extracting {total_files} files...")
                
                # Extract dengan batch processing untuk large files
                batch_size = max(1, total_files // 20)  # 20 progress updates max
                
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, paths['extract_dir'])
                    
                    if self.progress_tracker and i % batch_size == 0:
                        progress = 20 + int((i / total_files) * 70)  # 20-90%
                        self.progress_tracker.update_stage(
                            progress, 
                            f"📦 Extracted: {i+1}/{total_files} files"
                        )
            
            self.logger.success(f"✅ Ekstraksi selesai: {total_files} files")
            
        except Exception as e:
            raise Exception(f"Extract failed: {str(e)}")
    
    def _organize_dataset(self, paths: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Organize dataset dengan config workers"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, f"🗂️ Organizing dengan {self.max_workers} workers...")
        
        try:
            # Use file processor dengan config workers
            if params['rename_files']:
                result = self.file_processor.organize_dataset_with_renaming(
                    paths['extract_dir'], paths['dataset_dir']
                )
            else:
                result = self.file_processor.organize_dataset(
                    paths['extract_dir'], paths['dataset_dir']
                )
            
            if result['status'] != 'success':
                raise Exception(f"Organization failed: {result['message']}")
            
            if self.progress_tracker:
                total_images = result.get('total_images', 0)
                self.progress_tracker.update_stage(90, f"🗂️ Organization complete: {total_images} images")
            
            return result
            
        except Exception as e:
            raise Exception(f"Organization failed: {str(e)}")
    
    def _validate_results(self, dataset_path: Path) -> None:
        """Validate results dengan config workers"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(20, f"✅ Validating dengan {self.max_workers} workers...")
        
        try:
            validation_result = self.validator.validate_extracted_dataset(dataset_path)
            
            if self.progress_tracker:
                if validation_result['valid']:
                    splits_count = len(validation_result.get('splits', []))
                    self.progress_tracker.update_stage(80, f"✅ Valid dataset: {splits_count} splits")
                else:
                    issues = validation_result.get('issues', [])
                    self.progress_tracker.update_stage(60, f"⚠️ Issues found: {len(issues)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Validation warning: {str(e)}")
    
    def _handle_backup(self, dataset_path: Path) -> None:
        """Handle backup dengan progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(20, "💾 Creating backup...")
        
        backup_result = FileHelper.backup_directory(dataset_path)
        
        if self.progress_tracker:
            if backup_result['success']:
                self.progress_tracker.update_stage(80, "💾 Backup created successfully")
            else:
                self.progress_tracker.update_stage(60, "⚠️ Backup failed")
    
    def _has_existing_dataset(self, dataset_path: Path) -> bool:
        """Enhanced check existing dataset dengan smart detection"""
        if not dataset_path.exists():
            return False
            
        # Check for actual content, not just directories
        for split in ['train', 'valid', 'test']:
            split_images = dataset_path / split / 'images'
            if split_images.exists() and any(split_images.iterdir()):
                return True
        
        return False
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create consistent error response"""
        return {
            'status': 'error',
            'message': message,
            'stats': {
                'total_images': 0, 
                'total_labels': 0, 
                'splits': {},
                'workers_used': self.max_workers,
                'parallel_enabled': self.parallel_downloads
            },
            'output_dir': '',
            'duration': 0.0
        }


def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Enhanced factory dengan comprehensive config validation dan optimal workers"""
    from smartcash.common.logger import get_logger
    logger = logger or get_logger('downloader.factory')
    
    try:
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Apply defaults dengan optimal workers
        from smartcash.common.threadpools import get_download_workers, get_optimal_thread_count
        
        defaults = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'output_format': 'yolov5pytorch',
            'rename_files': True,
            'organize_dataset': True,
            'validate_download': True,
            'backup_existing': False,
            'max_workers': get_download_workers(),
            'retry_count': 3,
            'timeout': 30,
            'chunk_size': 8192,
            'parallel_downloads': True
        }
        
        merged_config = {**defaults, **config}
        
        # Validate required fields
        validation = ValidationHelper.validate_config(
            merged_config, ['workspace', 'project', 'version', 'api_key']
        )
        
        if not validation['valid']:
            logger.error(f"❌ Config validation failed: {'; '.join(validation['errors'])}")
            return None
        
        service = DownloadService(merged_config, logger)
        logger.success(f"✅ Download service created with {merged_config['max_workers']} workers")
        return service
        
    except Exception as e:
        logger.error(f"❌ Error creating download service: {str(e)}")
        return None