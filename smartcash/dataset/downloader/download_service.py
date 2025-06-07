"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: REFACTORED download service dengan progress tracking yang konsisten
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.dataset.downloader.base import BaseDownloaderComponent, ValidationHelper, PathHelper, FileHelper
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.downloader.file_processor import create_file_processor
from smartcash.dataset.downloader.validators import create_dataset_validator
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage
from smartcash.common.environment import get_environment_manager

class DownloadService(BaseDownloaderComponent):
    """REFACTORED download service dengan consistent progress tracking"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        self.env_manager = get_environment_manager()
        self.progress_tracker = None
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
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
            self._file_processor = create_file_processor(self.logger)
        return self._file_processor
    
    @property
    def validator(self):
        if not self._validator:
            self._validator = create_dataset_validator(self.logger)
        return self._validator
    
    def download_dataset(self) -> Dict[str, Any]:
        """
        Download dataset dengan consistent progress tracking
        
        Returns:
            Dictionary dengan format yang diharapkan UI
        """
        start_time = time.time()
        
        try:
            # Stage 1: Initialization
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.INIT, "Memulai proses download...")
            
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
            
            # Setup paths using environment manager
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
            
            # Stage 4: Download
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.DOWNLOAD, "Mengunduh dataset...")
            
            self._download_dataset_file(download_url, paths)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Download selesai")
            
            # Stage 5: Extract
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.EXTRACT, "Mengekstrak dataset...")
            
            self._extract_dataset(paths)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Ekstraksi selesai")
            
            # Stage 6: Organize
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.ORGANIZE, "Mengorganisasi dataset...")
            
            stats = self._organize_dataset(paths, params)
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("✅ Organisasi selesai")
            
            # Stage 7: Validate
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
            success_msg = f"Dataset berhasil didownload: {stats.get('total_images', 0):,} gambar"
            
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
                    'naming_stats': stats.get('naming_stats', {}) if params['rename_files'] else None
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
        """Validate config dengan required fields"""
        required_fields = ['workspace', 'project', 'version', 'api_key']
        return ValidationHelper.validate_config(self.config, required_fields)
    
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
            'rename_files': self.config.get('rename_files', True)
        }
    
    def _setup_download_paths(self, dataset_path: Path, params: Dict[str, Any]) -> Dict[str, Path]:
        """Setup download paths using environment-aware dataset path"""
        dataset_name = f"{params['workspace']}_{params['project']}_v{params['version']}"
        
        paths = {
            'dataset_dir': dataset_path,
            'temp_dir': dataset_path / 'downloads' / f"{dataset_name}_temp",
            'temp_zip': dataset_path / 'downloads' / f"{dataset_name}_temp" / 'dataset.zip',
            'extract_dir': dataset_path / 'downloads' / f"{dataset_name}_temp" / 'extracted'
        }
        
        # Create temp directories
        FileHelper.ensure_directory(paths['temp_dir'])
        
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
        """Download dataset file dengan progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, "📥 Starting download...")
        
        import requests
        
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(paths['temp_zip'], 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
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
                
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, paths['extract_dir'])
                    
                    if self.progress_tracker and i % max(1, total_files // 10) == 0:
                        progress = 20 + int((i / total_files) * 70)  # 20-90%
                        self.progress_tracker.update_stage(
                            progress, 
                            f"📦 Extracted: {i+1}/{total_files} files"
                        )
            
            self.logger.success(f"✅ Ekstraksi selesai: {total_files} files")
            
        except Exception as e:
            raise Exception(f"Extract failed: {str(e)}")
    
    def _organize_dataset(self, paths: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Organize dataset dengan progress tracking termasuk UUID renaming"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, "🗂️ Analyzing structure...")
        
        try:
            # Phase 1: Basic organization
            result = self.file_processor.organize_dataset(
                paths['extract_dir'], paths['dataset_dir']
            )
            
            if result['status'] != 'success':
                raise Exception(f"Organization failed: {result['message']}")
            
            if self.progress_tracker:
                self.progress_tracker.complete_stage("🗂️ Basic organization complete")
            
            # Phase 2: UUID Renaming (if enabled)
            if params['rename_files']:
                if self.progress_tracker:
                    self.progress_tracker.start_stage(DownloadStage.UUID_RENAME, "🔄 Memulai UUID renaming...")
                
                rename_result = self._execute_uuid_renaming(paths['dataset_dir'], params)
                
                # Merge rename results
                result['naming_stats'] = rename_result.get('naming_stats', {})
                result['uuid_renamed'] = True
                
                if self.progress_tracker:
                    renamed_count = rename_result.get('naming_stats', {}).get('total_files', 0)
                    self.progress_tracker.complete_stage(f"🔄 UUID renaming selesai: {renamed_count} files")
            
            if self.progress_tracker:
                total_images = result.get('total_images', 0)
                self.progress_tracker.update_stage(90, f"🗂️ Final organization: {total_images} images")
            
            return result
            
        except Exception as e:
            raise Exception(f"Organization failed: {str(e)}")
    
    def _execute_uuid_renaming(self, dataset_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute UUID renaming dengan detailed progress tracking"""
        try:
            from smartcash.common.utils.file_naming_manager import FileNamingManager
            
            if self.progress_tracker:
                self.progress_tracker.update_stage(10, "🔄 Initializing UUID naming manager...")
            
            naming_manager = FileNamingManager(logger=self.logger)
            
            # Scan all splits for renaming
            splits = ['train', 'valid', 'test']
            total_renamed = 0
            split_stats = {}
            
            for i, split in enumerate(splits):
                split_dir = dataset_path / split
                if not split_dir.exists():
                    continue
                
                split_progress = 20 + (i / len(splits)) * 60  # 20-80%
                
                if self.progress_tracker:
                    self.progress_tracker.update_stage(
                        int(split_progress), 
                        f"🔄 Renaming {split} files..."
                    )
                
                # Rename files in this split
                renamed_count = self._rename_split_files(split_dir, naming_manager)
                split_stats[split] = renamed_count
                total_renamed += renamed_count
                
                if self.progress_tracker:
                    self.progress_tracker.update_stage(
                        int(split_progress + 15), 
                        f"🔄 {split}: {renamed_count} files renamed"
                    )
            
            # Generate statistics
            naming_stats = {
                'total_files': total_renamed,
                'splits': split_stats,
                'uuid_format': True,
                'naming_strategy': 'research_uuid'
            }
            
            if self.progress_tracker:
                self.progress_tracker.update_stage(90, f"🔄 UUID statistics generated")
            
            return {
                'status': 'success',
                'naming_stats': naming_stats,
                'total_renamed': total_renamed
            }
            
        except Exception as e:
            raise Exception(f"UUID renaming failed: {str(e)}")
    
    def _rename_split_files(self, split_dir: Path, naming_manager) -> int:
        """Rename files dalam single split directory"""
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            return 0
        
        renamed_count = 0
        
        # Process each image file
        for img_file in images_dir.glob('*.*'):
            if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            
            try:
                # Get corresponding label
                label_file = labels_dir / f"{img_file.stem}.txt"
                primary_class = None
                
                if label_file.exists():
                    primary_class = naming_manager.extract_primary_class_from_label(label_file)
                
                # Generate new filename
                file_info = naming_manager.generate_file_info(img_file.name, primary_class)
                new_filename = file_info.get_filename()
                
                # Skip if already in UUID format
                if naming_manager.parse_existing_filename(img_file.name):
                    continue
                
                # Rename image
                new_img_path = images_dir / new_filename
                if not new_img_path.exists():
                    img_file.rename(new_img_path)
                    renamed_count += 1
                
                # Rename corresponding label
                if label_file.exists():
                    new_label_filename = f"{Path(new_filename).stem}.txt"
                    new_label_path = labels_dir / new_label_filename
                    if not new_label_path.exists():
                        label_file.rename(new_label_path)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error renaming {img_file.name}: {str(e)}")
        
        return renamed_count
    
    def _validate_results(self, dataset_path: Path) -> None:
        """Validate results dengan progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(20, "✅ Validating structure...")
        
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
        """Check existing dataset"""
        return (dataset_path.exists() and 
                any((dataset_path / split).exists() and any((dataset_path / split).iterdir())
                    for split in ['train', 'valid', 'test']))
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response dengan format yang konsisten"""
        return {
            'status': 'error',
            'message': message,
            'stats': {'total_images': 0, 'total_labels': 0, 'splits': {}},
            'output_dir': '',
            'duration': 0.0
        }

def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Factory dengan comprehensive validation"""
    from smartcash.common.logger import get_logger
    logger = logger or get_logger('downloader.factory')
    
    try:
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Apply defaults
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