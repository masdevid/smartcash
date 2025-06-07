"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: UPDATED download service yang menggunakan config workers
"""

class DownloadService(BaseDownloaderComponent):
    """Download service dengan config-aware workers"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        self.env_manager = get_environment_manager()
        self.progress_tracker = None
        
        # Extract worker config
        self.max_workers = config.get('max_workers', 4)
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.chunk_size = config.get('chunk_size', 8192)
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
    @property
    def file_processor(self):
        if not self._file_processor:
            # Pass max_workers to file processor
            self._file_processor = create_file_processor(self.logger, self.max_workers)
        return self._file_processor
    
    def _download_dataset_file(self, download_url: str, paths: Dict[str, Path]) -> None:
        """Download dengan config timeout dan chunk_size"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, "📥 Starting download...")
        
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
                            progress = int((downloaded / total_size) * 80) + 10
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

def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Factory dengan config validation dan worker optimization"""
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
            'chunk_size': 8192
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