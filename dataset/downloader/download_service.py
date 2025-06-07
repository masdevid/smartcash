"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Download service dengan optimized chunk_size untuk performance
"""

class DownloadService(BaseDownloaderComponent):
    """Download service dengan config-aware workers dan optimized chunks"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        self.env_manager = get_environment_manager()
        self.progress_tracker = None
        
        # Extract worker config dengan optimized defaults
        self.max_workers = config.get('max_workers', 4)
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.chunk_size = config.get('chunk_size', 262144)  # 256KB for faster downloads
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
    def _download_dataset_file(self, download_url: str, paths: Dict[str, Path]) -> None:
        """Download dengan optimized chunk size untuk performance"""
        if self.progress_tracker:
            self.progress_tracker.update_stage(10, f"📥 Starting download (chunk: {self.chunk_size/1024:.0f}KB)...")
        
        import requests
        
        try:
            response = requests.get(download_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            last_progress = 0
            
            with open(paths['temp_zip'], 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress less frequently untuk avoid UI flooding
                        if total_size > 0 and self.progress_tracker:
                            current_progress = int((downloaded / total_size) * 80) + 10
                            
                            # Only update every 5% untuk prevent browser crash
                            if current_progress >= last_progress + 5:
                                size_mb = downloaded / (1024 * 1024)
                                total_mb = total_size / (1024 * 1024)
                                speed_mbps = (downloaded / (1024 * 1024)) / max(1, (time.time() - start_time))
                                
                                self.progress_tracker.update_stage(
                                    current_progress, 
                                    f"📥 {size_mb:.1f}/{total_mb:.1f} MB ({speed_mbps:.1f} MB/s)"
                                )
                                last_progress = current_progress
            
            file_size_mb = paths['temp_zip'].stat().st_size / (1024 * 1024)
            duration = time.time() - start_time
            avg_speed = file_size_mb / max(1, duration)
            
            self.logger.success(f"✅ Download selesai: {file_size_mb:.2f} MB in {duration:.1f}s ({avg_speed:.1f} MB/s)")
            
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Factory dengan optimized defaults untuk performance"""
    from smartcash.common.logger import get_logger
    logger = logger or get_logger('downloader.factory')
    
    try:
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Apply defaults dengan optimized values
        from smartcash.common.threadpools import get_download_workers
        
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
            'chunk_size': 262144  # 256KB optimized chunks
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
        chunk_kb = merged_config['chunk_size'] / 1024
        logger.success(f"✅ Download service created: {merged_config['max_workers']} workers, {chunk_kb:.0f}KB chunks")
        return service
        
    except Exception as e:
        logger.error(f"❌ Error creating download service: {str(e)}")
        return None