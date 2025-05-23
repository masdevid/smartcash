"""
File: smartcash/ui/dataset/download/services/ui_download_service_final.py
Deskripsi: Enhanced download service dengan progress integration yang lebih robust dan step tracking yang akurat
"""

import time
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.ui_roboflow_downloader import UIRoboflowDownloader
from smartcash.ui.dataset.download.services.progress_bridge import ProgressBridge

class UIDownloadServiceFinal:
    """Enhanced download service dengan accurate progress tracking dan robust error handling."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Enhanced progress bridge dengan step tracking
        self.progress_bridge = ProgressBridge(
            observer_manager=ui_components.get('observer_manager'),
            namespace="download"
        )
        
        # Create downloader dengan enhanced progress callback
        self.downloader = UIRoboflowDownloader(logger=self.logger)
        self.downloader.set_progress_callback(self._enhanced_progress_callback)
        
        # Step tracking
        self.total_steps = 4
        self.current_step = 0
        self.step_names = ['validate', 'metadata', 'download', 'verify']
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dengan enhanced progress tracking dan comprehensive error handling."""
        start_time = time.time()
        
        try:
            # 🚀 Initialize
            self.progress_bridge.notify_start("Memulai download dataset", self.total_steps)
            
            # Step 1: Validate parameters
            self._next_step("validate", "Memvalidasi parameter")
            self._validate_params(params)
            self.progress_bridge.notify_progress(25, 100, "Parameter valid", "validate", 1, self.total_steps)
            
            # Step 2: Get metadata
            self._next_step("metadata", "Mendapatkan informasi dataset")
            metadata = self._get_metadata(params)
            self.progress_bridge.notify_progress(50, 100, "Metadata diperoleh", "metadata", 2, self.total_steps)
            
            # Step 3: Download & extract
            self._next_step("download", "Mendownload dataset")
            output_path = Path(params['output_dir'])
            download_url = metadata['export']['link']
            
            if not self.downloader.download_and_extract(download_url, output_path):
                raise Exception("Download atau ekstraksi gagal")
            
            # Step 4: Verify and finalize
            self._next_step("verify", "Memverifikasi hasil download")
            stats = self._get_stats(output_path)
            self.progress_bridge.notify_progress(100, 100, "Verifikasi selesai", "verify", 4, self.total_steps)
            
            # 🎉 Complete
            duration = time.time() - start_time
            total_images = stats.get('total_images', 0)
            self.progress_bridge.notify_complete(f"Download selesai: {total_images} gambar dalam {duration:.1f}s", duration)
            
            return {
                'status': 'success',
                'output_dir': str(output_path),
                'stats': stats,
                'duration': duration,
                'drive_storage': self._is_drive_storage(output_path)
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.progress_bridge.notify_error(f"Download gagal: {error_msg}")
            
            if self.logger:
                self.logger.error(f"❌ Download error setelah {duration:.1f}s: {error_msg}")
            
            return {'status': 'error', 'message': error_msg, 'duration': duration}
    
    def _next_step(self, step_name: str, description: str) -> None:
        """Move to next step dengan logging."""
        self.current_step += 1
        if self.logger:
            self.logger.info(f"📋 Step {self.current_step}/{self.total_steps}: {description}")
    
    def _enhanced_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Enhanced callback dengan step-aware progress calculation."""
        try:
            # Map step ke overall progress ranges yang lebih akurat
            step_ranges = {
                'download': (50, 85),   # 50-85% untuk download 
                'extract': (85, 95),    # 85-95% untuk extract
            }
            
            if step in step_ranges:
                start_pct, end_pct = step_ranges[step]
                
                # Calculate progress dalam range
                if total > 0:
                    step_progress = (current / total)
                    overall_progress = start_pct + (step_progress * (end_pct - start_pct))
                else:
                    overall_progress = start_pct
                
                # Step info mapping
                step_info_map = {
                    'download': ('Mendownload dataset', 3, self.total_steps),
                    'extract': ('Mengekstrak dataset', 3, self.total_steps)  # Same step visually
                }
                
                step_display_name, current_step_display, total_steps = step_info_map.get(
                    step, (message, self.current_step, self.total_steps)
                )
                
                # Enhanced progress notification
                self.progress_bridge.notify_progress(
                    int(overall_progress), 100, message,
                    step_display_name, current_step_display, total_steps
                )
                
                # Log significant progress milestones
                if self.logger and current % max(1, total // 4) == 0:
                    percentage = int((current / total) * 100) if total > 0 else 0
                    self.logger.debug(f"🔄 {step}: {percentage}% - {message}")
        
        except Exception as e:
            # Fallback progress notification
            if self.logger:
                self.logger.debug(f"Progress callback error: {str(e)}")
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Enhanced parameter validation dengan detailed error messages."""
        required = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise ValueError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        # Validate API key format
        api_key = params['api_key']
        if len(api_key) < 10:
            raise ValueError("API key terlalu pendek, periksa kembali")
        
        # Ensure output directory
        output_path = Path(params['output_dir'])
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Tidak dapat membuat direktori output: {str(e)}")
        
        if self.logger:
            self.logger.debug("✅ Parameter validation berhasil")
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get dataset metadata dengan enhanced error handling."""
        import requests
        
        metadata_url = (f"https://api.roboflow.com/{params['workspace']}"
                       f"/{params['project']}/{params['version']}"
                       f"/yolov5pytorch?api_key={params['api_key']}")
        
        try:
            if self.logger:
                self.logger.debug(f"🌐 Requesting metadata: {params['workspace']}/{params['project']}:{params['version']}")
            
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            metadata = response.json()
            
            # Enhanced metadata validation
            if 'export' not in metadata:
                raise ValueError("Response tidak mengandung export data")
            
            if 'link' not in metadata['export']:
                raise ValueError("Export link tidak ditemukan dalam response")
            
            # Log dataset info jika tersedia
            if self.logger and 'project' in metadata:
                project_info = metadata['project']
                classes_count = len(project_info.get('classes', []))
                version_info = metadata.get('version', {})
                images_count = version_info.get('images', 0)
                
                self.logger.info(f"📊 Dataset info: {classes_count} kelas, {images_count} gambar")
            
            return metadata
            
        except requests.RequestException as e:
            if "404" in str(e):
                raise Exception(f"Dataset tidak ditemukan: {params['workspace']}/{params['project']}:{params['version']}")
            elif "401" in str(e) or "403" in str(e):
                raise Exception("API key tidak valid atau tidak memiliki akses ke dataset")
            else:
                raise Exception(f"Error koneksi ke Roboflow: {str(e)}")
        except Exception as e:
            raise Exception(f"Error mendapatkan metadata: {str(e)}")
    
    def _get_stats(self, output_path: Path) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {'total_images': 0, 'total_labels': 0}
        
        try:
            # Count files per split
            for split in ['train', 'valid', 'test']:
                images_dir = output_path / split / 'images'
                labels_dir = output_path / split / 'labels'
                
                if images_dir.exists():
                    img_count = len(list(images_dir.glob('*.*')))
                    stats['total_images'] += img_count
                    stats[f'{split}_images'] = img_count
                
                if labels_dir.exists():
                    label_count = len(list(labels_dir.glob('*.txt')))
                    stats['total_labels'] += label_count
                    stats[f'{split}_labels'] = label_count
            
            # Calculate dataset size
            try:
                total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
                stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception:
                stats['total_size_mb'] = 0
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Error menghitung statistik: {str(e)}")
        
        return stats
    
    def _is_drive_storage(self, output_path: Path) -> bool:
        """Check if dataset disimpan di Google Drive."""
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if not env_manager.is_colab or not env_manager.drive_path:
                return False
            
            # Resolve paths untuk comparison
            resolved_output = output_path.resolve()
            resolved_drive = env_manager.drive_path.resolve()
            
            return str(resolved_output).startswith(str(resolved_drive))
        except Exception:
            return False