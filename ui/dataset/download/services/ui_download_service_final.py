"""
File: smartcash/ui/dataset/download/services/ui_download_service_final.py
Deskripsi: Enhanced download service dengan accurate step-by-step progress tracking dan robust error handling
"""

import time
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.ui_roboflow_downloader import UIRoboflowDownloader
from smartcash.ui.dataset.download.services.progress_bridge import ProgressBridge

class UIDownloadServiceFinal:
    """Enhanced download service dengan accurate step-by-step progress tracking."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Enhanced progress bridge dengan UI reference
        self.progress_bridge = ProgressBridge(
            observer_manager=ui_components.get('observer_manager'),
            namespace="download"
        )
        self.progress_bridge.set_ui_components_reference(ui_components)
        
        # Create downloader dengan step-aware progress callback
        self.downloader = UIRoboflowDownloader(logger=self.logger)
        self.downloader.set_progress_callback(self._step_aware_progress_callback)
        
        # Step definitions yang lebih detail
        self.step_definitions = {
            'validate': {'weight': 5, 'name': 'Validasi'},
            'metadata': {'weight': 15, 'name': 'Metadata'},
            'download': {'weight': 60, 'name': 'Download'},
            'extract': {'weight': 15, 'name': 'Ekstraksi'},
            'verify': {'weight': 5, 'name': 'Verifikasi'}
        }
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dengan enhanced step-by-step progress tracking."""
        start_time = time.time()
        
        try:
            # 🚀 Initialize dengan clear messaging
            self.progress_bridge.notify_start("Memulai download dataset", total_steps=5)
            
            # Step 1: Validate parameters (5%)
            self._execute_step('validate', "Memvalidasi parameter", 
                             lambda: self._validate_params(params))
            
            # Step 2: Get metadata (15%)
            self._execute_step('metadata', "Mendapatkan informasi dataset",
                             lambda: self._get_metadata(params))
            metadata = self._last_step_result
            
            # Step 3: Download & extract (75% total - 60% download + 15% extract)
            output_path = Path(params['output_dir'])
            download_url = metadata['export']['link']
            
            # Download akan menggunakan callback untuk progress yang smooth
            if not self.downloader.download_and_extract(download_url, output_path):
                raise Exception("Download atau ekstraksi gagal")
            
            # Step 4: Verify (5%)
            self._execute_step('verify', "Memverifikasi hasil download",
                             lambda: self._get_stats(output_path))
            stats = self._last_step_result
            
            # 🎉 Complete
            duration = time.time() - start_time
            total_images = stats.get('total_images', 0)
            success_message = f"Download selesai: {total_images} gambar dalam {duration:.1f}s"
            
            self.progress_bridge.notify_complete(success_message, duration)
            
            if self.logger:
                self.logger.success(f"✅ {success_message}")
            
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
    
    def _execute_step(self, step_name: str, description: str, step_function) -> None:
        """Execute step dengan progress tracking yang akurat."""
        try:
            step_info = self.step_definitions.get(step_name, {'weight': 20, 'name': step_name})
            
            if self.logger:
                self.logger.info(f"📋 {step_info['name']}: {description}")
            
            # Start step progress
            self.progress_bridge.notify_step_progress(step_name, 0, f"Memulai {description.lower()}")
            
            # Execute step function
            result = step_function()
            self._last_step_result = result
            
            # Complete step
            self.progress_bridge.notify_step_progress(step_name, 100, f"{step_info['name']} selesai")
            
        except Exception as e:
            self.progress_bridge.notify_error(f"Error pada {description.lower()}: {str(e)}")
            raise
    
    def _step_aware_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Enhanced callback yang aware terhadap step dan weight distribution."""
        try:
            # Map downloader steps ke progress steps kita
            step_mapping = {
                'download': 'download',
                'extract': 'extract'
            }
            
            mapped_step = step_mapping.get(step, step)
            
            if mapped_step in self.step_definitions:
                # Calculate step progress
                step_progress = int((current / total) * 100) if total > 0 else 0
                
                # Send step progress dengan proper weight
                self.progress_bridge.notify_step_progress(mapped_step, step_progress, message)
                
                # Log significant milestones untuk feedback
                if self.logger and step_progress % 25 == 0 and step_progress > 0:
                    step_name = self.step_definitions[mapped_step]['name']
                    self.logger.debug(f"🔄 {step_name}: {step_progress}% - {message}")
        
        except Exception as e:
            # Fallback progress notification jika calculation gagal
            if self.logger:
                self.logger.debug(f"Progress callback error: {str(e)}")
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Enhanced parameter validation dengan detailed feedback."""
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
        """Get dataset metadata dengan enhanced error handling dan logging."""
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
            
            # Log dataset info untuk user feedback
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
        """Get comprehensive dataset statistics dengan progress feedback."""
        stats = {'total_images': 0, 'total_labels': 0}
        
        try:
            # Count files per split dengan mini progress updates
            for i, split in enumerate(['train', 'valid', 'test']):
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
                
                # Mini progress update untuk step verify
                mini_progress = int(((i + 1) / 3) * 100)
                self.progress_bridge.notify_step_progress('verify', mini_progress, f"Memeriksa {split}")
            
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