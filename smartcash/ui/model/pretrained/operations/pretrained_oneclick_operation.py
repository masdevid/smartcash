"""
One-click download operation for pretrained models.

This operation combines check existing -> download if missing -> validate -> cleanup invalid files.
"""

import os
import sys
import time
import urllib.request
from contextlib import contextmanager
from typing import Dict, Any, Optional
import io

from .pretrained_base_operation import PretrainedBaseOperation
from smartcash.ui.model.pretrained.constants import DEFAULT_MODEL_URLS, PretrainedModelType


class PretrainedOneClickOperation(PretrainedBaseOperation):
    """
    One-click download operation that handles the complete pretrained model workflow:
    1. Check existing models
    2. Download missing models 
    3. Validate all models
    4. Clean up invalid/corrupted files
    """
    
    @contextmanager
    def _suppress_console_output(self):
        """Context manager to suppress console output during downloads."""
        try:
            # Save original stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # Redirect to null
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            yield
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute one-click operation with comprehensive model management."""
        try:
            self.update_progress(0, "🚀 Starting one-click pretrained model setup...")
            location_context = 'drive' if '/drive/' in self.models_dir else 'local'
            
            # Phase 1: Initial setup and directory preparation (5%)
            self.update_progress(5, "📁 Preparing models directory...")
            self.log(f"📁 Using models directory: {self.models_dir}", 'info')
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Phase 2: Check existing models (10%)
            self.update_progress(10, "🔍 Checking existing models...")
            self.log("🔍 Checking existing models", 'info')
            existing_models = self.check_existing_models(self.models_dir)
            
            # Track operation results
            downloaded_models = []
            validation_results = {}
            cleaned_files = []
            
            # Phase 3: Download YOLOv5s if needed (15-45%)
            yolo_path = os.path.join(self.models_dir, 'yolov5s.pt')
            if not os.path.exists(yolo_path) or 'yolov5s' not in existing_models:
                self.update_progress(15, "📥 Downloading YOLOv5s model...")
                result = self._download_yolo_model(yolo_path)
                if result['success']:
                    downloaded_models.append('yolov5s')
                    self.log(result['message'], 'success')
                else:
                    self.log(result['message'], 'warning')
            else:
                self.update_progress(30, "✅ YOLOv5s model already exists")
                file_size_mb = os.path.getsize(yolo_path) / (1024 * 1024)
                self.log(f"✅ YOLOv5s model exists on {location_context} ({file_size_mb:.1f}MB)", 'info')
            
            # Phase 4: Download EfficientNet-B4 if needed (45-75%)
            efficientnet_path = os.path.join(self.models_dir, 'efficientnet_b4.pth')
            if not os.path.exists(efficientnet_path) or 'efficientnet_b4' not in existing_models:
                self.update_progress(45, "📥 Downloading EfficientNet-B4 model...")
                result = self._download_efficientnet_model(efficientnet_path)
                if result['success']:
                    downloaded_models.append('efficientnet_b4')
                    self.log(result['message'], 'success')
                else:
                    self.log(result['message'], 'warning')
            else:
                self.update_progress(60, "✅ EfficientNet-B4 model already exists")
                file_size_mb = os.path.getsize(efficientnet_path) / (1024 * 1024)
                self.log(f"✅ EfficientNet-B4 model exists on {location_context} ({file_size_mb:.1f}MB)", 'info')
            
            # Phase 5: Validate all models (75-85%)
            self.update_progress(75, "🔍 Validating all models...")
            self.log("🔍 Validating all models", 'info')
            validation_results = self.validate_downloaded_models(self.models_dir)
            
            # Log validation results
            valid_models = []
            invalid_models = []
            for model_name, result in validation_results.items():
                if result.get('valid', False):
                    valid_models.append(model_name)
                    file_size_mb = result.get('size_mb', 0)
                    self.log(f"✅ {model_name} validation passed ({file_size_mb:.1f}MB)", 'info')
                else:
                    invalid_models.append(model_name)
                    error_msg = result.get('error', 'Unknown validation error')
                    self.log(f"❌ {model_name} validation failed: {error_msg}", 'warning')
            
            # Phase 6: Clean up invalid files (85-95%)
            if invalid_models:
                self.update_progress(85, "🗑️ Cleaning up invalid models...")
                cleanup_result = self._cleanup_invalid_models(validation_results)
                cleaned_files = cleanup_result['cleaned_files']
                self.log(cleanup_result['message'], 'info')
            else:
                self.update_progress(90, "✅ No cleanup needed - all models valid")
                self.log("✅ No invalid models found, cleanup not needed", 'info')
            
            # Phase 7: Final validation and completion (95-100%)
            self.update_progress(95, "🔍 Final validation...")
            if invalid_models:
                # Re-validate after cleanup
                final_validation = self.validate_downloaded_models(self.models_dir)
                final_valid = [m for m, r in final_validation.items() if r.get('valid', False)]
                final_invalid = [m for m, r in final_validation.items() if not r.get('valid', False)]
            else:
                final_valid = valid_models
                final_invalid = []
            
            # Complete operation
            self.update_progress(100, "✅ One-click setup complete!")
            
            # Generate summary
            total_models = len(final_valid) + len(final_invalid)
            success_rate = (len(final_valid) / total_models * 100) if total_models > 0 else 100
            
            summary_parts = []
            if downloaded_models:
                summary_parts.append(f"Downloaded: {', '.join(downloaded_models)}")
            if cleaned_files:
                summary_parts.append(f"Cleaned: {len(cleaned_files)} invalid files")
            
            summary = f"Setup complete! {len(final_valid)}/{total_models} models ready ({success_rate:.0f}% success)"
            if summary_parts:
                summary += f" | {' | '.join(summary_parts)}"
            
            self.log(f"🎉 {summary}", 'success')
            
            return {
                'success': True,
                'message': summary,
                'models_downloaded': downloaded_models,
                'models_cleaned': cleaned_files,
                'validation_results': validation_results,
                'final_valid_models': final_valid,
                'final_invalid_models': final_invalid,
                'success_rate': success_rate
            }
            
        except Exception as e:
            self.update_progress(100, f"❌ One-click setup failed: {e}")
            self.log(f"❌ Error in one-click operation: {e}", 'error')
            return {'success': False, 'error': str(e), 'message': f'One-click setup failed: {e}'}
    
    def _download_yolo_model(self, yolo_path: str) -> Dict[str, Any]:
        """Download YOLOv5s model with progress tracking."""
        try:
            yolo_url = self.config.get('yolo_url') or DEFAULT_MODEL_URLS[PretrainedModelType.YOLOV5S.value]
            
            if not yolo_url:
                return {'success': False, 'message': '⚠️ YOLOv5s URL not provided, skipping'}
            
            self.log("📥 Downloading YOLOv5s model...", 'info')
            start_time = time.time()
            
            # Update progress during download
            self.update_progress(
                progress=20,
                message="📥 Downloading YOLOv5s model...",
                secondary_progress=0,
                secondary_message="Starting download..."
            )
            
            with self._suppress_console_output():
                urllib.request.urlretrieve(yolo_url, yolo_path)
            
            end_time = time.time()
            
            # Get file size and calculate metrics
            file_size_bytes = os.path.getsize(yolo_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            download_time = end_time - start_time
            
            self.update_progress(
                progress=30,
                message="✅ YOLOv5s download complete",
                secondary_progress=100,
                secondary_message=f"Downloaded {file_size_mb:.1f}MB"
            )
            
            return {
                'success': True,
                'message': f'✅ YOLOv5s downloaded {file_size_mb:.1f}MB in {download_time:.1f}s'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'❌ YOLOv5s download failed: {e}'}
    
    def _download_efficientnet_model(self, efficientnet_path: str) -> Dict[str, Any]:
        """Download EfficientNet-B4 model with progress tracking."""
        try:
            efficientnet_url = self.config.get('efficientnet_url')
            
            self.update_progress(
                progress=50,
                message="📥 Downloading EfficientNet-B4 model...",
                secondary_progress=0,
                secondary_message="Starting download..."
            )
            
            start_time = time.time()
            
            if efficientnet_url:
                # Download from custom URL
                self.log("📥 Downloading EfficientNet-B4 from custom URL...", 'info')
                with self._suppress_console_output():
                    urllib.request.urlretrieve(efficientnet_url, efficientnet_path)
            else:
                # Use timm to download
                self.log("📥 Downloading EfficientNet-B4 via timm...", 'info')
                import timm
                import torch
                
                with self._suppress_console_output():
                    model = timm.create_model('efficientnet_b4', pretrained=True)
                    torch.save(model.state_dict(), efficientnet_path)
            
            end_time = time.time()
            
            # Get file size and calculate metrics
            file_size_bytes = os.path.getsize(efficientnet_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            download_time = end_time - start_time
            
            source = "custom URL" if efficientnet_url else "timm"
            
            self.update_progress(
                progress=60,
                message="✅ EfficientNet-B4 download complete",
                secondary_progress=100,
                secondary_message=f"Downloaded {file_size_mb:.1f}MB"
            )
            
            return {
                'success': True,
                'message': f'✅ EfficientNet-B4 downloaded {file_size_mb:.1f}MB in {download_time:.1f}s (via {source})'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'❌ EfficientNet-B4 download failed: {e}'}
    
    def _cleanup_invalid_models(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up invalid/corrupted model files."""
        try:
            cleaned_files = []
            space_freed = 0
            removed_models = []
            
            for model_name, result in validation_results.items():
                if not result.get('valid', False):
                    file_path = result.get('file_path')
                    if file_path and os.path.exists(file_path):
                        try:
                            file_size = os.path.getsize(file_path)
                            file_size_mb = file_size / (1024 * 1024)
                            
                            os.remove(file_path)
                            
                            cleaned_files.append(file_path)
                            removed_models.append(f"{model_name} ({file_size_mb:.1f}MB)")
                            space_freed += file_size
                            
                            self.log(f"🗑️ Removed invalid {model_name} ({file_size_mb:.1f}MB)", 'info')
                        except Exception as e:
                            self.log(f"⚠️ Failed to remove {model_name}: {e}", 'warning')
            
            # Clean up empty directories
            if os.path.exists(self.models_dir):
                try:
                    for root, dirs, _ in os.walk(self.models_dir, topdown=False):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            if os.path.exists(dir_path) and not os.listdir(dir_path):
                                os.rmdir(dir_path)
                                self.log(f"📁 Removed empty directory: {dir_path}", 'info')
                except Exception as e:
                    self.log(f"⚠️ Error cleaning directories: {e}", 'warning')
            
            space_freed_mb = space_freed / (1024 * 1024)
            message = f"🗑️ Cleanup complete: {len(cleaned_files)} files removed ({space_freed_mb:.1f}MB freed)"
            
            return {
                'success': True,
                'message': message,
                'cleaned_files': cleaned_files,
                'removed_models': removed_models,
                'space_freed_mb': space_freed_mb
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ Cleanup failed: {e}',
                'cleaned_files': [],
                'removed_models': [],
                'space_freed_mb': 0
            }
    
    def update_progress(
        self, 
        progress: int, 
        message: str = "", 
        secondary_progress: Optional[int] = None, 
        secondary_message: str = ""
    ) -> None:
        """
        Update dual progress tracker for pretrained operations.
        
        Args:
            progress: Main progress percentage (0-100)
            message: Main progress message
            secondary_progress: Secondary progress percentage (0-100), optional
            secondary_message: Secondary progress message
        """
        try:
            # Get operation container from UI module
            if hasattr(self, 'ui_module') and self.ui_module:
                operation_container = self.ui_module.get_component('operation_container')
                
                if operation_container and hasattr(operation_container, 'get'):
                    update_func = operation_container.get('update_progress')
                    if update_func:
                        # Call with dual progress parameters
                        if secondary_progress is not None:
                            update_func(
                                progress=max(0, min(100, progress)),
                                message=message,
                                secondary_progress=max(0, min(100, secondary_progress)),
                                secondary_message=secondary_message
                            )
                        else:
                            # Single progress mode
                            update_func(progress=max(0, min(100, progress)), message=message)
                        
        except Exception as e:
            # Fail silently on progress update errors to not break operations
            if hasattr(self, 'logger'):
                self.logger.debug(f"Progress update failed: {e}")