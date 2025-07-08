"""
File: smartcash/ui/model/pretrained/services/pretrained_service.py
Service layer for pretrained model operations - bridges UI with backend functionality.
"""

import os
import asyncio
import requests
import torch
import timm
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor

from ..constants import (
    PretrainedModelType, 
    DEFAULT_MODELS_DIR, 
    DEFAULT_MODEL_URLS, 
    EXPECTED_FILE_SIZES,
    VALIDATION_CONFIG,
    MODEL_INFO
)


class PretrainedService:
    """
    Service class for pretrained model operations.
    Handles downloading YOLOv5s and EfficientNet-B4 models.
    """
    
    def __init__(self):
        """Initialize the pretrained service."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.download_sessions = {}
        self.backup_files = {}  # Track backup files for restoration
    
    async def check_existing_models(self, models_dir: str, progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Check which pretrained models already exist in the specified directory.
        
        Args:
            models_dir: Directory to check for existing models
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dictionary with model existence status and file info
        """
        if log_callback:
            log_callback("🔍 Checking existing pretrained models...")
        
        result = {
            "models_dir": models_dir,
            "models_found": [],
            "models_missing": [],
            "total_found": 0,
            "all_present": False
        }
        
        try:
            # Ensure directory exists
            os.makedirs(models_dir, exist_ok=True)
            models_path = Path(models_dir)
            
            for model_type in PretrainedModelType:
                model_info = MODEL_INFO[model_type.value]
                expected_file = f"{model_type.value}{model_info['file_extension']}"
                file_path = models_path / expected_file
                
                if file_path.exists() and file_path.is_file():
                    file_size = file_path.stat().st_size
                    result["models_found"].append({
                        "model_type": model_type.value,
                        "name": model_info["name"],
                        "file_path": str(file_path),
                        "file_size": file_size,
                        "file_size_mb": round(file_size / (1024 * 1024), 2)
                    })
                    if log_callback:
                        log_callback(f"✅ Found {model_info['name']}: {file_size / (1024 * 1024):.1f} MB")
                else:
                    result["models_missing"].append({
                        "model_type": model_type.value,
                        "name": model_info["name"],
                        "expected_file": expected_file
                    })
                    if log_callback:
                        log_callback(f"❌ Missing {model_info['name']}: {expected_file}")
            
            result["total_found"] = len(result["models_found"])
            result["all_present"] = len(result["models_missing"]) == 0
            
            if progress_callback:
                progress_callback(100, f"Check complete: {result['total_found']}/{len(PretrainedModelType)} models found")
                
            return result
            
        except Exception as e:
            error_msg = f"Error checking existing models: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            raise
    
    async def download_yolov5s(self, models_dir: str, custom_url: Optional[str] = None, progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> bool:
        """
        Download YOLOv5s model from GitHub releases.
        
        Args:
            models_dir: Directory to save the model
            custom_url: Optional custom URL to download from
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            True if download successful, False otherwise
        """
        model_type = PretrainedModelType.YOLOV5S
        url = custom_url or DEFAULT_MODEL_URLS[model_type.value]
        
        if not url:
            if log_callback:
                log_callback("❌ No URL provided for YOLOv5s download")
            return False
        
        if log_callback:
            log_callback(f"📥 Starting YOLOv5s download from GitHub...")
        
        try:
            # Prepare file path
            filename = f"{model_type.value}.pt"
            file_path = Path(models_dir) / filename
            
            # Create backup of existing file
            if not self._backup_existing_file(str(file_path), log_callback):
                if log_callback:
                    log_callback("❌ Failed to backup existing YOLOv5s file")
                return False
            
            # Download with progress tracking
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._download_file_with_progress,
                url, str(file_path), progress_callback, log_callback
            )
            
            if success:
                # Validate downloaded file
                if self._validate_model_file(str(file_path), model_type.value):
                    # Download successful - cleanup backup
                    self._cleanup_backup_file(str(file_path), log_callback)
                    if log_callback:
                        log_callback(f"✅ YOLOv5s downloaded and validated successfully")
                    return True
                else:
                    # Validation failed - restore backup
                    self._restore_backup_file(str(file_path), log_callback)
                    if log_callback:
                        log_callback("❌ YOLOv5s download validation failed - restored backup")
                    return False
            else:
                # Download failed - restore backup
                self._restore_backup_file(str(file_path), log_callback)
                if log_callback:
                    log_callback("❌ YOLOv5s download failed - restored backup")
                return False
                
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Error downloading YOLOv5s: {str(e)}")
            return False
    
    async def download_efficientnet_b4(self, models_dir: str, custom_url: Optional[str] = None, progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> bool:
        """
        Download EfficientNet-B4 model using timm or custom URL.
        
        Args:
            models_dir: Directory to save the model
            custom_url: Optional custom URL to download from
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            True if download successful, False otherwise
        """
        model_type = PretrainedModelType.EFFICIENTNET_B4
        
        if log_callback:
            log_callback(f"📥 Starting EfficientNet-B4 download...")
        
        try:
            filename = f"{model_type.value}.pth"
            file_path = Path(models_dir) / filename
            
            # Create backup of existing file
            if not self._backup_existing_file(str(file_path), log_callback):
                if log_callback:
                    log_callback("❌ Failed to backup existing EfficientNet-B4 file")
                return False
            
            if custom_url:
                # Download from custom URL
                if log_callback:
                    log_callback(f"📥 Downloading from custom URL: {custom_url}")
                
                success = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._download_file_with_progress,
                    custom_url, str(file_path), progress_callback, log_callback
                )
                
                if not success:
                    if log_callback:
                        log_callback("❌ Custom URL download failed, falling back to timm...")
                    return await self._download_efficientnet_via_timm(models_dir, progress_callback, log_callback)
            else:
                # Use timm to download
                return await self._download_efficientnet_via_timm(models_dir, progress_callback, log_callback)
                
            if self._validate_model_file(str(file_path), model_type.value):
                # Download successful - cleanup backup
                self._cleanup_backup_file(str(file_path), log_callback)
                if log_callback:
                    log_callback(f"✅ EfficientNet-B4 downloaded and validated successfully")
                return True
            else:
                # Validation failed - restore backup
                self._restore_backup_file(str(file_path), log_callback)
                if log_callback:
                    log_callback("❌ EfficientNet-B4 download validation failed - restored backup")
                return False
                
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Error downloading EfficientNet-B4: {str(e)}")
            return False
    
    async def _download_efficientnet_via_timm(self, models_dir: str, progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> bool:
        """
        Download EfficientNet-B4 using timm library.
        
        Args:
            models_dir: Directory to save the model
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            if log_callback:
                log_callback("📥 Downloading EfficientNet-B4 via timm...")
            
            if progress_callback:
                progress_callback(10, "Loading EfficientNet-B4 via timm...")
            
            # Download model using timm
            def download_timm_model():
                model = timm.create_model('efficientnet_b4', pretrained=True)
                return model
            
            model = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                download_timm_model
            )
            
            if progress_callback:
                progress_callback(70, "Saving EfficientNet-B4 model...")
            
            # Save model to file
            filename = f"{PretrainedModelType.EFFICIENTNET_B4.value}.pth"
            file_path = Path(models_dir) / filename
            
            # Create backup of existing file
            if not self._backup_existing_file(str(file_path), log_callback):
                if log_callback:
                    log_callback("❌ Failed to backup existing EfficientNet-B4 file")
                return False
            
            try:
                torch.save(model.state_dict(), str(file_path))
                
                # Download successful - cleanup backup
                self._cleanup_backup_file(str(file_path), log_callback)
                
                if progress_callback:
                    progress_callback(100, "EfficientNet-B4 download complete")
                
                if log_callback:
                    log_callback(f"✅ EfficientNet-B4 saved via timm to {filename}")
                
                return True
                
            except Exception as save_error:
                # Save failed - restore backup
                self._restore_backup_file(str(file_path), log_callback)
                if log_callback:
                    log_callback(f"❌ Failed to save EfficientNet-B4 model: {str(save_error)} - restored backup")
                return False
            
        except Exception as e:
            # If we get here, the model download/creation failed before saving
            # Try to restore backup if one exists
            filename = f"{PretrainedModelType.EFFICIENTNET_B4.value}.pth"
            file_path = Path(models_dir) / filename
            self._restore_backup_file(str(file_path), log_callback)
            
            if log_callback:
                log_callback(f"❌ Error downloading EfficientNet-B4 via timm: {str(e)} - restored backup")
            return False
    
    def _download_file_with_progress(self, url: str, file_path: str, progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> bool:
        """
        Download a file with progress tracking (synchronous, runs in executor).
        
        Args:
            url: URL to download from
            file_path: Local path to save the file
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            if progress_callback:
                progress_callback(5, "Starting download...")
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every 1MB or on completion
                        if total_size > 0 and (downloaded % (1024 * 1024) == 0 or downloaded == total_size):
                            progress = min(95, int((downloaded / total_size) * 90) + 5)  # 5-95%
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            message = f"Downloaded {size_mb:.1f}/{total_mb:.1f} MB"
                            if progress_callback:
                                progress_callback(progress, message)
            
            if progress_callback:
                progress_callback(100, "Download complete")
            
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Download error: {str(e)}")
            # Clean up partial file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return False
    
    def _validate_model_file(self, file_path: str, model_type: str) -> bool:
        """
        Validate a downloaded model file.
        
        Args:
            file_path: Path to the model file
            model_type: Type of model being validated
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            if not os.path.isfile(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            
            # Check minimum size
            if file_size < VALIDATION_CONFIG["min_file_size"]:
                return False
            
            # Check expected size (with tolerance)
            expected_size = EXPECTED_FILE_SIZES.get(model_type, 0)
            if expected_size > 0:
                tolerance = VALIDATION_CONFIG["size_tolerance"]
                min_size = expected_size * (1 - tolerance)
                max_size = expected_size * (1 + tolerance)
                if not (min_size <= file_size <= max_size):
                    return False
            
            # Check file extension
            model_info = MODEL_INFO[model_type]
            expected_ext = model_info["file_extension"]
            if not file_path.lower().endswith(expected_ext.lower()):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _backup_existing_file(self, file_path: str, log_callback: Optional[Callable] = None) -> bool:
        """
        Create backup of existing file by renaming it to .bak extension.
        
        Args:
            file_path: Path to the file to backup
            log_callback: Optional callback for logging
            
        Returns:
            True if backup successful or file doesn't exist, False if backup failed
        """
        try:
            if not os.path.exists(file_path):
                return True  # No file to backup
            
            backup_path = f"{file_path}.bak"
            
            # Remove existing backup if it exists
            if os.path.exists(backup_path):
                os.remove(backup_path)
                if log_callback:
                    log_callback(f"🗑️ Removed old backup: {os.path.basename(backup_path)}")
            
            # Create backup
            os.rename(file_path, backup_path)
            self.backup_files[file_path] = backup_path
            
            if log_callback:
                log_callback(f"💾 Backed up existing file: {os.path.basename(file_path)} → {os.path.basename(backup_path)}")
            
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Failed to backup {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def _restore_backup_file(self, file_path: str, log_callback: Optional[Callable] = None) -> bool:
        """
        Restore backup file by renaming .bak file back to original name.
        
        Args:
            file_path: Original file path to restore
            log_callback: Optional callback for logging
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            backup_path = self.backup_files.get(file_path)
            if not backup_path or not os.path.exists(backup_path):
                return False
            
            # Remove failed download if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Restore backup
            os.rename(backup_path, file_path)
            
            if log_callback:
                log_callback(f"🔄 Restored backup: {os.path.basename(backup_path)} → {os.path.basename(file_path)}")
            
            # Remove from backup tracking
            del self.backup_files[file_path]
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Failed to restore backup for {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def _cleanup_backup_file(self, file_path: str, log_callback: Optional[Callable] = None) -> bool:
        """
        Remove backup file after successful download.
        
        Args:
            file_path: Original file path whose backup should be cleaned up
            log_callback: Optional callback for logging
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            backup_path = self.backup_files.get(file_path)
            if not backup_path or not os.path.exists(backup_path):
                return True  # No backup to clean up
            
            os.remove(backup_path)
            
            if log_callback:
                log_callback(f"🧹 Cleaned up backup: {os.path.basename(backup_path)}")
            
            # Remove from backup tracking
            del self.backup_files[file_path]
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Failed to cleanup backup for {os.path.basename(file_path)}: {str(e)}")
            return False
    
    async def download_all_models(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Download all pretrained models (YOLOv5s and EfficientNet-B4).
        
        Args:
            config: Configuration containing models_dir and optional custom URLs
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dictionary with download results
        """
        models_dir = config.get("models_dir", DEFAULT_MODELS_DIR)
        custom_urls = config.get("model_urls", {})
        
        if log_callback:
            log_callback("🚀 Starting pretrained models download...")
        
        # Ensure directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        results = {
            "models_dir": models_dir,
            "downloads": [],
            "success_count": 0,
            "total_count": 2,
            "all_successful": False
        }
        
        try:
            # Download YOLOv5s
            if progress_callback:
                progress_callback(10, "Downloading YOLOv5s...")
            
            yolo_url = custom_urls.get(PretrainedModelType.YOLOV5S.value)
            yolo_success = await self.download_yolov5s(models_dir, yolo_url, progress_callback, log_callback)
            
            results["downloads"].append({
                "model": "YOLOv5s",
                "success": yolo_success,
                "url_used": yolo_url or DEFAULT_MODEL_URLS[PretrainedModelType.YOLOV5S.value]
            })
            
            if yolo_success:
                results["success_count"] += 1
            
            # Download EfficientNet-B4
            if progress_callback:
                progress_callback(50, "Downloading EfficientNet-B4...")
            
            efficientnet_url = custom_urls.get(PretrainedModelType.EFFICIENTNET_B4.value)
            efficientnet_success = await self.download_efficientnet_b4(models_dir, efficientnet_url, progress_callback, log_callback)
            
            results["downloads"].append({
                "model": "EfficientNet-B4",
                "success": efficientnet_success,
                "url_used": efficientnet_url or "timm library"
            })
            
            if efficientnet_success:
                results["success_count"] += 1
            
            results["all_successful"] = results["success_count"] == results["total_count"]
            
            if progress_callback:
                progress_callback(100, f"Download complete: {results['success_count']}/{results['total_count']} models")
            
            if log_callback:
                if results["all_successful"]:
                    log_callback("✅ All pretrained models downloaded successfully!")
                else:
                    log_callback(f"⚠️ {results['success_count']}/{results['total_count']} models downloaded successfully")
            
            return results
            
        except Exception as e:
            error_msg = f"Error during model downloads: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            results["error"] = error_msg
            return results
    
    def get_models_summary(self, models_dir: str) -> Dict[str, Any]:
        """
        Get summary information about available models.
        
        Args:
            models_dir: Directory to check for models
            
        Returns:
            Dictionary with models summary
        """
        summary = {
            "models_dir": models_dir,
            "available_models": [],
            "total_size_mb": 0,
            "models_count": 0
        }
        
        try:
            models_path = Path(models_dir)
            if not models_path.exists():
                return summary
            
            for model_type in PretrainedModelType:
                model_info = MODEL_INFO[model_type.value]
                expected_file = f"{model_type.value}{model_info['file_extension']}"
                file_path = models_path / expected_file
                
                if file_path.exists() and file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    
                    summary["available_models"].append({
                        "name": model_info["name"],
                        "type": model_type.value,
                        "file_path": str(file_path),
                        "size_mb": round(file_size_mb, 2),
                        "source": model_info["source"]
                    })
                    
                    summary["total_size_mb"] += file_size_mb
                    summary["models_count"] += 1
            
            summary["total_size_mb"] = round(summary["total_size_mb"], 2)
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary