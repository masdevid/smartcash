"""
Download operation for pretrained models.
"""

import os
import time
import urllib.request
from typing import Dict, Any

from .pretrained_base_operation import PretrainedBaseOperation
from smartcash.ui.model.pretrained.constants import DEFAULT_MODEL_URLS, PretrainedModelType


class PretrainedDownloadOperation(PretrainedBaseOperation):
    """Download operation for pretrained models."""
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute download operation with actual model downloads."""
        try:
            self.log(f"📁 Using models directory: {self.models_dir}", 'info')
            
            # Step 1: Check existing models
            self.log("🔍 Checking existing models", 'info')
            existing_models = self.check_existing_models(self.models_dir)
            
            # Step 2: Prepare download directory
            self.log("📁 Preparing download directory", 'info')
            os.makedirs(self.models_dir, exist_ok=True)
            
            downloaded_models = []
            
            # Step 3: Download YOLOv5s model
            yolo_path = os.path.join(self.models_dir, 'yolov5s.pt')
            location_context = 'drive' if '/drive/' in self.models_dir else 'local'
            
            if not os.path.exists(yolo_path) or 'yolov5s' not in existing_models:
                self.log("📥 Downloading YOLOv5s model", 'info')
                yolo_url = self.config.get('yolo_url') or DEFAULT_MODEL_URLS[PretrainedModelType.YOLOV5S.value]
                
                if yolo_url:
                    start_time = time.time()
                    urllib.request.urlretrieve(yolo_url, yolo_path)
                    end_time = time.time()
                    
                    # Get file size in MB
                    file_size_bytes = os.path.getsize(yolo_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    download_time = end_time - start_time
                    
                    downloaded_models.append('yolov5s')
                    self.log(f"✅ Pretrain YOLOv5 downloaded {file_size_mb:.1f}MB in {download_time:.1f}s", 'success')
                else:
                    self.log("⚠️ YOLOv5s URL not provided, skipping", 'warning')
            else:
                # Check file size for existing model
                if os.path.exists(yolo_path):
                    file_size_bytes = os.path.getsize(yolo_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    self.log(f"✅ Pretrained model exists on {location_context}, skipping download (YOLOv5s {file_size_mb:.1f}MB)", 'info')
            
            # Step 4: Download EfficientNet-B4 model
            efficientnet_path = os.path.join(self.models_dir, 'efficientnet_b4.pth')
            if not os.path.exists(efficientnet_path) or 'efficientnet_b4' not in existing_models:
                self.log("📥 Downloading EfficientNet-B4 model", 'info')
                efficientnet_url = self.config.get('efficientnet_url')
                
                if efficientnet_url:
                    # Download from custom URL
                    start_time = time.time()
                    urllib.request.urlretrieve(efficientnet_url, efficientnet_path)
                    end_time = time.time()
                    
                    # Get file size and timing
                    file_size_bytes = os.path.getsize(efficientnet_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    download_time = end_time - start_time
                    
                    downloaded_models.append('efficientnet_b4')
                    self.log(f"✅ Pretrain EfficientNet-B4 downloaded {file_size_mb:.1f}MB in {download_time:.1f}s", 'success')
                else:
                    # Use timm to download
                    try:
                        import timm
                        import torch
                        
                        start_time = time.time()
                        model = timm.create_model('efficientnet_b4', pretrained=True)
                        torch.save(model.state_dict(), efficientnet_path)
                        end_time = time.time()
                        
                        # Get file size and timing
                        file_size_bytes = os.path.getsize(efficientnet_path)
                        file_size_mb = file_size_bytes / (1024 * 1024)
                        download_time = end_time - start_time
                        
                        downloaded_models.append('efficientnet_b4')
                        self.log(f"✅ Pretrain EfficientNet-B4 downloaded {file_size_mb:.1f}MB in {download_time:.1f}s (via timm)", 'success')
                    except Exception as e:
                        self.log(f"⚠️ Failed to download EfficientNet-B4 via timm: {e}", 'warning')
            else:
                # Check file size for existing model
                if os.path.exists(efficientnet_path):
                    file_size_bytes = os.path.getsize(efficientnet_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    self.log(f"✅ Pretrained model exists on {location_context}, skipping download (EfficientNet-B4 {file_size_mb:.1f}MB)", 'info')
            
            # Step 5: Validate downloaded models
            self.log("🔍 Validating downloaded models", 'info')
            validation_results = self.validate_downloaded_models(self.models_dir)
            
            # Step 6: Complete
            self.log("✅ Download complete", 'success')
            
            return {
                'success': True,
                'message': f'Model download completed. Downloaded: {", ".join(downloaded_models) if downloaded_models else "No new models"}',
                'models_downloaded': downloaded_models,
                'validation_results': validation_results
            }
            
        except Exception as e:
            self.log(f"❌ Error in download operation: {e}", 'error')
            return {'success': False, 'error': str(e)}