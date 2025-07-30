"""
Pretrained Weights Management for YOLOv5 Integration
Handles downloading and managing pretrained model weights
"""

import os
import shutil
from pathlib import Path
from smartcash.common.logger import SmartCashLogger


class YOLOv5PretrainedWeights:
    """
    Manages pretrained weights for YOLOv5 models
    """
    
    def __init__(self, logger=None):
        """
        Initialize pretrained weights manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.pretrained_dir = self.project_root / "data" / "pretrained"
        
        # Ensure directory exists
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    def get_weights_path(self, weights_name="yolov5s.pt"):
        """
        Get path for pretrained weights, handling download if necessary
        
        Args:
            weights_name: Name of the weights file (e.g., 'yolov5s.pt')
            
        Returns:
            Path to the weights file in /data/pretrained/ folder
        """
        weights_path = self.pretrained_dir / weights_name
        
        # If weights file doesn't exist, let YOLOv5 download it to our directory
        if not weights_path.exists():
            self.logger.info(f"📥 Downloading pretrained weights {weights_name} to {self.pretrained_dir}")
            self._download_weights(weights_name, weights_path)
        
        self.logger.info(f"📂 Using pretrained weights from {weights_path}")
        return str(weights_path)
    
    def _download_weights(self, weights_name, weights_path):
        """
        Download pretrained weights using YOLOv5's download utilities
        
        Args:
            weights_name: Name of the weights file
            weights_path: Target path for the weights file
        """
        try:
            import sys
            yolov5_path = self.project_root / "yolov5"
            if str(yolov5_path) not in sys.path:
                sys.path.append(str(yolov5_path))
            
            from utils.downloads import attempt_download
            
            # Change to the pretrained directory and download there
            original_cwd = Path.cwd()
            try:
                os.chdir(str(self.pretrained_dir))
                
                # Download to current directory (pretrained_dir)
                downloaded_path = attempt_download(weights_name)
                
                # Check if the file was downloaded to the pretrained directory
                if not weights_path.exists():
                    # If not in pretrained dir, try to find and move it
                    possible_locations = [
                        original_cwd / weights_name,  # Original directory
                        Path.home() / '.cache' / 'torch' / 'hub' / weights_name,  # Torch cache
                        yolov5_path / weights_name,  # YOLOv5 directory
                    ]
                    
                    for possible_path in possible_locations:
                        if possible_path.exists():
                            shutil.move(str(possible_path), str(weights_path))
                            self.logger.info(f"📁 Moved {weights_name} to {weights_path}")
                            break
                    else:
                        self.logger.warning(f"⚠️ Could not locate downloaded {weights_name}")
                        
            finally:
                # Always restore original working directory
                os.chdir(str(original_cwd))
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import YOLOv5 download utilities: {e}")
            # Fallback: return the path anyway, YOLOv5 will handle downloading
        except Exception as e:
            self.logger.warning(f"⚠️ Error downloading pretrained weights: {e}")
            # Fallback: return the path anyway