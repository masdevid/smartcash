"""
File: smartcash/dataset/preprocessor/utils/file_processor.py
Deskripsi: Enhanced file processor dengan improved image handling dan consistency dengan augmentor
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from smartcash.common.logger import get_logger

class FileProcessor:
    """🔧 Enhanced file processor dengan improved image handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Enhanced configuration
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.default_quality = config.get('performance', {}).get('compression_level', 90)
    
    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """🖼️ Enhanced image reading dengan better error handling"""
        try:
            path = Path(image_path)
            
            if not path.exists():
                self.logger.warning(f"⚠️ File tidak ditemukan: {image_path}")
                return None
            
            if path.suffix.lower() not in self.supported_formats:
                self.logger.warning(f"⚠️ Format tidak didukung: {path.suffix}")
                return None
            
            # Try OpenCV first (faster)
            image = cv2.imread(str(path))
            if image is not None:
                # Convert BGR to RGB untuk consistency
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Fallback ke PIL
            with Image.open(path) as img:
                return np.array(img.convert('RGB'))
                
        except Exception as e:
            self.logger.error(f"❌ Error reading image {image_path}: {str(e)}")
            return None
    
    def write_image(self, image_path: str, image: np.ndarray, format_hint: str = None) -> bool:
        """💾 Enhanced image writing dengan format optimization"""
        try:
            output_path = Path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format
            if format_hint:
                file_format = format_hint.lower()
            else:
                file_format = output_path.suffix.lower()
            
            # Handle different data types
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Normalize untuk image saving
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Convert RGB back to BGR untuk OpenCV
            if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_uint8
            
            # Save dengan optimized settings
            if file_format in ['.jpg', '.jpeg']:
                success = cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.default_quality])
            elif file_format == '.png':
                success = cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                success = cv2.imwrite(str(output_path), image_bgr)
            
            if not success:
                raise Exception("OpenCV write failed")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error writing image {image_path}: {str(e)}")
            return False
    
    def save_image(self, image: np.ndarray, output_path: Path, quality: int = None) -> bool:
        """💾 Save image dengan enhanced quality control (backward compatibility)"""
        quality = quality or self.default_quality
        return self.write_image(str(output_path), image)
    
    def read_label_file(self, label_path: Path) -> List[List[float]]:
        """📋 Enhanced label file reading dengan validation"""
        try:
            if not label_path.exists():
                return []
            
            bboxes = []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty dan comment lines
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 5:  # Format YOLO: class x_center y_center width height
                            bbox = [float(part) for part in parts[:5]]
                            
                            # Basic validation
                            if 0 <= bbox[1] <= 1 and 0 <= bbox[2] <= 1 and 0 < bbox[3] <= 1 and 0 < bbox[4] <= 1:
                                bboxes.append(bbox)
                            else:
                                self.logger.warning(f"⚠️ Invalid bbox di {label_path.name} line {line_num}: {line}")
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"⚠️ Error parsing {label_path.name} line {line_num}: {str(e)}")
            
            return bboxes
            
        except Exception as e:
            self.logger.error(f"❌ Error reading label {label_path}: {str(e)}")
            return []
    
    def save_label_file(self, bboxes: List[List[float]], output_path: Path) -> bool:
        """💾 Enhanced label file saving dengan validation"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    if len(bbox) >= 5:
                        # Format: class_id x_center y_center width height
                        line = ' '.join(f'{val:.6f}' if i > 0 else f'{int(val)}' for i, val in enumerate(bbox[:5]))
                        f.write(line + '\n')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving label {output_path}: {str(e)}")
            return False
    
    def copy_file(self, src_path: Path, dst_path: Path, preserve_metadata: bool = True) -> bool:
        """📋 Enhanced file copying dengan metadata preservation"""
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if preserve_metadata:
                import shutil
                shutil.copy2(src_path, dst_path)
            else:
                import shutil
                shutil.copy(src_path, dst_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error copying {src_path} to {dst_path}: {str(e)}")
            return False
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """📊 Get enhanced image information"""
        try:
            path = Path(image_path)
            
            if not path.exists():
                return {'exists': False}
            
            # Get file stats
            stat = path.stat()
            file_size = stat.st_size
            
            # Try to get image dimensions
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format
            except Exception:
                # Fallback ke OpenCV
                img = cv2.imread(str(path))
                if img is not None:
                    height, width = img.shape[:2]
                    mode = 'BGR'
                    format_name = path.suffix.upper()
                else:
                    width = height = 0
                    mode = 'Unknown'
                    format_name = 'Unknown'
            
            return {
                'exists': True,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'width': width,
                'height': height,
                'mode': mode,
                'format': format_name,
                'extension': path.suffix.lower(),
                'aspect_ratio': round(width / height, 2) if height > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting image info {image_path}: {str(e)}")
            return {'exists': False, 'error': str(e)}
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    preserve_aspect_ratio: bool = True, interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        """🔧 Enhanced image resizing dengan aspect ratio handling (consistency dengan augmentor)"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            if preserve_aspect_ratio:
                # Calculate scale
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize
                resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
                
                # Pad to target size
                result = np.full((target_h, target_w, 3), 128, dtype=image.dtype)  # Gray padding
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                
                return result
            else:
                # Direct resize
                return cv2.resize(image, target_size, interpolation=interpolation)
                
        except Exception as e:
            self.logger.error(f"❌ Error resizing image: {str(e)}")
            return image

# Factory function
def create_file_processor(config: Dict[str, Any]) -> FileProcessor:
    """🏭 Factory untuk create enhanced file processor"""
    return FileProcessor(config)