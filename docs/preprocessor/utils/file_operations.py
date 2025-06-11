"""
File: smartcash/dataset/preprocessor/utils/file_operations.py
Deskripsi: Konsolidasi operasi file dengan image I/O, scanning, dan batch operations yang optimized
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

class FileOperations:
    """🔧 Konsolidasi operasi file dengan optimasi performa dan error handling"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Enhanced configuration
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_label_formats = {'.txt'}
        self.default_quality = self.config.get('compression_level', 90)
        self.max_workers = get_optimal_thread_count('io')
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 32)
        self.use_threading = self.config.get('use_threading', True)
    
    # === IMAGE I/O OPERATIONS ===
    
    def read_image(self, image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """🖼️ Enhanced image reading dengan auto format detection dan optional resizing"""
        try:
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in self.supported_image_formats:
                return None
            
            # Try OpenCV first (faster untuk most formats)
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self._apply_target_size(image, target_size) if target_size else image
            
            # Fallback ke PIL untuk better format support
            with Image.open(path) as img:
                image = np.array(img.convert('RGB'))
                return self._apply_target_size(image, target_size) if target_size else image
                
        except Exception as e:
            self.logger.debug(f"❌ Error reading {image_path}: {str(e)}")
            return None
    
    def write_image(self, image_path: Union[str, Path], image: np.ndarray, 
                   quality: Optional[int] = None, preserve_metadata: bool = False) -> bool:
        """💾 Enhanced image writing dengan format optimization"""
        try:
            output_path = Path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize image data
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Convert RGB to BGR untuk OpenCV
            if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_uint8
            
            # Format-specific optimization
            file_format = output_path.suffix.lower()
            quality = quality or self.default_quality
            
            if file_format in ['.jpg', '.jpeg']:
                return cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif file_format == '.png':
                compression = max(0, min(9, 9 - (quality // 10)))  # Convert quality to compression
                return cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            else:
                return cv2.imwrite(str(output_path), image_bgr)
                
        except Exception as e:
            self.logger.error(f"❌ Error writing {image_path}: {str(e)}")
            return False
    
    def save_normalized_array(self, array_path: Union[str, Path], array: np.ndarray, 
                            metadata: Optional[Dict] = None) -> bool:
        """💾 Save normalized array sebagai .npy dengan optional metadata"""
        try:
            output_path = Path(array_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save array
            np.save(output_path, array.astype(np.float32))
            
            # Save metadata jika ada
            if metadata:
                meta_path = output_path.with_suffix('.meta.npy')
                np.save(meta_path, metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving array {array_path}: {str(e)}")
            return False
    
    def load_normalized_array(self, array_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """📂 Load normalized array dengan metadata"""
        try:
            path = Path(array_path)
            if not path.exists():
                return None, None
            
            # Load array
            array = np.load(path)
            
            # Load metadata jika ada
            meta_path = path.with_suffix('.meta.npy')
            metadata = np.load(meta_path, allow_pickle=True).item() if meta_path.exists() else None
            
            return array, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error loading array {array_path}: {str(e)}")
            return None, None
    
    # === LABEL OPERATIONS ===
    
    def read_yolo_label(self, label_path: Union[str, Path]) -> List[List[float]]:
        """📋 Enhanced YOLO label reading dengan validation"""
        try:
            path = Path(label_path)
            if not path.exists():
                return []
            
            bboxes = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            bbox = [float(part) for part in parts[:5]]
                            # Basic validation
                            if self._is_valid_bbox(bbox):
                                bboxes.append(bbox)
                            else:
                                self.logger.debug(f"⚠️ Invalid bbox in {path.name}:{line_num}")
                    except (ValueError, IndexError):
                        self.logger.debug(f"⚠️ Parse error in {path.name}:{line_num}")
            
            return bboxes
            
        except Exception as e:
            self.logger.error(f"❌ Error reading label {label_path}: {str(e)}")
            return []
    
    def write_yolo_label(self, label_path: Union[str, Path], bboxes: List[List[float]]) -> bool:
        """💾 Enhanced YOLO label writing dengan validation"""
        try:
            output_path = Path(label_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    if len(bbox) >= 5 and self._is_valid_bbox(bbox):
                        # Format: class_id x_center y_center width height
                        line = ' '.join(f'{val:.6f}' if i > 0 else f'{int(val)}' for i, val in enumerate(bbox[:5]))
                        f.write(line + '\n')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error writing label {label_path}: {str(e)}")
            return False
    
    # === FILE SCANNING OPERATIONS ===
    
    def scan_images(self, directory: Union[str, Path], recursive: bool = False) -> List[Path]:
        """🔍 Scan directory untuk image files dengan performance optimization"""
        return self.scan_files(directory, self.supported_image_formats, recursive)
    
    def scan_labels(self, directory: Union[str, Path], recursive: bool = False) -> List[Path]:
        """🔍 Scan directory untuk label files"""
        return self.scan_files(directory, self.supported_label_formats, recursive)
    
    def scan_files(self, directory: Union[str, Path], extensions: Set[str], recursive: bool = False) -> List[Path]:
        """🔍 Generic file scanning dengan extension filtering"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists() or not dir_path.is_dir():
                return []
            
            files = []
            pattern = '**/*' if recursive else '*'
            
            for ext in extensions:
                files.extend(dir_path.glob(f'{pattern}{ext}'))
                files.extend(dir_path.glob(f'{pattern}{ext.upper()}'))  # Case insensitive
            
            return sorted(set(files))  # Remove duplicates dan sort
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {directory}: {str(e)}")
            return []
    
    def find_image_label_pairs(self, image_dir: Union[str, Path], 
                              label_dir: Union[str, Path]) -> List[Tuple[Path, Optional[Path]]]:
        """🔗 Find matching image-label pairs dengan efficient lookup"""
        try:
            img_files = self.scan_images(image_dir)
            label_files = self.scan_labels(label_dir)
            
            # Create efficient lookup map
            label_map = {f.stem: f for f in label_files}
            
            pairs = []
            for img_file in img_files:
                label_file = label_map.get(img_file.stem)
                pairs.append((img_file, label_file))
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"❌ Error finding pairs: {str(e)}")
            return []
    
    def find_orphan_files(self, image_dir: Union[str, Path], 
                         label_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """🔍 Find orphaned files (images without labels, labels without images)"""
        try:
            img_files = self.scan_images(image_dir)
            label_files = self.scan_labels(label_dir)
            
            img_stems = {f.stem for f in img_files}
            label_stems = {f.stem for f in label_files}
            
            return {
                'orphan_images': [f for f in img_files if f.stem not in label_stems],
                'orphan_labels': [f for f in label_files if f.stem not in img_stems]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error finding orphans: {str(e)}")
            return {'orphan_images': [], 'orphan_labels': []}
    
    # === BATCH OPERATIONS ===
    
    def batch_read_images(self, image_paths: List[Path], target_size: Optional[Tuple[int, int]] = None,
                         progress_callback: Optional[callable] = None) -> List[Tuple[Path, Optional[np.ndarray]]]:
        """📦 Batch image reading dengan threading dan progress tracking"""
        results = []
        total = len(image_paths)
        
        def read_single(i_path_tuple):
            i, path = i_path_tuple
            image = self.read_image(path, target_size)
            if progress_callback and i % max(1, total // 20) == 0:  # Report every 5%
                progress_callback('current', i + 1, total, f"Reading {path.name}")
            return (path, image)
        
        if self.use_threading and total > 10:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                indexed_paths = list(enumerate(image_paths))
                results = list(executor.map(read_single, indexed_paths))
        else:
            # Sequential untuk small batches
            for i, path in enumerate(image_paths):
                results.append(read_single((i, path)))
        
        return results
    
    def batch_copy_files(self, file_pairs: List[Tuple[Path, Path]], 
                        preserve_metadata: bool = True,
                        progress_callback: Optional[callable] = None) -> Dict[str, int]:
        """📦 Batch file copying dengan progress tracking"""
        stats = {'success': 0, 'failed': 0}
        total = len(file_pairs)
        
        def copy_single(i_pair_tuple):
            i, (src, dst) = i_pair_tuple
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if preserve_metadata:
                    shutil.copy2(src, dst)
                else:
                    shutil.copy(src, dst)
                
                if progress_callback and i % max(1, total // 20) == 0:
                    progress_callback('current', i + 1, total, f"Copying {src.name}")
                
                return True
            except Exception as e:
                self.logger.error(f"❌ Copy failed {src} -> {dst}: {str(e)}")
                return False
        
        if self.use_threading and total > 10:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                indexed_pairs = list(enumerate(file_pairs))
                results = list(executor.map(copy_single, indexed_pairs))
                stats['success'] = sum(results)
                stats['failed'] = total - stats['success']
        else:
            # Sequential
            for i, pair in enumerate(file_pairs):
                success = copy_single((i, pair))
                stats['success' if success else 'failed'] += 1
        
        return stats
    
    # === UTILITY METHODS ===
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """📊 Get comprehensive file information"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {'exists': False}
            
            stat = path.stat()
            info = {
                'exists': True,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'extension': path.suffix.lower(),
                'stem': path.stem,
                'name': path.name,
                'modified': stat.st_mtime
            }
            
            # Additional info untuk images
            if path.suffix.lower() in self.supported_image_formats:
                try:
                    with Image.open(path) as img:
                        info.update({
                            'width': img.width,
                            'height': img.height,
                            'mode': img.mode,
                            'format': img.format,
                            'aspect_ratio': round(img.width / img.height, 2) if img.height > 0 else 0
                        })
                except Exception:
                    info.update({'width': 0, 'height': 0, 'mode': 'unknown'})
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting file info {file_path}: {str(e)}")
            return {'exists': False, 'error': str(e)}
    
    def _apply_target_size(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """🔧 Apply target size dengan efficient resizing"""
        if image.shape[:2] == target_size[::-1]:  # Already correct size
            return image
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def _is_valid_bbox(self, bbox: List[float]) -> bool:
        """✅ Validate YOLO bbox coordinates"""
        if len(bbox) < 5:
            return False
        _, x, y, w, h = bbox[:5]
        return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1

# === FACTORY FUNCTIONS ===

def create_file_operations(config: Dict[str, Any] = None) -> FileOperations:
    """🏭 Factory untuk create FileOperations instance"""
    return FileOperations(config)

# === CONVENIENCE FUNCTIONS ===

def read_image_safe(image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """🖼️ One-liner safe image reading"""
    return create_file_operations().read_image(image_path, target_size)

def write_image_safe(image_path: Union[str, Path], image: np.ndarray, quality: int = 90) -> bool:
    """💾 One-liner safe image writing"""
    return create_file_operations().write_image(image_path, image, quality)

def scan_image_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """🔍 One-liner image file scanning"""
    return create_file_operations().scan_images(directory, recursive)

def find_pairs_safe(image_dir: Union[str, Path], label_dir: Union[str, Path]) -> List[Tuple[Path, Optional[Path]]]:
    """🔗 One-liner safe pair finding"""
    return create_file_operations().find_image_label_pairs(image_dir, label_dir)