"""
File: smartcash/dataset/preprocessor/core/file_processor.py
Deskripsi: File operations untuk preprocessing dengan focus pada I/O efficiency
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

class FileProcessor:
    """📁 Efficient file operations untuk preprocessing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # File handling settings
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.max_workers = self.config.get('max_workers', get_optimal_thread_count('io'))
        self.use_threading = self.config.get('use_threading', True)
    
    def read_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """🖼️ Read image dengan error handling"""
        try:
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in self.supported_image_formats:
                return None
            
            image = cv2.imread(str(path))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
            
        except Exception as e:
            self.logger.debug(f"❌ Error reading {image_path}: {str(e)}")
            return None
    
    def save_normalized_array(self, output_path: Union[str, Path], array: np.ndarray, 
                            metadata: Optional[Dict] = None) -> bool:
        """💾 Save normalized array sebagai .npy"""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save array dengan compression
            np.save(path, array.astype(np.float32))
            
            # Save metadata terpisah
            if metadata:
                meta_path = path.with_suffix('.meta.json')
                import json
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving {output_path}: {str(e)}")
            return False
    
    def load_normalized_array(self, array_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """📂 Load normalized array dengan metadata"""
        try:
            path = Path(array_path)
            if not path.exists():
                return None, None
            
            # Load array
            array = np.load(path)
            
            # Load metadata
            meta_path = path.with_suffix('.meta.json')
            metadata = None
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            return array, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {array_path}: {str(e)}")
            return None, None
    
    def scan_files(self, directory: Union[str, Path], pattern: str = None, 
                  extensions: set = None) -> List[Path]:
        """🔍 Scan files dengan pattern matching"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return []
            
            extensions = extensions or self.supported_image_formats
            
            files = []
            for ext in extensions:
                if pattern:
                    files.extend(dir_path.glob(f'{pattern}*{ext}'))
                else:
                    files.extend(dir_path.glob(f'*{ext}'))
            
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {directory}: {str(e)}")
            return []
    
    def scan_preprocessed_files(self, directory: Union[str, Path]) -> Dict[str, List[Path]]:
        """🔍 Scan preprocessed files by type"""
        dir_path = Path(directory)
        return {
            'raw': self.scan_files(dir_path, 'rp_'),
            'preprocessed': self.scan_files(dir_path, 'pre_', {'.npy'}),
            'augmented': self.scan_files(dir_path, 'aug_', {'.npy'}),
            'samples': self.scan_files(dir_path, 'sample_')
        }
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """📄 Copy single file dengan error handling"""
        try:
            dst_path = Path(dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"❌ Copy failed {src} -> {dst}: {str(e)}")
            return False
    
    def batch_copy_files(self, file_pairs: List[Tuple[Path, Path]], 
                        progress_callback: Optional[callable] = None) -> Dict[str, int]:
        """📦 Batch copy dengan threading"""
        stats = {'success': 0, 'failed': 0}
        
        def copy_single(pair_with_index):
            i, (src, dst) = pair_with_index
            success = self.copy_file(src, dst)
            
            if progress_callback and i % max(1, len(file_pairs) // 20) == 0:
                progress_callback('current', i + 1, len(file_pairs), f"Copying {src.name}")
            
            return success
        
        if self.use_threading and len(file_pairs) > 10:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                indexed_pairs = list(enumerate(file_pairs))
                results = list(executor.map(copy_single, indexed_pairs))
                stats['success'] = sum(results)
                stats['failed'] = len(results) - stats['success']
        else:
            for i, pair in enumerate(file_pairs):
                success = copy_single((i, pair))
                stats['success' if success else 'failed'] += 1
        
        return stats
    
    def read_yolo_label(self, label_path: Union[str, Path]) -> List[List[float]]:
        """📋 Read YOLO label file"""
        try:
            path = Path(label_path)
            if not path.exists():
                return []
            
            bboxes = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split()
                            if len(parts) >= 5:
                                bbox = [float(parts[0])] + [float(x) for x in parts[1:5]]
                                bboxes.append(bbox)
                        except ValueError:
                            continue
            
            return bboxes
            
        except Exception as e:
            self.logger.error(f"❌ Error reading label {label_path}: {str(e)}")
            return []
    
    def write_yolo_label(self, label_path: Union[str, Path], bboxes: List[List[float]]) -> bool:
        """💾 Write YOLO label file"""
        try:
            path = Path(label_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                for bbox in bboxes:
                    if len(bbox) >= 5:
                        line = f"{int(bbox[0])} {' '.join(f'{x:.6f}' for x in bbox[1:5])}\n"
                        f.write(line)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error writing label {label_path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """📊 Get file information"""
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
                'name': path.name
            }
            
            # Additional info untuk images
            if path.suffix.lower() in self.supported_image_formats:
                try:
                    image = cv2.imread(str(path))
                    if image is not None:
                        h, w = image.shape[:2]
                        info.update({
                            'width': w,
                            'height': h,
                            'channels': 3 if len(image.shape) == 3 else 1,
                            'aspect_ratio': round(w / h, 2) if h > 0 else 0
                        })
                except Exception:
                    pass
            
            return info
            
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    def cleanup_directory(self, directory: Union[str, Path], pattern: str = None) -> int:
        """🧹 Cleanup files dengan pattern"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return 0
            
            files_removed = 0
            if pattern:
                for file_path in dir_path.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        files_removed += 1
            else:
                shutil.rmtree(dir_path)
                files_removed = 1
            
            return files_removed
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {str(e)}")
            return 0