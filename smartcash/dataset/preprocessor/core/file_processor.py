"""
File: smartcash/dataset/preprocessor/core/file_processor.py
Deskripsi: Integrated file processor menggunakan FileNamingManager untuk DRY pattern handling
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.worker_utils import get_optimal_worker_count
from smartcash.common.utils.file_naming_manager import create_file_naming_manager, is_research_format

class FileProcessor:
    """📁 Integrated file processor menggunakan FileNamingManager untuk DRY operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager(config)
        
        # File handling settings
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.max_workers = self.config.get('max_workers', get_optimal_worker_count('io'))
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
        """🔍 Scan files menggunakan FileNamingManager patterns"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return []
            
            extensions = extensions or self.supported_image_formats
            
            files = []
            for ext in extensions:
                if pattern:
                    # Support patterns yang compatible dengan FileNamingManager
                    files.extend(dir_path.glob(f'{pattern}*{ext}'))
                else:
                    files.extend(dir_path.glob(f'*{ext}'))
            
            # Filter menggunakan naming manager untuk research format
            if pattern == 'rp_':
                research_files = [f for f in files if is_research_format(f.name)]
                return sorted(research_files)
            
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {directory}: {str(e)}")
            return []
    
    def scan_preprocessed_files(self, directory: Union[str, Path]) -> Dict[str, List[Path]]:
        """🔍 Scan semua preprocessed files menggunakan naming manager patterns"""
        dir_path = Path(directory)
        results = {}
        
        for file_type in self.naming_manager.get_supported_types():
            prefix = self.naming_manager.get_prefix(file_type)
            if prefix:
                if file_type in ['preprocessed', 'augmented']:
                    results[file_type] = self.scan_files(dir_path, prefix, {'.npy'})
                else:
                    results[file_type] = self.scan_files(dir_path, prefix)
        
        return results
    
    def scan_files_by_type(self, directory: Union[str, Path], file_type: str, 
                          extensions: set = None) -> List[Path]:
        """🔍 Scan files untuk specific type menggunakan naming manager"""
        prefix = self.naming_manager.get_prefix(file_type)
        if not prefix:
            return []
        
        default_extensions = {
            'raw': self.supported_image_formats,
            'preprocessed': {'.npy'},
            'augmented': {'.npy', '.jpg'},
            'sample': {'.jpg'},
            'augmented_sample': {'.jpg'}
        }
        
        extensions = extensions or default_extensions.get(file_type, self.supported_image_formats)
        return self.scan_files(directory, prefix, extensions)
    
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
            
            # Reduce progress callback frequency (every 10% instead of 5%)
            if progress_callback and i % max(1, len(file_pairs) // 10) == 0:
                batch_num = (i // max(1, len(file_pairs) // 10)) + 1
                progress_callback('current', i + 1, len(file_pairs), f"Copying batch {batch_num}/10...")
            
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
        """📊 Get file information dengan naming manager integration"""
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
            
            # Add naming info menggunakan FileNamingManager
            validation = self.naming_manager.validate_filename_format(path.name)
            if validation['valid']:
                info.update({
                    'research_format': True,
                    'nominal': validation['parsed']['nominal'],
                    'description': validation['parsed']['description']
                })
            
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
    
    def generate_preprocessed_filename(self, original_path: Union[str, Path]) -> str:
        """🔧 Generate preprocessed filename menggunakan FileNamingManager"""
        from ..utils.metadata_extractor import MetadataExtractor
        
        path = Path(original_path)
        extractor = MetadataExtractor()
        
        # Parse original filename menggunakan integrated extractor
        parsed = extractor.parse_research_filename(path.name)
        if parsed and parsed['type'] == 'raw':
            return f"pre_{parsed['nominal']}_{parsed['uuid']}.npy"
        
        # Generate new filename menggunakan naming manager
        primary_class = self._extract_primary_class(path)
        file_info = self.naming_manager.generate_file_info(
            path.name, primary_class, 'preprocessed'
        )
        return f"pre_{file_info.nominal}_{file_info.uuid}.npy"
    
    def generate_sample_filename(self, preprocessed_path: Union[str, Path]) -> str:
        """🎲 Generate sample filename menggunakan FileNamingManager"""
        from ..utils.metadata_extractor import MetadataExtractor
        
        path = Path(preprocessed_path)
        extractor = MetadataExtractor()
        
        # Parse preprocessed filename
        parsed = extractor.parse_research_filename(path.name)
        if parsed and parsed['type'] == 'preprocessed':
            return f"sample_pre_{parsed['nominal']}_{parsed['uuid']}.jpg"
        
        # Generate new filename menggunakan naming manager
        file_info = self.naming_manager.generate_file_info(
            path.name, None, 'sample'
        )
        return f"sample_pre_{file_info.nominal}_{file_info.uuid}.jpg"
    
    def find_corresponding_label(self, image_path: Union[str, Path]) -> Optional[Path]:
        """🔍 Find corresponding label menggunakan naming manager consistency"""
        from ..utils.metadata_extractor import MetadataExtractor
        
        img_path = Path(image_path)
        extractor = MetadataExtractor()
        
        # Parse image filename menggunakan integrated extractor
        parsed = extractor.parse_research_filename(img_path.name)
        if parsed:
            # Generate corresponding label filename dengan same pattern
            if parsed['type'] == 'raw':
                label_name = f"rp_{parsed['nominal']}_{parsed['uuid']}.txt"
            elif parsed['type'] == 'preprocessed':
                label_name = f"pre_{parsed['nominal']}_{parsed['uuid']}.txt"
            elif parsed['type'] == 'sample':
                label_name = f"sample_pre_{parsed['nominal']}_{parsed['uuid']}.txt"
            else:
                label_name = f"{img_path.stem}.txt"
        else:
            label_name = f"{img_path.stem}.txt"
        
        # Check in same directory first
        same_dir_label = img_path.parent / label_name
        if same_dir_label.exists():
            return same_dir_label
        
        # Check in labels directory structure
        if img_path.parent.name == 'images':
            labels_dir = img_path.parent.parent / 'labels'
            labels_dir_file = labels_dir / label_name
            if labels_dir_file.exists():
                return labels_dir_file
        
        return None
    
    def validate_image_label_pair(self, image_path: Union[str, Path], 
                                 label_path: Union[str, Path] = None) -> Dict[str, Any]:
        """✅ Validate image-label pair menggunakan naming manager consistency"""
        from ..utils.metadata_extractor import MetadataExtractor
        
        img_path = Path(image_path)
        
        # Auto-find label jika tidak provided
        if label_path is None:
            label_path = self.find_corresponding_label(img_path)
        
        if label_path is None:
            return {
                'valid': False,
                'reason': 'Label file not found',
                'image_exists': img_path.exists(),
                'label_exists': False
            }
        
        label_path = Path(label_path)
        
        # Check file existence
        if not img_path.exists():
            return {
                'valid': False,
                'reason': 'Image file not found',
                'image_exists': False,
                'label_exists': label_path.exists()
            }
        
        if not label_path.exists():
            return {
                'valid': False,
                'reason': 'Label file not found',
                'image_exists': True,
                'label_exists': False
            }
        
        # Validate filename consistency menggunakan integrated extractor
        extractor = MetadataExtractor()
        consistency = extractor.validate_filename_consistency(img_path, label_path)
        
        if not consistency['consistent']:
            return {
                'valid': False,
                'reason': f"Filename inconsistency: {consistency['reason']}",
                'image_exists': True,
                'label_exists': True,
                'consistency': consistency
            }
        
        # Validate label content
        bboxes = self.read_yolo_label(label_path)
        
        return {
            'valid': True,
            'image_exists': True,
            'label_exists': True,
            'consistency': consistency,
            'bbox_count': len(bboxes),
            'has_annotations': len(bboxes) > 0,
            'nominal_info': {
                'nominal': consistency['nominal'],
                'description': consistency.get('description', 'Unknown')
            }
        }
    
    def _extract_primary_class(self, file_path: Path) -> Optional[str]:
        """💰 Extract primary class menggunakan naming manager"""
        # Try to find corresponding label
        label_path = self.find_corresponding_label(file_path)
        if label_path:
            return self.naming_manager.extract_primary_class_from_label(label_path)
        return None
    
    def get_research_format_stats(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """📊 Get statistics untuk research format files dalam directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return {'error': f'Directory not found: {directory}'}
        
        # Scan all image files
        all_files = []
        for ext in self.supported_image_formats:
            all_files.extend(dir_path.glob(f'*{ext}'))
        
        if not all_files:
            return {'total_files': 0, 'research_format': 0, 'legacy_format': 0}
        
        # Classify menggunakan naming manager
        research_count = 0
        legacy_count = 0
        nominal_distribution = {}
        
        for file_path in all_files:
            validation = self.naming_manager.validate_filename_format(file_path.name)
            if validation['valid']:
                research_count += 1
                nominal = validation['parsed']['nominal']
                nominal_distribution[nominal] = nominal_distribution.get(nominal, 0) + 1
            else:
                legacy_count += 1
        
        return {
            'total_files': len(all_files),
            'research_format': research_count,
            'legacy_format': legacy_count,
            'research_percentage': round((research_count / len(all_files)) * 100, 1),
            'nominal_distribution': nominal_distribution,
            'naming_manager_stats': self.naming_manager.get_nominal_statistics()
        }