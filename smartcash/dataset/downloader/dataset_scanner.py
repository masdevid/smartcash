"""
File: smartcash/dataset/downloader/dataset_scanner.py
Deskripsi: Dataset scanner dengan content validation dan optimal workers
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.dataset.downloader.base import BaseDownloaderComponent
from smartcash.common.environment import get_environment_manager

class DatasetScanner(BaseDownloaderComponent):
    """Dataset scanner dengan deep content validation dan parallel processing"""
    
    def __init__(self, logger=None, max_workers: int = None):
        super().__init__(logger)
        self.env_manager = get_environment_manager()
        self.dataset_path = self.env_manager.get_dataset_path()
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        if max_workers is None:
            from smartcash.common.threadpools import get_optimal_thread_count
            max_workers = get_optimal_thread_count('io')
        
        self.max_workers = max_workers
        self.logger.info(f"🔍 DatasetScanner initialized with {self.max_workers} workers")
    
    def quick_check_existing(self) -> bool:
        """Quick check dengan actual content validation"""
        try:
            if not self.dataset_path.exists():
                return False
            
            # Check actual image files, not just directories
            total_images = 0
            for split in ['train', 'valid', 'test']:
                split_images_dir = self.dataset_path / split / 'images'
                if split_images_dir.exists():
                    images = [f for f in split_images_dir.glob('*.*') 
                             if f.suffix.lower() in self.img_extensions]
                    total_images += len(images)
            
            # Also check downloads for content
            downloads_dir = self.dataset_path / 'downloads'
            if downloads_dir.exists():
                download_files = list(downloads_dir.glob('*.*'))
                if download_files:
                    total_images += len(download_files)
            
            return total_images > 0
            
        except Exception:
            return False
    
    def scan_existing_dataset_parallel(self) -> Dict[str, Any]:
        """Comprehensive scan dengan parallel content validation"""
        self._notify_progress("scan_start", 0, 100, f"🔍 Starting deep scan with {self.max_workers} workers...")
        
        try:
            if not self.dataset_path.exists():
                return self._create_empty_scan_result("Dataset directory tidak ditemukan")
            
            self._notify_progress("scan_structure", 20, 100, "🔍 Analyzing dataset structure...")
            structure = self._analyze_dataset_structure()
            
            self._notify_progress("scan_downloads", 40, 100, "📂 Scanning downloads content...")
            downloads_result = self._scan_downloads_content()
            
            self._notify_progress("scan_splits", 60, 100, "📊 Scanning splits content...")
            splits_result = self._scan_splits_content_parallel()
            
            self._notify_progress("scan_aggregate", 80, 100, "📈 Aggregating statistics...")
            
            result = {
                'status': 'success',
                'dataset_path': str(self.dataset_path),
                'structure': structure,
                'downloads': downloads_result,
                'splits': splits_result,
                'summary': self._create_summary(downloads_result, splits_result),
                'scan_time': self._get_scan_timestamp(),
                'workers_used': self.max_workers
            }
            
            total_content = result['summary']['total_images'] + result['summary']['download_files']
            self._notify_progress("scan_complete", 100, 100, f"✅ Deep scan complete: {total_content} files found")
            return result
            
        except Exception as e:
            error_msg = f"Error during deep scanning: {str(e)}"
            self._notify_progress("scan_error", 0, 100, f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def _scan_downloads_content(self) -> Dict[str, Any]:
        """Scan downloads dengan content validation"""
        downloads_dir = self.dataset_path / 'downloads'
        
        if not downloads_dir.exists():
            return {'status': 'not_found', 'file_count': 0, 'total_size': 0}
        
        try:
            all_files = [f for f in downloads_dir.rglob('*') if f.is_file()]
            
            if not all_files:
                return {'status': 'empty', 'file_count': 0, 'total_size': 0}
            
            # Parallel processing untuk large directories
            file_chunks = [all_files[i:i + 100] for i in range(0, len(all_files), 100)]
            
            file_stats = {}
            total_files = 0
            total_size = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._count_file_chunk, chunk) for chunk in file_chunks]
                
                for future in futures:
                    try:
                        chunk_stats, chunk_files, chunk_size = future.result()
                        
                        for ext, stats in chunk_stats.items():
                            if ext not in file_stats:
                                file_stats[ext] = {'count': 0, 'size': 0}
                            file_stats[ext]['count'] += stats['count']
                            file_stats[ext]['size'] += stats['size']
                        
                        total_files += chunk_files
                        total_size += chunk_size
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error processing chunk: {str(e)}")
            
            return {
                'status': 'success',
                'file_count': total_files,
                'total_size': total_size,
                'size_formatted': self._format_file_size(total_size),
                'file_types': file_stats,
                'image_files': file_stats.get('.jpg', {}).get('count', 0) + file_stats.get('.png', {}).get('count', 0)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'file_count': 0, 'total_size': 0}
    
    def _scan_splits_content_parallel(self) -> Dict[str, Any]:
        """Parallel scan splits dengan deep content validation"""
        splits = ['train', 'valid', 'test']
        split_results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(splits))) as executor:
            future_to_split = {
                executor.submit(self._scan_single_split_content, split): split 
                for split in splits
            }
            
            for future in future_to_split:
                split_name = future_to_split[future]
                try:
                    split_results[split_name] = future.result()
                except Exception as e:
                    split_results[split_name] = {
                        'status': 'error', 'message': str(e),
                        'images': 0, 'labels': 0
                    }
        
        return split_results
    
    def _scan_single_split_content(self, split_name: str) -> Dict[str, Any]:
        """Deep scan single split dengan actual file validation"""
        split_dir = self.dataset_path / split_name
        
        if not split_dir.exists():
            return {'status': 'not_found', 'images': 0, 'labels': 0}
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Actual file counting dengan validation
        image_count = 0
        label_count = 0
        image_size = 0
        label_size = 0
        
        # Count valid images
        if images_dir.exists():
            image_files = [f for f in images_dir.glob('*.*') if f.suffix.lower() in self.img_extensions]
            image_count = len(image_files)
            
            # Calculate size untuk sample files (avoid processing too many)
            sample_files = image_files[:min(100, len(image_files))]
            for img_file in sample_files:
                try:
                    image_size += img_file.stat().st_size
                except Exception:
                    continue
            
            # Estimate total size
            if sample_files:
                avg_size = image_size / len(sample_files)
                image_size = int(avg_size * image_count)
        
        # Count valid labels
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            label_count = len(label_files)
            
            # Calculate label sizes
            for label_file in label_files[:min(100, len(label_files))]:
                try:
                    label_size += label_file.stat().st_size
                except Exception:
                    continue
            
            # Estimate total size
            if label_files and label_count > 100:
                avg_size = label_size / min(100, len(label_files))
                label_size = int(avg_size * label_count)
        
        # Validate image-label pairing
        paired_count = 0
        if image_count > 0 and label_count > 0:
            # Sample check untuk pairing validation
            sample_images = list(images_dir.glob('*.*'))[:min(50, image_count)]
            for img_file in sample_images:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    paired_count += 1
        
        return {
            'status': 'success',
            'images': image_count,
            'labels': label_count,
            'image_size': image_size,
            'label_size': label_size,
            'total_size': image_size + label_size,
            'size_formatted': self._format_file_size(image_size + label_size),
            'images_dir_exists': images_dir.exists(),
            'labels_dir_exists': labels_dir.exists(),
            'paired_files': paired_count,
            'pairing_ratio': paired_count / min(image_count, 50) if image_count > 0 else 0
        }
    
    def _count_file_chunk(self, file_chunk: List[Path]) -> Tuple[Dict, int, int]:
        """Count files dalam chunk dengan extension categorization"""
        chunk_stats = {}
        chunk_files = 0
        chunk_size = 0
        
        for file_path in file_chunk:
            if file_path.is_file():
                try:
                    ext = file_path.suffix.lower()
                    size = file_path.stat().st_size
                    
                    if ext not in chunk_stats:
                        chunk_stats[ext] = {'count': 0, 'size': 0}
                    
                    chunk_stats[ext]['count'] += 1
                    chunk_stats[ext]['size'] += size
                    chunk_files += 1
                    chunk_size += size
                    
                except Exception:
                    continue
        
        return chunk_stats, chunk_files, chunk_size
    
    def get_cleanup_targets(self) -> Dict[str, Any]:
        """Get cleanup targets dengan actual content counting"""
        try:
            cleanup_targets = {
                'downloads': self.dataset_path / 'downloads',
                'train_images': self.dataset_path / 'train' / 'images',
                'train_labels': self.dataset_path / 'train' / 'labels',
                'valid_images': self.dataset_path / 'valid' / 'images',
                'valid_labels': self.dataset_path / 'valid' / 'labels',
                'test_images': self.dataset_path / 'test' / 'images',
                'test_labels': self.dataset_path / 'test' / 'labels'
            }
            
            existing_targets = {}
            total_files = 0
            total_size = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_target = {
                    executor.submit(self._count_files_recursive, target_path): target_name
                    for target_name, target_path in cleanup_targets.items()
                    if target_path.exists()
                }
                
                for future in future_to_target:
                    target_name = future_to_target[future]
                    try:
                        file_count, size_bytes = future.result()
                        
                        if file_count > 0:
                            existing_targets[target_name] = {
                                'path': str(cleanup_targets[target_name]),
                                'file_count': file_count,
                                'size_bytes': size_bytes,
                                'size_formatted': self._format_file_size(size_bytes)
                            }
                            total_files += file_count
                            total_size += size_bytes
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error counting {target_name}: {str(e)}")
            
            return {
                'status': 'success',
                'targets': existing_targets,
                'summary': {
                    'total_files': total_files,
                    'total_size': total_size,
                    'size_formatted': self._format_file_size(total_size)
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Error getting cleanup targets: {str(e)}")
    
    def _count_files_recursive(self, directory: Path) -> Tuple[int, int]:
        """Count files recursively dengan size calculation"""
        if not directory.exists():
            return 0, 0
            
        file_count = 0
        total_size = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
        except Exception:
            pass
            
        return file_count, total_size
    
    def _analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze dataset structure dengan content awareness"""
        structure = {
            'has_downloads': (self.dataset_path / 'downloads').exists(),
            'has_splits': False,
            'split_dirs': [],
            'additional_files': [],
            'total_directories': 0,
            'empty_directories': []
        }
        
        try:
            # Check split directories dengan content validation
            split_candidates = ['train', 'valid', 'test', 'val']
            for split in split_candidates:
                split_dir = self.dataset_path / split
                if split_dir.exists() and split_dir.is_dir():
                    # Check if split has actual content
                    images_dir = split_dir / 'images'
                    if images_dir.exists():
                        images = [f for f in images_dir.glob('*.*') if f.suffix.lower() in self.img_extensions]
                        if images:  # Only count splits with actual images
                            structure['split_dirs'].append(split)
                        else:
                            structure['empty_directories'].append(f"{split}/images")
                    else:
                        structure['empty_directories'].append(f"{split}/images (missing)")
            
            structure['has_splits'] = len(structure['split_dirs']) > 0
            
            # Count total directories
            structure['total_directories'] = len([d for d in self.dataset_path.rglob('*') if d.is_dir()])
            
            # Check additional files
            for item in self.dataset_path.iterdir():
                if item.is_file() and item.name.endswith(('.yaml', '.yml', '.txt', '.md')):
                    structure['additional_files'].append(item.name)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing structure: {str(e)}")
        
        return structure
    
    def _create_summary(self, downloads_result: Dict, splits_result: Dict) -> Dict[str, Any]:
        """Create comprehensive summary dengan content metrics"""
        total_images = sum(split.get('images', 0) for split in splits_result.values())
        total_labels = sum(split.get('labels', 0) for split in splits_result.values())
        download_files = downloads_result.get('file_count', 0)
        download_images = downloads_result.get('image_files', 0)
        
        # Calculate pairing quality
        valid_splits = [s for s in splits_result.values() if s.get('images', 0) > 0]
        avg_pairing = sum(s.get('pairing_ratio', 0) for s in valid_splits) / len(valid_splits) if valid_splits else 0
        
        return {
            'total_images': total_images,
            'total_labels': total_labels,
            'download_files': download_files,
            'download_images': download_images,
            'valid_splits': len(valid_splits),
            'dataset_complete': total_images > 0 and total_labels > 0,
            'average_pairing_quality': avg_pairing,
            'has_content': total_images > 0 or download_files > 0
        }
    
    def _create_empty_scan_result(self, message: str) -> Dict[str, Any]:
        """Create empty scan result"""
        return {
            'status': 'empty',
            'message': message,
            'dataset_path': str(self.dataset_path),
            'summary': {'total_images': 0, 'total_labels': 0, 'download_files': 0, 'has_content': False}
        }
    
    def _get_scan_timestamp(self) -> str:
        """Get scan timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size dari bytes"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def create_dataset_scanner(logger=None, max_workers: int = None) -> DatasetScanner:
    """Factory dengan optimal workers support"""
    return DatasetScanner(logger, max_workers)