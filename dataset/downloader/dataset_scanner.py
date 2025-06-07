"""
File: smartcash/dataset/downloader/dataset_scanner.py
Deskripsi: Backend service untuk scanning direktori dataset dan counting files
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.dataset.downloader.base import BaseDownloaderComponent
from smartcash.common.environment import get_environment_manager

class DatasetScanner(BaseDownloaderComponent):
    """Backend service untuk scanning dan counting dataset files"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.env_manager = get_environment_manager()
        self.dataset_path = self.env_manager.get_dataset_path()
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def scan_existing_dataset(self) -> Dict[str, Any]:
        """Scan existing dataset dengan comprehensive analysis"""
        self._notify_progress("scan_start", 0, 100, "🔍 Memulai scanning dataset...")
        
        try:
            # Check dataset path existence
            if not self.dataset_path.exists():
                self._notify_progress("scan_warning", 0, 100, f"⚠️ Dataset path tidak ditemukan: {self.dataset_path}")
                return self._create_empty_scan_result("Dataset directory tidak ditemukan")
            
            self._notify_progress("scan_structure", 20, 100, "🔍 Analyzing dataset structure...")
            
            # Scan structure
            structure = self._analyze_dataset_structure()
            
            self._notify_progress("scan_downloads", 40, 100, "📂 Scanning downloads directory...")
            downloads_result = self._scan_downloads_directory()
            
            self._notify_progress("scan_splits", 60, 100, "📊 Scanning splits directories...")
            splits_result = self._scan_splits_directories()
            
            self._notify_progress("scan_aggregate", 80, 100, "📈 Aggregating statistics...")
            
            # Aggregate results
            result = {
                'status': 'success',
                'dataset_path': str(self.dataset_path),
                'structure': structure,
                'downloads': downloads_result,
                'splits': splits_result,
                'summary': self._create_summary(downloads_result, splits_result),
                'scan_time': self._get_scan_timestamp()
            }
            
            self._notify_progress("scan_complete", 100, 100, f"✅ Scan selesai: {result['summary']['total_images']} gambar")
            return result
            
        except Exception as e:
            error_msg = f"Error saat scanning dataset: {str(e)}"
            self._notify_progress("scan_error", 0, 100, f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def quick_check_existing(self) -> bool:
        """Quick check apakah dataset sudah ada"""
        try:
            if not self.dataset_path.exists():
                return False
                
            # Check jika ada content minimal
            content_indicators = [
                self.dataset_path / 'downloads',
                self.dataset_path / 'train',
                self.dataset_path / 'valid',
                self.dataset_path / 'test'
            ]
            
            return any(path.exists() and any(path.iterdir()) for path in content_indicators)
            
        except Exception:
            return False
    
    def get_cleanup_targets(self) -> Dict[str, Any]:
        """Get direktori dan files yang akan di-cleanup"""
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
            
            # Scan targets yang ada
            existing_targets = {}
            total_files = 0
            total_size = 0
            
            for target_name, target_path in cleanup_targets.items():
                if target_path.exists():
                    file_count, size_bytes = self._count_files_recursive(target_path)
                    existing_targets[target_name] = {
                        'path': str(target_path),
                        'file_count': file_count,
                        'size_bytes': size_bytes,
                        'size_formatted': self._format_file_size(size_bytes)
                    }
                    total_files += file_count
                    total_size += size_bytes
            
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
    
    def _analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze overall dataset structure"""
        structure = {
            'has_downloads': (self.dataset_path / 'downloads').exists(),
            'has_splits': False,
            'split_dirs': [],
            'additional_files': []
        }
        
        # Check for split directories
        split_candidates = ['train', 'valid', 'test', 'val']
        for split in split_candidates:
            split_dir = self.dataset_path / split
            if split_dir.exists() and split_dir.is_dir():
                structure['split_dirs'].append(split)
        
        structure['has_splits'] = len(structure['split_dirs']) > 0
        
        # Check for additional files
        for item in self.dataset_path.iterdir():
            if item.is_file() and item.name.endswith(('.yaml', '.yml', '.txt', '.md')):
                structure['additional_files'].append(item.name)
        
        return structure
    
    def _scan_downloads_directory(self) -> Dict[str, Any]:
        """Scan downloads directory"""
        downloads_dir = self.dataset_path / 'downloads'
        
        if not downloads_dir.exists():
            return {'status': 'not_found', 'file_count': 0, 'total_size': 0}
        
        # Count files with extensions
        file_stats = {}
        total_files = 0
        total_size = 0
        
        try:
            for file_path in downloads_dir.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    size = file_path.stat().st_size
                    
                    if ext not in file_stats:
                        file_stats[ext] = {'count': 0, 'size': 0}
                    
                    file_stats[ext]['count'] += 1
                    file_stats[ext]['size'] += size
                    total_files += 1
                    total_size += size
            
            return {
                'status': 'success',
                'file_count': total_files,
                'total_size': total_size,
                'size_formatted': self._format_file_size(total_size),
                'file_types': file_stats
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'file_count': 0, 'total_size': 0}
    
    def _scan_splits_directories(self) -> Dict[str, Any]:
        """Scan all splits directories dengan parallel processing"""
        splits = ['train', 'valid', 'test']
        split_results = {}
        
        # Use ThreadPoolExecutor untuk parallel scanning
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_split = {
                executor.submit(self._scan_single_split, split): split 
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
    
    def _scan_single_split(self, split_name: str) -> Dict[str, Any]:
        """Scan single split directory"""
        split_dir = self.dataset_path / split_name
        
        if not split_dir.exists():
            return {'status': 'not_found', 'images': 0, 'labels': 0}
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Count images dan labels
        image_count = self._count_images(images_dir) if images_dir.exists() else 0
        label_count = self._count_labels(labels_dir) if labels_dir.exists() else 0
        
        # Get size info
        image_size = self._get_directory_size(images_dir) if images_dir.exists() else 0
        label_size = self._get_directory_size(labels_dir) if labels_dir.exists() else 0
        
        return {
            'status': 'success',
            'images': image_count,
            'labels': label_count,
            'image_size': image_size,
            'label_size': label_size,
            'total_size': image_size + label_size,
            'size_formatted': self._format_file_size(image_size + label_size),
            'images_dir_exists': images_dir.exists(),
            'labels_dir_exists': labels_dir.exists()
        }
    
    def _count_images(self, directory: Path) -> int:
        """Count image files dalam directory"""
        if not directory.exists():
            return 0
        return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in self.img_extensions)
    
    def _count_labels(self, directory: Path) -> int:
        """Count label files dalam directory"""
        if not directory.exists():
            return 0
        return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.txt')
    
    def _count_files_recursive(self, directory: Path) -> Tuple[int, int]:
        """Count files recursively dalam directory"""
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
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size dari directory"""
        if not directory.exists():
            return 0
        return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    
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
    
    def _create_summary(self, downloads_result: Dict, splits_result: Dict) -> Dict[str, Any]:
        """Create aggregate summary dari scan results"""
        total_images = sum(split.get('images', 0) for split in splits_result.values())
        total_labels = sum(split.get('labels', 0) for split in splits_result.values())
        download_files = downloads_result.get('file_count', 0)
        
        return {
            'total_images': total_images,
            'total_labels': total_labels,
            'download_files': download_files,
            'valid_splits': len([s for s in splits_result.values() if s.get('images', 0) > 0]),
            'dataset_complete': total_images > 0 and total_labels > 0
        }
    
    def _create_empty_scan_result(self, message: str) -> Dict[str, Any]:
        """Create empty scan result"""
        return {
            'status': 'empty',
            'message': message,
            'dataset_path': str(self.dataset_path),
            'summary': {'total_images': 0, 'total_labels': 0, 'download_files': 0}
        }
    
    def _get_scan_timestamp(self) -> str:
        """Get scan timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_dataset_scanner(logger=None) -> DatasetScanner:
    """Factory untuk DatasetScanner"""
    return DatasetScanner(logger)