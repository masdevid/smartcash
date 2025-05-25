"""
File: smartcash/dataset/preprocessor/storage/preprocessed_storage_manager.py
Deskripsi: Enhanced storage manager untuk hasil preprocessing dengan metadata management
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger


class PreprocessedStorageManager:
    """Enhanced storage manager untuk preprocessing results dengan comprehensive metadata."""
    
    def __init__(self, output_dir: str, logger=None):
        """Initialize storage manager dengan output directory."""
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger()
        self.metadata_dir = self.output_dir / 'metadata'
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def get_split_path(self, split: str) -> Path:
        """Get path untuk split directory."""
        return self.output_dir / split
    
    def save_processing_metadata(self, split: str, metadata: Dict[str, Any]) -> bool:
        """Save processing metadata untuk split."""
        try:
            metadata_file = self.metadata_dir / f"{split}_processing.json"
            serializable_data = self._make_json_serializable(metadata)
            
            with open(metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving metadata: {str(e)}")
            return False
    
    def load_processing_metadata(self, split: str) -> Dict[str, Any]:
        """Load processing metadata untuk split."""
        try:
            metadata_file = self.metadata_dir / f"{split}_processing.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Error loading metadata: {str(e)}")
        
        return {}
    
    def update_split_stats(self, split: str, stats: Dict[str, Any]) -> None:
        """Update statistics untuk split."""
        try:
            stats_file = self.metadata_dir / f"{split}_stats.json"
            serializable_stats = self._make_json_serializable(stats)
            
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Error saving stats: {str(e)}")
    
    def get_split_stats(self, split: str) -> Dict[str, Any]:
        """Get statistics untuk split."""
        try:
            stats_file = self.metadata_dir / f"{split}_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Error loading stats: {str(e)}")
        
        return {}
    
    def cleanup_split_storage(self, split: Optional[str] = None) -> Dict[str, Any]:
        """Cleanup storage untuk split atau semua splits."""
        cleanup_stats = {'splits_cleaned': 0, 'files_removed': 0, 'bytes_freed': 0}
        
        try:
            if split:
                # Clean specific split
                split_path = self.get_split_path(split)
                if split_path.exists():
                    size_before = self._calculate_directory_size(split_path)
                    shutil.rmtree(split_path)
                    cleanup_stats['splits_cleaned'] = 1
                    cleanup_stats['bytes_freed'] = size_before
                
                # Clean metadata untuk split
                self._cleanup_split_metadata(split)
            else:
                # Clean all splits
                for item in self.output_dir.iterdir():
                    if item.is_dir() and item.name != 'metadata':
                        size_before = self._calculate_directory_size(item)
                        shutil.rmtree(item)
                        cleanup_stats['splits_cleaned'] += 1
                        cleanup_stats['bytes_freed'] += size_before
                
                # Clean all metadata
                if self.metadata_dir.exists():
                    shutil.rmtree(self.metadata_dir)
                    self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {str(e)}")
        
        return cleanup_stats
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get storage summary dengan comprehensive info."""
        summary = {'splits': {}, 'total_size_mb': 0, 'total_files': 0}
        
        try:
            for split in ['train', 'valid', 'test']:
                split_path = self.get_split_path(split)
                if split_path.exists():
                    split_info = self._analyze_split_storage(split_path)
                    summary['splits'][split] = split_info
                    summary['total_size_mb'] += split_info['size_mb']
                    summary['total_files'] += split_info['file_count']
        except Exception as e:
            self.logger.error(f"❌ Storage summary error: {str(e)}")
        
        return summary
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data ke JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate size directory dalam bytes."""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _analyze_split_storage(self, split_path: Path) -> Dict[str, Any]:
        """Analyze storage untuk single split."""
        info = {'exists': split_path.exists(), 'file_count': 0, 'size_bytes': 0, 'size_mb': 0}
        
        if info['exists']:
            for file_path in split_path.rglob('*'):
                if file_path.is_file():
                    info['file_count'] += 1
                    info['size_bytes'] += file_path.stat().st_size
            
            info['size_mb'] = round(info['size_bytes'] / (1024 * 1024), 2)
        
        return info
    
    def _cleanup_split_metadata(self, split: str) -> None:
        """Cleanup metadata untuk specific split."""
        metadata_patterns = [f"{split}_*.json"]
        
        for pattern in metadata_patterns:
            for metadata_file in self.metadata_dir.glob(pattern):
                try:
                    metadata_file.unlink()
                except Exception:
                    pass