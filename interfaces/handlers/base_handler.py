# File: src/interfaces/handlers/base_handler.py
# Author: Alfrida Sabar
# Deskripsi: Base handler untuk data management interface

from pathlib import Path
from typing import Dict, Optional
from termcolor import colored
from interfaces.base_interface import BaseInterface
from utils.logging import ColoredLogger

class BaseHandler:
    """Base handler for data management operations"""
    def __init__(self, config):
        self.cfg = config
        self.logger = ColoredLogger('DataManager')
        
        # Define directory structure
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.rupiah_dir = self.data_dir / "rupiah"
        
        # States
        self.current_operation = None
        self.operation_stats = {}
        
    def get_split_dirs(self):
        """Get training split directories"""
        splits = ['train', 'val', 'test']
        return [(self.rupiah_dir / split / 'images', 
                self.rupiah_dir / split / 'labels') 
                for split in splits]
                
    def validate_directory(self, path: Path) -> bool:
        """Validate if directory exists and is accessible"""
        try:
            if not path.exists():
                path.mkdir(parents=True)
            return True
        except Exception as e:
            self.logger.error(f"Gagal mengakses direktori {path}: {str(e)}")
            return False
            
    def update_stats(self, operation: str, stats: Dict):
        """Update operation statistics"""
        self.operation_stats[operation] = stats
        
    def get_stats(self, operation: str) -> Optional[Dict]:
        """Get statistics for specific operation"""
        return self.operation_stats.get(operation)
        
    def clear_stats(self):
        """Clear all statistics"""
        self.operation_stats = {}
        
    def log_operation(self, operation: str, status: str, detail: str = None):
        """Log operation with colored status"""
        status_color = 'green' if status == 'success' else 'red'
        self.logger.info(
            f"{operation}: {colored(status, status_color)}"
            + (f" - {detail}" if detail else "")
        )