"""
File: smartcash/ui/dataset/downloader/services/backends/dataset_scanner.py
Description: Service for scanning and analyzing existing datasets.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from smartcash.ui.dataset.downloader.services.core.base_service import BaseService

class DatasetScannerService(BaseService):
    """Service for scanning and analyzing existing datasets."""
    
    def get_existing_dataset_count(self) -> int:
        """Get count of existing dataset files.
        
        Returns:
            int: Number of existing dataset files
        """
        try:
            has_content, total_images, _ = self.check_existing_dataset()
            return total_images if has_content else 0
        except Exception as e:
            self.log_error(f"Error getting dataset count: {e}")
            return 0
    
    def check_existing_dataset(self) -> Tuple[bool, int, Dict[str, Any]]:
        """Check existing dataset using backend scanner.
        
        Returns:
            Tuple[bool, int, Dict]: (has_content, total_images, summary_data)
        """
        try:
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            
            scanner = create_dataset_scanner(self.logger)
            
            # Quick check to determine if content exists
            if not scanner.quick_check_existing():
                return False, 0, {}
            
            # Get detailed summary if content exists
            result = scanner.scan_existing_dataset_parallel()
            
            if result.get('status') == 'success':
                summary = result.get('summary', {})
                total_images = sum(split.get('images', 0) for split in result.get('splits', {}).values())
                return True, total_images, summary
            
            return False, 0, {}
            
        except Exception as e:
            self.log_error(f"Error checking existing dataset: {e}")
            return False, 0, {}
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get detailed summary of the existing dataset.
        
        Returns:
            Dict with dataset summary information
        """
        try:
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            
            scanner = create_dataset_scanner(self.logger)
            
            if not scanner.quick_check_existing():
                return {}
                
            result = scanner.scan_existing_dataset_parallel()
            return result if result.get('status') == 'success' else {}
            
        except Exception as e:
            self.log_error(f"Error getting dataset summary: {e}")
            return {}
