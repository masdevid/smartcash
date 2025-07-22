"""
File: smartcash/ui/model/mixins/model_discovery_mixin.py
Description: Mixin for model checkpoint discovery and file scanning functionality.

Centralizes all checkpoint discovery logic to eliminate duplication across
backbone, training, evaluation, and pretrained modules.
"""

import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import torch

from smartcash.common.logger import get_logger


class ModelDiscoveryMixin:
    """
    Mixin for model checkpoint discovery and file scanning.
    
    Provides standardized functionality for:
    - Configurable checkpoint discovery across multiple paths
    - Enhanced filename parsing with regex patterns
    - Model metadata extraction and validation
    - Cross-module consistent file scanning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use parent module's logging system instead of separate logger
        # self._discovery_logger will be handled by delegation methods
    
    def _log_discovery(self, message: str, level: str = 'info') -> None:
        """
        Log discovery messages using the parent module's logging system.
        Falls back to basic print if module logging isn't available.
        """
        try:
            # Try to use parent module's logging system first
            if hasattr(self, 'log_info') and level == 'info':
                self.log_info(message)
            elif hasattr(self, 'log_debug') and level == 'debug':
                self.log_debug(message)
            elif hasattr(self, 'log_warning') and level == 'warning':
                self.log_warning(message)
            elif hasattr(self, 'log_error') and level == 'error':
                self.log_error(message)
            elif hasattr(self, 'log'):
                self.log(message, level)
            else:
                # Fallback to suppress console output (avoid double logging)
                pass
        except Exception:
            # Suppress any logging errors to avoid breaking discovery
            pass
    
    def discover_checkpoints(
        self, 
        discovery_paths: List[str] = None, 
        filename_patterns: List[str] = None,
        validation_requirements: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover checkpoint files using configurable paths and patterns.
        
        Args:
            discovery_paths: List of paths to search for checkpoints
            filename_patterns: List of glob patterns to match
            validation_requirements: Validation criteria for checkpoints
            
        Returns:
            List of discovered checkpoint metadata dictionaries
        """
        # Use default paths if not provided (sync with evaluation_config.yaml)
        if discovery_paths is None:
            discovery_paths = [
                'data/checkpoints',                     # Primary checkpoint directory
                'runs/train/*/weights',                 # YOLOv5 training output pattern  
                'experiments/*/checkpoints'             # Alternative experiment directory
            ]
        
        # Use default patterns if not provided (sync with evaluation_config.yaml)
        if filename_patterns is None:
            filename_patterns = [
                'best_*.pt',                           # Standard best model format
                'best_{model}_{backbone}_{mode}_{date}.pt',  # Enhanced format
                'last.pt',                             # Latest checkpoint fallback
                'epoch_*.pt'                           # Epoch-based checkpoints
            ]
        
        discovered_checkpoints = []
        total_files_scanned = 0
        
        # Scan each discovery path
        for discovery_path in discovery_paths:
            discovery_path_obj = Path(discovery_path)
            
            # Handle wildcard patterns in paths (e.g., runs/train/*/weights)
            if '*' in str(discovery_path):
                expanded_paths = glob.glob(str(discovery_path))
                paths_to_scan = [Path(p) for p in expanded_paths if Path(p).exists()]
            else:
                paths_to_scan = [discovery_path_obj] if discovery_path_obj.exists() else []
            
            if not paths_to_scan:
                self._log_discovery(f"Discovery path not found or no matches: {discovery_path}", 'debug')
                continue
            
            # Scan each expanded path
            for scan_path in paths_to_scan:
                self._log_discovery(f"📁 Scanning directory: {scan_path}", 'debug')
                
                # Apply each filename pattern
                for pattern in filename_patterns:
                    checkpoint_files = list(scan_path.glob(pattern))
                    
                    for checkpoint_file in checkpoint_files:
                        total_files_scanned += 1
                        
                        # Extract metadata and validate
                        metadata = self.extract_metadata_from_filename(
                            str(checkpoint_file), 
                            checkpoint_file.name
                        )
                        
                        # Apply validation if requirements provided
                        if validation_requirements:
                            is_valid, validation_msg = self.validate_checkpoint_file(
                                str(checkpoint_file), 
                                validation_requirements
                            )
                            metadata['valid'] = is_valid
                            metadata['validation_message'] = validation_msg
                            
                            if not is_valid:
                                self._log_discovery(f"⚠️ Invalid checkpoint {checkpoint_file.name}: {validation_msg}", 'debug')
                                continue
                        else:
                            metadata['valid'] = True
                        
                        discovered_checkpoints.append(metadata)
        
        # Sort by modification time (newest first) and limit to 5 latest
        discovered_checkpoints = sorted(
            discovered_checkpoints, 
            key=lambda x: Path(x['path']).stat().st_mtime, 
            reverse=True
        )[:5]
        
        self._log_discovery(f"📋 Discovered {len(discovered_checkpoints)} valid checkpoints from {total_files_scanned} files scanned", 'info')
        return discovered_checkpoints
    
    def scan_directory(
        self, 
        directory: str, 
        extensions: List[str] = None,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan directory for model files with specified extensions.
        
        Args:
            directory: Directory path to scan
            extensions: List of file extensions to include
            recursive: Whether to scan subdirectories
            
        Returns:
            List of file metadata dictionaries
        """
        if extensions is None:
            extensions = ['.pt', '.pth', '.onnx', '.pb', '.tflite', '.bin']
        
        directory_path = Path(directory)
        if not directory_path.exists():
            self._log_discovery(f"Directory does not exist: {directory}", 'warning')
            return []
        
        files_found = []
        
        # Scan for files
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                file_metadata = {
                    'path': str(file_path),
                    'filename': file_path.name,
                    'extension': file_path.suffix,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'directory': str(file_path.parent)
                }
                files_found.append(file_metadata)
        
        self._log_discovery(f"📁 Scanned {directory}: found {len(files_found)} files", 'debug')
        return files_found
    
    def extract_metadata_from_filename(
        self, 
        filepath: str, 
        filename: str,
        custom_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from checkpoint filename using enhanced regex patterns.
        
        Args:
            filepath: Full path to the checkpoint file
            filename: Filename to parse
            custom_patterns: Custom regex patterns to use
            
        Returns:
            Dictionary containing extracted metadata
        """
        filepath_obj = Path(filepath)
        
        # Base metadata
        metadata = {
            'path': filepath,
            'filename': filename,
            'display_name': filename,
            'model_name': 'unknown',
            'backbone': 'unknown',
            'layer_mode': 'unknown',
            'date': 'unknown',
            'file_size_mb': filepath_obj.stat().st_size / (1024 * 1024) if filepath_obj.exists() else 0,
            'valid': True
        }
        
        # Enhanced patterns for various checkpoint naming conventions
        patterns = custom_patterns or [
            r'best_(\w+)_(efficientnet_b4)_(\w+)_(\d{8})\.pt',    # EfficientNet-B4 specific
            r'best_(\w+)_(cspdarknet)_(\w+)_(\d{8})\.pt',         # CSPDarknet specific  
            r'best_(\w+)_(\w+)_(\w+)_(\d{8})\.pt',               # General pattern
            r'best_(\w+)_(b4)_(\w+)_(\d{8})\.pt',                 # Short form B4
            r'best_.*_(efficientnet).*\.pt',                         # General EfficientNet
            r'best_.*_(yolov5).*\.pt',                               # YOLOv5 models
            r'(\w+)_(efficientnet_b4|cspdarknet)_(\w+)\.pt',      # Alternative format
        ]
        
        # Try to parse filename with each pattern
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                
                # Extract based on number of groups
                if len(groups) >= 4:
                    metadata['model_name'] = groups[0]
                    metadata['backbone'] = self._normalize_backbone_name(groups[1])
                    metadata['layer_mode'] = groups[2]
                    
                    # Try to parse date
                    try:
                        date_str = groups[3]
                        date_obj = datetime.strptime(date_str, '%m%d%Y')
                        metadata['date'] = date_obj.strftime('%d/%m/%Y')
                    except ValueError:
                        metadata['date'] = groups[3]
                        
                elif len(groups) >= 2:
                    metadata['backbone'] = self._normalize_backbone_name(groups[-1])
                    if len(groups) >= 3:
                        metadata['layer_mode'] = groups[-2]
                
                # Create display name
                backbone = metadata['backbone']
                date = metadata['date']
                metadata['display_name'] = f"{backbone.title()} - {date}"
                break
        
        return metadata
    
    def validate_checkpoint_file(
        self, 
        filepath: str, 
        requirements: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Validate checkpoint file against requirements.
        
        Args:
            filepath: Path to checkpoint file
            requirements: Validation requirements dictionary
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        filepath_obj = Path(filepath)
        
        # Check file exists
        if not filepath_obj.exists():
            return False, f"File does not exist: {filepath_obj.name}"
        
        if not requirements:
            return True, "No validation requirements specified"
        
        try:
            # Load checkpoint for validation
            checkpoint_data = torch.load(filepath, map_location='cpu')
            
            # Check required keys
            required_keys = requirements.get('required_keys', [])
            missing_keys = [key for key in required_keys if key not in checkpoint_data]
            if missing_keys:
                return False, f"Missing required keys: {', '.join(missing_keys)}"
            
            # Check supported backbones
            supported_backbones = requirements.get('supported_backbones', [])
            if supported_backbones:
                model_config = checkpoint_data.get('config', {})
                backbone = model_config.get('backbone', 'unknown')
                if backbone not in supported_backbones:
                    return False, f"Unsupported backbone: {backbone} (supported: {', '.join(supported_backbones)})"
            
            # Check minimum mAP threshold
            min_val_map = requirements.get('min_val_map')
            if min_val_map is not None:
                metrics = checkpoint_data.get('metrics', {})
                val_map = metrics.get('val_map', 0)
                if val_map < min_val_map:
                    return False, f"mAP below threshold: {val_map:.3f} < {min_val_map}"
            
            return True, "Checkpoint validation passed"
            
        except Exception as e:
            return False, f"Error loading checkpoint: {str(e)}"
    
    def get_checkpoint_stats(
        self, 
        checkpoints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about discovered checkpoints.
        
        Args:
            checkpoints: List of checkpoint metadata dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not checkpoints:
            return {
                'total_count': 0,
                'backbone_distribution': {},
                'size_stats': {},
                'latest_date': None,
                'average_size_mb': 0
            }
        
        # Calculate statistics
        backbone_counts = {}
        sizes = []
        dates = []
        
        for checkpoint in checkpoints:
            # Count by backbone
            backbone = checkpoint.get('backbone', 'unknown')
            backbone_counts[backbone] = backbone_counts.get(backbone, 0) + 1
            
            # Collect sizes
            size_mb = checkpoint.get('file_size_mb', 0)
            if size_mb > 0:
                sizes.append(size_mb)
            
            # Collect dates
            date_str = checkpoint.get('date', '')
            if date_str and date_str != 'unknown':
                try:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    dates.append(date_obj)
                except ValueError:
                    pass
        
        # Size statistics
        size_stats = {}
        if sizes:
            size_stats = {
                'min_mb': min(sizes),
                'max_mb': max(sizes),
                'average_mb': sum(sizes) / len(sizes),
                'total_mb': sum(sizes)
            }
        
        return {
            'total_count': len(checkpoints),
            'backbone_distribution': backbone_counts,
            'size_stats': size_stats,
            'latest_date': max(dates) if dates else None,
            'average_size_mb': size_stats.get('average_mb', 0)
        }
    
    def _normalize_backbone_name(self, backbone: str) -> str:
        """Normalize backbone name for consistency."""
        backbone_lower = backbone.lower()
        
        # Normalize common variations
        if backbone_lower in ['b4', 'efficientnet', 'efficientnet_b4']:
            return 'efficientnet_b4'
        elif backbone_lower in ['cspdarknet', 'csp', 'yolov5']:
            return 'cspdarknet'
        else:
            return backbone_lower