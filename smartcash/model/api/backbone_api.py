"""
File: smartcash/model/api/backbone_api.py
Description: Backend API for backbone module - data validation and model discovery.
"""

import glob
import os
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import logging

# Use module-level logger for backend API
logger = logging.getLogger(__name__)


def check_data_prerequisites() -> Dict[str, Any]:
    """
    Check data prerequisites for backbone training.
    Backend API endpoint for data validation.
    
    Returns:
        Dict containing validation results
    """
    try:
        # Check for pretrained models
        pretrained_result = check_pretrained_models()
        
        # Check for raw data availability  
        raw_data_result = check_raw_data()
        
        # Check for preprocessed data
        preprocessed_result = check_preprocessed_data()
        
        # Determine overall readiness
        prerequisites_ready = (
            pretrained_result.get('available', False) or
            raw_data_result.get('available', False) or 
            preprocessed_result.get('available', False)
        )
        
        return {
            'success': True,
            'prerequisites_ready': prerequisites_ready,
            'pretrained_models': pretrained_result,
            'raw_data': raw_data_result,
            'preprocessed_data': preprocessed_result,
            'message': _format_prerequisites_message(
                prerequisites_ready, pretrained_result, raw_data_result, preprocessed_result
            )
        }
        
    except Exception as e:
        error_msg = f"Error checking data prerequisites: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'prerequisites_ready': False,
            'message': error_msg
        }


def check_pretrained_models() -> Dict[str, Any]:
    """
    Check for pretrained model availability.
    
    Returns:
        Dict with pretrained model information
    """
    try:
        pretrained_paths = ['data/pretrained', '/content/data/pretrained']
        
        for path_str in pretrained_paths:
            path = Path(path_str)
            if path.exists():
                # Count pretrained model files
                model_files = list(path.glob('*.pt')) + list(path.glob('*.pth'))
                if model_files:
                    return {
                        'available': True,
                        'count': len(model_files),
                        'path': str(path),
                        'files': [f.name for f in model_files[:3]]  # Show first 3
                    }
        
        return {
            'available': False,
            'count': 0,
            'message': 'No pretrained models found'
        }
        
    except Exception as e:
        logger.error(f"Error checking pretrained models: {e}")
        return {
            'available': False,
            'count': 0,
            'error': str(e)
        }


def check_raw_data() -> Dict[str, Any]:
    """
    Check for raw data availability.
    
    Returns:
        Dict with raw data information
    """
    try:
        data_paths = ['data', '/content/data']
        
        for base_path in data_paths:
            base = Path(base_path)
            if not base.exists():
                continue
            
            total_images = 0
            total_labels = 0
            splits_found = []
            
            for split in ['train', 'valid', 'test']:
                split_path = base / split
                if split_path.exists():
                    # Count images and labels
                    images_path = split_path / 'images'
                    labels_path = split_path / 'labels'
                    
                    if images_path.exists():
                        images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                        total_images += len(images)
                    
                    if labels_path.exists():
                        labels = list(labels_path.glob('*.txt'))
                        total_labels += len(labels)
                    
                    if images_path.exists() or labels_path.exists():
                        splits_found.append(split)
            
            if total_images > 0 or total_labels > 0:
                return {
                    'available': True,
                    'total_images': total_images,
                    'total_labels': total_labels,
                    'splits': splits_found,
                    'path': str(base)
                }
        
        return {
            'available': False,
            'total_images': 0,
            'total_labels': 0,
            'message': 'No raw data found'
        }
        
    except Exception as e:
        logger.error(f"Error checking raw data: {e}")
        return {
            'available': False,
            'error': str(e)
        }


def check_preprocessed_data() -> Dict[str, Any]:
    """
    Check for preprocessed data availability.
    
    Returns:
        Dict with preprocessed data information
    """
    try:
        preprocessed_paths = ['data/preprocessed', '/content/data/preprocessed']
        
        for base_path in preprocessed_paths:
            base = Path(base_path)
            if not base.exists():
                continue
            
            total_files = 0
            splits_found = []
            
            for split in ['train', 'valid', 'test']:
                split_path = base / split
                if split_path.exists():
                    # Count preprocessed files (.npy)
                    images_path = split_path / 'images'
                    labels_path = split_path / 'labels'
                    
                    split_files = 0
                    if images_path.exists():
                        npy_files = list(images_path.glob('*.npy'))
                        split_files += len(npy_files)
                    
                    if labels_path.exists():
                        label_files = list(labels_path.glob('*.txt'))
                        split_files += len(label_files)
                    
                    if split_files > 0:
                        total_files += split_files
                        splits_found.append(split)
            
            if total_files > 0:
                return {
                    'available': True,
                    'total_files': total_files,
                    'splits': splits_found,
                    'path': str(base)
                }
        
        return {
            'available': False,
            'total_files': 0,
            'message': 'No preprocessed data found'
        }
        
    except Exception as e:
        logger.error(f"Error checking preprocessed data: {e}")
        return {
            'available': False,
            'error': str(e)
        }


def _format_prerequisites_message(ready: bool, pretrained: Dict, raw: Dict, preprocessed: Dict) -> str:
    """
    Format a human-readable message about data prerequisites.
    
    Args:
        ready: Overall readiness status
        pretrained: Pretrained models info
        raw: Raw data info  
        preprocessed: Preprocessed data info
        
    Returns:
        Formatted message string
    """
    if not ready:
        return "❌ No training data found. Please prepare data first."
    
    messages = []
    
    if pretrained.get('available'):
        count = pretrained.get('count', 0)
        messages.append(f"✅ {count} pretrained model(s)")
    
    if raw.get('available'):
        images = raw.get('total_images', 0)
        splits = len(raw.get('splits', []))
        messages.append(f"✅ {images} raw images ({splits} splits)")
    
    if preprocessed.get('available'):
        files = preprocessed.get('total_files', 0)
        splits = len(preprocessed.get('splits', []))
        messages.append(f"✅ {files} preprocessed files ({splits} splits)")
    
    return f"📊 Data ready: {', '.join(messages)}"


def check_built_models() -> Dict[str, Any]:
    """
    Check for built models by backbone type using optimized parallel discovery.
    Backend API endpoint for model discovery.
    
    Returns:
        Dictionary with discovery results organized by backbone type
    """
    try:
        # Define search paths with priorities - check both initial models and trained checkpoints
        search_configs = [
            {
                'paths': ['data/models'],  # Initial built models (higher priority)
                'patterns': [
                    '*backbone*efficientnet*.pt', '*backbone*efficientnet*.pth',   # EfficientNet backbone models
                    '*backbone*cspdarknet*.pt', '*backbone*cspdarknet*.pth',       # CSPDarkNet backbone models
                    '*backbone*.pt', '*backbone*.pth',                            # General backbone files
                    '*efficientnet*.pt', '*efficientnet*.pth',                    # EfficientNet models
                    '*cspdarknet*.pt', '*cspdarknet*.pth',                        # CSPDarkNet models
                    '*smartcash*.pt', '*smartcash*.pth'                           # SmartCash models
                ],
                'priority': 1
            },
            {
                'paths': ['data/checkpoints'],  # Trained model checkpoints (lower priority)
                'patterns': [
                    '*backbone*smartcash*efficientnet*.pt', '*backbone*smartcash*efficientnet*.pth',  # EfficientNet format
                    '*backbone*smartcash*cspdarknet*.pt', '*backbone*smartcash*cspdarknet*.pth',     # CSPDarkNet format
                    '*backbone*.pt', '*backbone*.pth',                                               # General backbone files
                    '*smartcash*.pt', '*smartcash*.pth'                                              # SmartCash models
                ],
                'priority': 2
            },
            {
                'paths': ['runs/train/*/weights'],
                'patterns': ['best.pt', 'best_*.pt', 'last_*.pt', 'last.pt'],
                'priority': 3
            },
            {
                'paths': ['data/models'],
                'patterns': ['*.pt', '*.pth'],
                'priority': 4
            }
        ]
        
        # Parallel discovery across all configurations
        all_discovered_models = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="ModelScan") as executor:
            # Submit search tasks for each configuration
            future_to_config = {}
            for config in search_configs:
                for path in config['paths']:
                    future = executor.submit(_scan_path_pattern, path, config['patterns'])
                    future_to_config[future] = {
                        'path': path, 
                        'patterns': config['patterns'],
                        'priority': config['priority']
                    }
            
            # Collect results as they complete
            for future in as_completed(future_to_config, timeout=10):
                try:
                    config_info = future_to_config[future]
                    models_found = future.result(timeout=2)
                    
                    # Merge results with priority handling
                    for model_key, model_info in models_found.items():
                        model_info['discovery_priority'] = config_info['priority']
                        
                        # Keep higher priority discoveries
                        if model_key not in all_discovered_models:
                            all_discovered_models[model_key] = model_info
                        elif config_info['priority'] < all_discovered_models[model_key].get('discovery_priority', 999):
                            all_discovered_models[model_key] = model_info
                            
                except Exception:
                    # Suppress individual task failure logs to reduce noise
                    continue
        
        # Organize by backbone type
        by_backbone = {}
        for model_key, model_info in all_discovered_models.items():
            backbone = model_info['metadata'].get('backbone', 'unknown')
            normalized_backbone = _normalize_backbone_name(backbone)
            
            if normalized_backbone not in by_backbone:
                by_backbone[normalized_backbone] = []
            
            by_backbone[normalized_backbone].append({
                'path': model_info['filepath'],
                'size_mb': model_info['metadata'].get('size_mb', 0),
                'timestamp': model_info['metadata'].get('timestamp', 'unknown'),
                'model_type': model_info['metadata'].get('model_type', 'unknown'),
                'accuracy': model_info['metadata'].get('accuracy', 'N/A'),
                'found_in': model_info['found_in_path'],
                'pattern': model_info['pattern']
            })
        
        # Sort models by timestamp (newest first)
        for backbone_type in by_backbone:
            by_backbone[backbone_type].sort(
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
        
        total_models = sum(len(models) for models in by_backbone.values())
        
        return {
            'success': True,
            'total_models': total_models,
            'by_backbone': by_backbone,
            'discovery_summary': f"Found {total_models} models across {len(by_backbone)} backbone types"
        }
        
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        return _fallback_model_scan()


def _scan_path_pattern(path_pattern: str, file_patterns: List[str]) -> Dict[str, Any]:
    """
    Scan a specific path with multiple patterns.
    
    Args:
        path_pattern: Path to scan
        file_patterns: List of file patterns to match
        
    Returns:
        Dictionary of found models with metadata
    """
    found_models = {}
    
    try:
        # Convert to absolute path for better reliability
        from pathlib import Path
        abs_path = Path(path_pattern).resolve()
        
        # Check if directory exists before scanning
        if not abs_path.exists():
            return found_models
            
        total_files = 0
        valid_models = 0
        
        for pattern in file_patterns:
            full_pattern = str(abs_path / pattern)
            files = glob.glob(full_pattern)
            total_files += len(files)
            
            for filepath in files:
                try:
                    metadata = extract_quick_metadata(filepath)
                    if metadata and metadata.get('valid', False):
                        model_key = f"{metadata.get('backbone', 'unknown')}_{metadata.get('timestamp', 'unknown')}"
                        found_models[model_key] = {
                            'filepath': filepath,
                            'metadata': metadata,
                            'found_in_path': path_pattern,
                            'pattern': pattern
                        }
                        valid_models += 1
                except Exception:
                    # Suppress per-file error logs to reduce noise
                    continue
        
        # Use summary logging instead of per-file logging
        if total_files > 0:
            pass  # Remove debug logging to avoid import issues
    except Exception as e:
        pass  # Remove debug logging to avoid import issues
    
    return found_models


def extract_quick_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from model file quickly without loading the full model.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Dictionary with extracted metadata
    """
    try:
        path_obj = Path(filepath)
        
        # Basic file validation
        if not path_obj.exists() or path_obj.suffix not in ['.pt', '.pth']:
            return {'valid': False, 'error': 'Invalid file'}
        
        # File size
        size_mb = path_obj.stat().st_size / (1024 * 1024)
        if size_mb < 1:  # Less than 1MB is probably not a valid model
            return {'valid': False, 'error': 'File too small'}
        
        # Extract info from filename
        filename = path_obj.stem
        metadata = {
            'valid': True,
            'filepath': filepath,
            'filename': filename,
            'size_mb': round(size_mb, 1),
            'timestamp': _extract_timestamp_from_filename(filename),
            'backbone': _extract_backbone_from_filename(filename),
            'model_type': _infer_model_type_from_filename(filename)
        }
        
        # Try to extract additional metadata from the file
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                # Extract metadata from checkpoint
                if 'metadata' in checkpoint:
                    metadata.update(checkpoint['metadata'])
                elif 'model_info' in checkpoint:
                    metadata.update(checkpoint['model_info'])
                
                # Extract accuracy if available
                if 'best_fitness' in checkpoint:
                    metadata['accuracy'] = checkpoint['best_fitness']
                elif 'best_mAP' in checkpoint:
                    metadata['accuracy'] = checkpoint['best_mAP']
                    
        except Exception:
            # If loading fails, stick with filename-based metadata
            pass
        
        return metadata
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def _extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename."""
    import re
    # Look for patterns like 20250723_1209
    timestamp_pattern = r'(\d{8}_\d{4})'
    match = re.search(timestamp_pattern, filename)
    return match.group(1) if match else 'unknown'


def _extract_backbone_from_filename(filename: str) -> str:
    """Extract backbone type from filename."""
    filename_lower = filename.lower()
    
    if 'efficientnet' in filename_lower:
        if 'b0' in filename_lower:
            return 'efficientnet_b0'
        elif 'b4' in filename_lower:
            return 'efficientnet_b4'
        else:
            return 'efficientnet'
    elif 'cspdarknet' in filename_lower:
        return 'cspdarknet'
    elif 'yolo' in filename_lower:
        return 'yolov5'
    else:
        return 'unknown'


def _infer_model_type_from_filename(filename: str) -> str:
    """Infer model type from filename."""
    filename_lower = filename.lower()
    
    if 'backbone' in filename_lower:
        return 'backbone'
    elif 'yolo' in filename_lower:
        return 'yolo'
    elif 'detection' in filename_lower:
        return 'detection'
    else:
        return 'model'


def _normalize_backbone_name(backbone: str) -> str:
    """
    Normalize backbone name for consistent organization.
    
    Args:
        backbone: Raw backbone name
        
    Returns:
        Normalized backbone name
    """
    backbone_lower = backbone.lower().strip()
    
    # EfficientNet variations
    if 'efficientnet' in backbone_lower:
        if 'b0' in backbone_lower:
            return 'efficientnet_b0'
        elif 'b4' in backbone_lower:
            return 'efficientnet_b4'
        else:
            return 'efficientnet'
    
    # CSPDarkNet variations
    elif 'csp' in backbone_lower or 'darknet' in backbone_lower:
        return 'cspdarknet'
    
    # YOLO variations
    elif 'yolo' in backbone_lower:
        return 'yolov5'
    
    # Default case
    else:
        return backbone_lower if backbone_lower else 'unknown'


def _fallback_model_scan() -> Dict[str, Any]:
    """
    Fallback model scanning when main discovery fails.
    
    Returns:
        Basic scan results
    """
    try:
        # Simple scan of common paths - prioritize initial models over checkpoints
        common_paths = ['data/models', 'data/checkpoints']
        total_found = 0
        
        for path_str in common_paths:
            path = Path(path_str)
            if path.exists():
                model_files = list(path.glob('*.pt')) + list(path.glob('*.pth'))
                total_found += len(model_files)
        
        return {
            'success': True,
            'total_models': total_found,
            'by_backbone': {'unknown': []} if total_found > 0 else {},
            'discovery_summary': f"Fallback scan found {total_found} model files",
            'fallback_used': True
        }
        
    except Exception as e:
        logger.error(f"Fallback model scan failed: {e}")
        return {
            'success': False,
            'total_models': 0,
            'by_backbone': {},
            'discovery_summary': 'No models found',
            'error': str(e)
        }