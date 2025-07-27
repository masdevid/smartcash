"""
File: smartcash/model/evaluation/checkpoint_selector.py
Deskripsi: UI checkpoint selection untuk evaluation dengan filtering dan metadata display
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import torch

from smartcash.common.logger import get_logger

class CheckpointSelector:
    """Selector checkpoint untuk evaluation dengan UI integration"""
    
    def __init__(self, checkpoints_dir: str = None, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Extract checkpoint configuration with fallbacks
        checkpoint_config = self.config.get('evaluation', {}).get('checkpoints', {})
        
        # Use configurable discovery paths, not hardcoded
        if checkpoints_dir:
            self.discovery_paths = [Path(checkpoints_dir)]
        else:
            discovery_paths = checkpoint_config.get('discovery_paths', ['data/checkpoints'])
            self.discovery_paths = [Path(path) for path in discovery_paths]
        
        # Get filename patterns from config
        self.filename_patterns = checkpoint_config.get('filename_patterns', ['best_*.pt'])
        
        # Get validation requirements from config
        self.required_keys = checkpoint_config.get('required_keys', ['model_state_dict', 'config'])
        self.supported_backbones = checkpoint_config.get('supported_backbones', ['cspdarknet', 'efficientnet_b4'])
        self.min_val_map = checkpoint_config.get('min_val_map', 0.3)
        
        self.logger = get_logger('checkpoint_selector')
        self._checkpoint_cache = {}
        
        self.logger.info(f"🔍 Checkpoint selector initialized with {len(self.discovery_paths)} discovery paths")
        
    def list_available_checkpoints(self, refresh_cache: bool = False) -> List[Dict[str, Any]]:
        """📋 List semua checkpoint tersedia dengan metadata"""
        if refresh_cache or not self._checkpoint_cache:
            self._scan_checkpoints()
        
        checkpoints = list(self._checkpoint_cache.values())
        sort_by = self.config.get('checkpoints', {}).get('sort_by', 'val_map')
        reverse = sort_by != 'val_loss'  # Descending except for loss
        
        # Sort dengan safe key access
        checkpoints.sort(
            key=lambda x: x.get('metrics', {}).get(sort_by, 0 if reverse else float('inf')), 
            reverse=reverse
        )
        
        max_checkpoints = self.config.get('checkpoints', {}).get('max_checkpoints', 10)
        return checkpoints[:max_checkpoints]
    
    def filter_checkpoints(self, backbone: Optional[str] = None, min_map: Optional[float] = None, 
                          layer_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """🔍 Filter checkpoint berdasarkan criteria"""
        checkpoints = self.list_available_checkpoints()
        filtered = checkpoints
        
        if backbone:
            filtered = [cp for cp in filtered if cp.get('backbone', '').lower() == backbone.lower()]
        
        if min_map is not None:
            filtered = [cp for cp in filtered if cp.get('metrics', {}).get('val_map', 0) >= min_map]
            
        if layer_mode:
            filtered = [cp for cp in filtered if cp.get('layer_mode', '') == layer_mode]
        
        self.logger.info(f"🔍 Filtered {len(filtered)}/{len(checkpoints)} checkpoints")
        return filtered
    
    def select_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """✅ Select dan validate checkpoint untuk evaluation"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
        
        # Load dan validate checkpoint
        try:
            # Use safe globals for PyTorch 2.6+ compatibility
            import torch.serialization
            try:
                from models.yolo import Model as YOLOModel
                from models.common import Conv, C3, SPPF, Bottleneck
                safe_globals = [YOLOModel, Conv, C3, SPPF, Bottleneck]
            except ImportError:
                safe_globals = []
            
            with torch.serialization.safe_globals(safe_globals):
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract metadata
            metadata = self._extract_checkpoint_metadata(checkpoint_path, checkpoint_data)
            
            self.logger.info(f"✅ Checkpoint selected: {metadata['display_name']}")
            
            # Safe format for mAP value
            val_map = metadata['metrics'].get('val_map', 0)
            if isinstance(val_map, (int, float)):
                self.logger.info(f"   🎯 mAP: {val_map:.3f}")
            else:
                self.logger.info(f"   🎯 mAP: {val_map}")
            
            self.logger.info(f"   🏗️ Backbone: {metadata['backbone']}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error loading checkpoint: {str(e)}")
            raise
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """📊 Detailed checkpoint information"""
        checkpoint_path = Path(checkpoint_path)
        
        if str(checkpoint_path) in self._checkpoint_cache:
            return self._checkpoint_cache[str(checkpoint_path)]
        
        return self._extract_checkpoint_metadata(checkpoint_path)
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """✅ Validate checkpoint compatibility"""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                return False, f"File tidak ditemukan: {checkpoint_path.name}"
            
            # Load checkpoint header
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate required keys
            missing_keys = [key for key in self.required_keys if key not in checkpoint_data]
            
            if missing_keys:
                return False, f"Missing keys: {', '.join(missing_keys)}"
            
            # Validate model config
            model_config = checkpoint_data.get('config', {})
            backbone = model_config.get('backbone', 'unknown')
            
            if backbone not in self.supported_backbones:
                return False, f"Backbone tidak didukung: {backbone} (supported: {', '.join(self.supported_backbones)})"
            
            return True, "Checkpoint valid untuk evaluation"
            
        except Exception as e:
            return False, f"Error validating checkpoint: {str(e)}"
    
    def _scan_checkpoints(self) -> None:
        """🔍 Scan discovery paths for checkpoints and build cache using configurable patterns"""
        import glob
        self._checkpoint_cache.clear()
        
        total_files_found = 0
        
        for discovery_path in self.discovery_paths:
            self.logger.debug(f"🔍 Scanning discovery path: {discovery_path}")
            
            # Handle wildcard patterns in paths (e.g., runs/train/*/weights)
            if '*' in str(discovery_path):
                # Use glob to expand wildcard patterns
                expanded_paths = glob.glob(str(discovery_path))
                paths_to_scan = [Path(p) for p in expanded_paths if Path(p).exists()]
            else:
                paths_to_scan = [discovery_path] if discovery_path.exists() else []
            
            if not paths_to_scan:
                self.logger.debug(f"⚠️ Discovery path not found or no matches: {discovery_path}")
                continue
            
            # Scan each expanded path
            for scan_path in paths_to_scan:
                self.logger.debug(f"📁 Scanning directory: {scan_path}")
                
                # Apply each filename pattern
                for pattern in self.filename_patterns:
                    checkpoint_files = list(scan_path.glob(pattern))
                    
                    for checkpoint_file in checkpoint_files:
                        try:
                            # Validate checkpoint before adding to cache
                            is_valid, validation_msg = self.validate_checkpoint(str(checkpoint_file))
                            
                            if is_valid:
                                metadata = self._extract_checkpoint_metadata(checkpoint_file)
                                
                                # Apply minimum mAP filter
                                val_map = metadata.get('metrics', {}).get('val_map', 0)
                                if val_map >= self.min_val_map:
                                    self._checkpoint_cache[str(checkpoint_file)] = metadata
                                    total_files_found += 1
                                else:
                                    self.logger.debug(f"⚠️ Checkpoint {checkpoint_file.name} below minimum mAP threshold: {val_map:.3f} < {self.min_val_map}")
                            else:
                                self.logger.debug(f"⚠️ Invalid checkpoint {checkpoint_file.name}: {validation_msg}")
                                
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error reading {checkpoint_file.name}: {str(e)}")
        
        self.logger.info(f"📋 Scanned {len(self._checkpoint_cache)} valid checkpoints from {len(self.discovery_paths)} discovery paths")
        self.logger.info(f"🎯 Found {total_files_found} checkpoints meeting criteria (mAP >= {self.min_val_map})")
    
    def _extract_checkpoint_metadata(self, checkpoint_path: Path, 
                                   checkpoint_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """📊 Extract metadata dari checkpoint"""
        if checkpoint_data is None:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Parse filename pattern: best_{model}_{backbone}_{mode}_{date}.pt
        # Enhanced pattern to handle various backbone naming conventions
        filename_patterns = [
            r'best_(\w+)_(\w+)_(\w+)_(\d{8})\.pt',  # Original pattern
            r'best_(\w+)_(efficientnet_b4)_(\w+)_(\d{8})\.pt',  # EfficientNet-B4 specific
            r'best_(\w+)_(cspdarknet)_(\w+)_(\d{8})\.pt',  # CSPDarknet specific
        ]
        
        matched = False
        for pattern in filename_patterns:
            match = re.match(pattern, checkpoint_path.name)
            if match:
                model_name, backbone, layer_mode, date_str = match.groups()
                
                # Normalize backbone names
                if backbone in ['b4', 'efficientnet_b4']:
                    backbone = 'efficientnet_b4'
                elif backbone in ['cspdarknet', 'csp']:
                    backbone = 'cspdarknet'
                
                try:
                    date_obj = datetime.strptime(date_str, '%m%d%Y')
                    formatted_date = date_obj.strftime('%d/%m/%Y')
                except ValueError:
                    formatted_date = date_str
                
                matched = True
                break
        
        if not matched:
            # Fallback parsing
            model_name = 'smartcash'
            backbone = 'unknown'
            layer_mode = 'single'
            formatted_date = 'unknown'
        
        # Extract metrics
        metrics = checkpoint_data.get('metrics', {})
        config = checkpoint_data.get('config', {})
        
        # Calculate file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        return {
            'path': str(checkpoint_path),
            'filename': checkpoint_path.name,
            'display_name': f"{backbone.title()} - {formatted_date}",
            'model_name': model_name,
            'backbone': backbone,
            'layer_mode': layer_mode,
            'date': formatted_date,
            'metrics': metrics,
            'config': config,
            'file_size_mb': round(file_size_mb, 2),
            'epoch': checkpoint_data.get('epoch', 0),
            'valid': True
        }
    
    def get_best_checkpoint(self, backbone: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """🏆 Get checkpoint terbaik berdasarkan mAP"""
        checkpoints = self.filter_checkpoints(backbone=backbone)
        
        if not checkpoints:
            return None
        
        # Auto-select best berdasarkan config
        auto_select = self.config.get('checkpoints', {}).get('auto_select_best', True)
        if auto_select:
            return checkpoints[0]  # Already sorted by val_map
        
        return None
    
    def create_checkpoint_options(self) -> List[Tuple[str, str]]:
        """🎨 Create options untuk UI dropdown"""
        checkpoints = self.list_available_checkpoints()
        
        options = []
        for cp in checkpoints:
            val_map = cp['metrics'].get('val_map', 0)
            val_loss = cp['metrics'].get('val_loss', 0)
            
            label = f"{cp['display_name']} | mAP: {val_map:.3f} | Loss: {val_loss:.3f}"
            options.append((label, cp['path']))
        
        return options
    
    def get_backbone_stats(self) -> Dict[str, Dict[str, Any]]:
        """📊 Statistics per backbone untuk comparison"""
        checkpoints = self.list_available_checkpoints()
        
        backbone_stats = {}
        for cp in checkpoints:
            backbone = cp['backbone']
            if backbone not in backbone_stats:
                backbone_stats[backbone] = {
                    'count': 0,
                    'best_map': 0,
                    'avg_map': 0,
                    'checkpoints': []
                }
            
            stats = backbone_stats[backbone]
            stats['count'] += 1
            stats['checkpoints'].append(cp)
            
            val_map = cp['metrics'].get('val_map', 0)
            stats['best_map'] = max(stats['best_map'], val_map)
        
        # Calculate averages
        for backbone, stats in backbone_stats.items():
            maps = [cp['metrics'].get('val_map', 0) for cp in stats['checkpoints']]
            stats['avg_map'] = sum(maps) / len(maps) if maps else 0
        
        return backbone_stats


# Factory functions untuk UI integration
def create_checkpoint_selector(config: Dict[str, Any] = None) -> CheckpointSelector:
    """🏭 Factory untuk CheckpointSelector"""
    return CheckpointSelector(config=config)

def get_available_checkpoints(config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """📋 One-liner untuk list available checkpoints"""
    return create_checkpoint_selector(config).list_available_checkpoints()

def select_best_checkpoint(backbone: str = None, config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """🏆 One-liner untuk select best checkpoint"""
    return create_checkpoint_selector(config).get_best_checkpoint(backbone)