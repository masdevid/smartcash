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
    
    def __init__(self, checkpoints_dir: str = 'data/checkpoints', config: Dict[str, Any] = None):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.config = config or {}
        self.logger = get_logger('checkpoint_selector')
        self._checkpoint_cache = {}
        
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
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract metadata
            metadata = self._extract_checkpoint_metadata(checkpoint_path, checkpoint_data)
            
            self.logger.info(f"✅ Checkpoint selected: {metadata['display_name']}")
            self.logger.info(f"   🎯 mAP: {metadata['metrics'].get('val_map', 'N/A'):.3f}")
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
            required_keys = ['model_state_dict', 'config']
            missing_keys = [key for key in required_keys if key not in checkpoint_data]
            
            if missing_keys:
                return False, f"Missing keys: {', '.join(missing_keys)}"
            
            # Validate model config
            model_config = checkpoint_data.get('config', {})
            backbone = model_config.get('backbone', 'unknown')
            
            if backbone not in ['cspdarknet', 'efficientnet_b4']:
                return False, f"Backbone tidak didukung: {backbone}"
            
            return True, "Checkpoint valid untuk evaluation"
            
        except Exception as e:
            return False, f"Error validating checkpoint: {str(e)}"
    
    def _scan_checkpoints(self) -> None:
        """🔍 Scan direktori checkpoint dan build cache"""
        self._checkpoint_cache.clear()
        
        if not self.checkpoints_dir.exists():
            self.logger.warning(f"⚠️ Direktori checkpoint tidak ditemukan: {self.checkpoints_dir}")
            return
        
        checkpoint_files = list(self.checkpoints_dir.glob('*.pt'))
        
        for checkpoint_file in checkpoint_files:
            try:
                metadata = self._extract_checkpoint_metadata(checkpoint_file)
                self._checkpoint_cache[str(checkpoint_file)] = metadata
            except Exception as e:
                self.logger.debug(f"⚠️ Error reading {checkpoint_file.name}: {str(e)}")
        
        self.logger.info(f"📋 Scanned {len(self._checkpoint_cache)} valid checkpoints")
    
    def _extract_checkpoint_metadata(self, checkpoint_path: Path, 
                                   checkpoint_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """📊 Extract metadata dari checkpoint"""
        if checkpoint_data is None:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Parse filename pattern: best_{model}_{backbone}_{mode}_{date}.pt
        filename_pattern = r'best_(\w+)_(\w+)_(\w+)_(\d{8})\.pt'
        match = re.match(filename_pattern, checkpoint_path.name)
        
        if match:
            model_name, backbone, layer_mode, date_str = match.groups()
            try:
                date_obj = datetime.strptime(date_str, '%m%d%Y')
                formatted_date = date_obj.strftime('%d/%m/%Y')
            except ValueError:
                formatted_date = date_str
        else:
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