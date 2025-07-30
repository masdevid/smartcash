"""
File: smartcash/dataset/augmentor/balancer/balancer.py
Deskripsi: Refactored balancer dengan preserved business logic
"""

from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter
from smartcash.common.logger import get_logger

# One-liner constants
get_layer1_classes = lambda: {'0', '1', '2', '3', '4', '5', '6'}
get_layer2_classes = lambda: {'7', '8', '9', '10', '11', '12', '13'}
get_layer3_classes = lambda: {'14', '15', '16'}
get_all_target_classes = lambda: get_layer1_classes().union(get_layer2_classes())
is_target_class = lambda cls: cls in get_all_target_classes()
is_layer3_class = lambda cls: cls in get_layer3_classes()
calc_need = lambda current, target: max(0, target - current)
calc_priority_multiplier = lambda cls: 1.5 if cls in get_layer1_classes() else 1.2 if cls in get_layer2_classes() else 1.0

class ClassBalancingStrategy:
    """⚖️ Strategy balancing dengan preserved business logic"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.communicator = communicator
        self.target_weights = self._setup_target_weights()
    
    def _setup_target_weights(self) -> Dict[str, float]:
        """Setup bobot target untuk setiap layer"""
        return {
            **{cls: 1.0 for cls in get_layer1_classes()},
            **{cls: 0.8 for cls in get_layer2_classes()},
        }
    
    def calculate_balancing_needs_split_aware(self, data_dir: str, target_split: str, target_count: int = 1000) -> Dict[str, int]:
        """Calculate needs dengan preserved logic"""
        try:
            from smartcash.dataset.augmentor.utils.file_processor import FileProcessor
            
            file_processor = FileProcessor({'data': {'dir': data_dir}})
            source_files = file_processor.get_split_files(target_split)
            
            if not source_files:
                self.logger.warning(f"Tidak ada file ditemukan untuk split {target_split}")
                return {}
            
            class_counts = self._extract_class_counts_from_files(source_files)
            needs = self._calculate_weighted_needs(class_counts, target_count)
            
            needy_classes = sum(1 for need in needs.values() if need > 0)
            total_needed = sum(needs.values())
            
            self.logger.info(f"💰 Class balancing {target_split}: {needy_classes}/{len(get_all_target_classes())} kelas butuh {total_needed} sampel")
            return needs
            
        except Exception as e:
            self.logger.error(f"Error calculating balancing needs: {str(e)}")
            return {}
    
    def calculate_balancing_needs(self, class_counts: Dict[str, int], target_count: int = 1000) -> Dict[str, int]:
        """Calculate needs dari existing class counts"""
        return self._calculate_weighted_needs(class_counts, target_count)
    
    def _calculate_weighted_needs(self, class_counts: Dict[str, int], target_count: int) -> Dict[str, int]:
        """Calculate needs dengan weighted priority"""
        needs = {}
        target_classes = get_all_target_classes()
        
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            weight = self.target_weights.get(cls, 1.0)
            weighted_target = int(target_count * weight)
            needs[cls] = calc_need(current, weighted_target)
        
        return needs
    
    def _extract_class_counts_from_files(self, source_files: List[str]) -> Dict[str, int]:
        """Extract class counts dengan preserved logic"""
        from pathlib import Path
        
        class_counts = defaultdict(int)
        
        for img_file in source_files:
            try:
                # Load labels menggunakan existing logic
                label_path = str(Path(img_file).parent.parent / 'labels' / f"{Path(img_file).stem}.txt")
                bboxes, class_labels = self._load_yolo_labels(label_path)
                
                for cls in class_labels:
                    class_counts[str(cls)] += 1
                    
            except Exception:
                continue
        
        return dict(class_counts)
    
    def _load_yolo_labels(self, label_path: str) -> tuple:
        """Load YOLO labels dengan preserved logic"""
        bboxes = []
        class_labels = []
        
        if not Path(label_path).exists():
            return bboxes, class_labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            
                            if all(0 <= x <= 1 for x in bbox) and bbox[2] > 0.001 and bbox[3] > 0.001:
                                bboxes.append(bbox)
                                class_labels.append(class_id)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.debug(f"Error loading labels {label_path}: {str(e)}")
        
        return bboxes, class_labels
    
    def get_balancing_priority_order(self, class_needs: Dict[str, int]) -> List[str]:
        """Get priority order dengan preserved logic"""
        needy_classes = {cls: need for cls, need in class_needs.items() if need > 0}
        
        priority_scores = {
            cls: need * calc_priority_multiplier(cls) 
            for cls, need in needy_classes.items()
        }
        
        sorted_classes = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        priority_order = [cls for cls, _ in sorted_classes]
        
        self.logger.info(f"🎯 Priority order: {' > '.join(priority_order[:5])}{'...' if len(priority_order) > 5 else ''}")
        return priority_order

