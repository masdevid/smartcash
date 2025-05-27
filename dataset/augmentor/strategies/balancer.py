"""
File: smartcash/dataset/augmentor/strategies/balancer.py
Deskripsi: Fixed balancer strategy menggunakan SRP utils modules dengan one-liner style
"""

from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter

# Updated imports dari SRP utils modules
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.file_operations import smart_find_images_split_aware
from smartcash.dataset.augmentor.utils.bbox_operations import load_yolo_labels

# One-liner constants untuk currency classes
get_layer1_classes = lambda: {'0', '1', '2', '3', '4', '5', '6'}
get_layer2_classes = lambda: {'7', '8', '9', '10', '11', '12', '13'}
get_all_target_classes = lambda: get_layer1_classes().union(get_layer2_classes())
is_target_class = lambda cls: cls in get_all_target_classes()

# One-liner calculators
calc_need = lambda current, target: max(0, target - current)
calc_priority_multiplier = lambda cls: 1.5 if cls in get_layer1_classes() else 1.2 if cls in get_layer2_classes() else 1.0
calc_deficit_ratio = lambda current, target: (target - current) / target if target > 0 else 0.0

class ClassBalancingStrategy:
    """Fixed balancer strategy menggunakan SRP utils modules"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        self.config = config or {}
        self.progress = create_progress_tracker(communicator)
        self.target_weights = self._setup_target_weights()
    
    def _setup_target_weights(self) -> Dict[str, float]:
        """Setup bobot target untuk setiap layer"""
        return {
            **{cls: 1.0 for cls in get_layer1_classes()},
            **{cls: 0.8 for cls in get_layer2_classes()},
        }
    
    def calculate_balancing_needs_split_aware(self, data_dir: str, target_split: str, target_count: int = 1000) -> Dict[str, int]:
        """Calculate needs menggunakan SRP file operations"""
        try:
            source_files = smart_find_images_split_aware(data_dir, target_split)
            if not source_files:
                self.progress.log_warning(f"Tidak ada file ditemukan untuk split {target_split}")
                return {}
            
            class_counts = self._extract_class_counts_from_files(source_files)
            needs = self._calculate_weighted_needs(class_counts, target_count)
            
            needy_classes = sum(1 for need in needs.values() if need > 0)
            total_needed = sum(needs.values())
            
            self.progress.log_info(f"💰 Class balancing {target_split}: {needy_classes}/{len(get_all_target_classes())} kelas butuh {total_needed} sampel")
            return needs
            
        except Exception as e:
            self.progress.log_error(f"Error calculating balancing needs: {str(e)}")
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
        """Extract class counts menggunakan SRP bbox operations"""
        from pathlib import Path
        
        class_counts = defaultdict(int)
        
        for img_file in source_files:
            try:
                label_path = str(Path(img_file).parent.parent / 'labels' / f"{Path(img_file).stem}.txt")
                bboxes, class_labels = load_yolo_labels(label_path)
                
                for cls in class_labels:
                    class_counts[str(cls)] += 1
                    
            except Exception:
                continue
        
        return dict(class_counts)
    
    def get_balancing_priority_order(self, class_needs: Dict[str, int]) -> List[str]:
        """Get priority order menggunakan weighted scoring"""
        needy_classes = {cls: need for cls, need in class_needs.items() if need > 0}
        
        priority_scores = {
            cls: need * calc_priority_multiplier(cls) 
            for cls, need in needy_classes.items()
        }
        
        sorted_classes = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        priority_order = [cls for cls, _ in sorted_classes]
        
        self.progress.log_info(f"🎯 Priority order: {' > '.join(priority_order[:5])}{'...' if len(priority_order) > 5 else ''}")
        return priority_order
    
    def is_balanced_enough(self, class_counts: Dict[str, int], target_count: int = 1000, threshold: float = 0.8) -> bool:
        """Check apakah dataset sudah cukup balance"""
        target_classes = get_all_target_classes()
        balanced_classes = 0
        
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            weight = self.target_weights.get(cls, 1.0)
            weighted_target = int(target_count * weight)
            
            if current >= weighted_target * threshold:
                balanced_classes += 1
        
        balance_ratio = balanced_classes / len(target_classes)
        is_balanced = balance_ratio >= threshold
        
        self.progress.log_info(f"⚖️ Balance status: {balanced_classes}/{len(target_classes)} kelas balanced ({balance_ratio:.1%})")
        return is_balanced
    
    def analyze_split_balance(self, data_dir: str, target_split: str) -> Dict[str, Any]:
        """Analyze balance untuk specific split"""
        try:
            source_files = smart_find_images_split_aware(data_dir, target_split)
            class_counts = self._extract_class_counts_from_files(source_files)
            
            total_samples = sum(class_counts.get(cls, 0) for cls in get_all_target_classes())
            layer1_samples = sum(class_counts.get(cls, 0) for cls in get_layer1_classes())
            layer2_samples = sum(class_counts.get(cls, 0) for cls in get_layer2_classes())
            
            class_values = [class_counts.get(cls, 0) for cls in get_all_target_classes()]
            avg_samples = sum(class_values) / len(class_values) if class_values else 0
            max_samples = max(class_values) if class_values else 0
            min_samples = min(class_values) if class_values else 0
            imbalance_ratio = max_samples / max(min_samples, 1)
            
            return {
                'target_split': target_split, 'total_samples': total_samples,
                'layer1_samples': layer1_samples, 'layer2_samples': layer2_samples,
                'class_distribution': dict(class_counts), 'avg_samples_per_class': avg_samples,
                'imbalance_ratio': imbalance_ratio,
                'balance_quality': 'excellent' if imbalance_ratio < 2 else 'good' if imbalance_ratio < 5 else 'poor'
            }
            
        except Exception as e:
            self.progress.log_error(f"Error analyzing split balance: {str(e)}")
            return {'target_split': target_split, 'error': str(e)}

class AdvancedBalancingStrategy(ClassBalancingStrategy):
    """Advanced balancer dengan adaptive thresholds"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        super().__init__(config, communicator)
        self.adaptive_thresholds = True
        self.multi_criteria_weights = {'deficit': 0.5, 'priority': 0.3, 'diversity': 0.2}
    
    def calculate_adaptive_targets_split_aware(self, data_dir: str, target_split: str, base_target: int = 1000) -> Dict[str, int]:
        """Calculate adaptive targets menggunakan SRP file operations"""
        try:
            source_files = smart_find_images_split_aware(data_dir, target_split)
            class_counts = self._extract_class_counts_from_files(source_files)
            
            return self._calculate_adaptive_targets_from_counts(class_counts, base_target)
            
        except Exception as e:
            self.progress.log_error(f"Error calculating adaptive targets: {str(e)}")
            return {cls: base_target for cls in get_all_target_classes()}
    
    def _calculate_adaptive_targets_from_counts(self, class_counts: Dict[str, int], base_target: int) -> Dict[str, int]:
        """Calculate adaptive targets dari class counts"""
        target_classes = get_all_target_classes()
        total_samples = sum(class_counts.get(cls, 0) for cls in target_classes)
        
        if total_samples == 0:
            return {cls: base_target for cls in target_classes}
        
        adaptive_targets = {}
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            proportion = current / total_samples if total_samples > 0 else 0
            weight = self.target_weights.get(cls, 1.0)
            
            adaptive_target = max(
                int(base_target * weight * 0.5),
                int(base_target * proportion * 1.5),
                current
            )
            
            adaptive_targets[cls] = adaptive_target
        
        avg_layer1 = sum(adaptive_targets[c] for c in get_layer1_classes()) // len(get_layer1_classes())
        self.progress.log_info(f"🔄 Adaptive targets: Layer1 avg={avg_layer1}")
        return adaptive_targets
    
    def calculate_multi_criteria_needs(self, class_counts: Dict[str, int], base_target: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Calculate needs berdasarkan multi-criteria"""
        adaptive_targets = self._calculate_adaptive_targets_from_counts(class_counts, base_target)
        target_classes = get_all_target_classes()
        
        multi_needs = {}
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            target = adaptive_targets[cls]
            
            deficit_score = calc_deficit_ratio(current, target)
            priority_score = calc_priority_multiplier(cls)
            diversity_score = 1.0 / (current + 1)
            
            combined_score = (
                deficit_score * self.multi_criteria_weights['deficit'] +
                priority_score * self.multi_criteria_weights['priority'] +
                diversity_score * self.multi_criteria_weights['diversity']
            )
            
            needed = calc_need(current, target)
            
            multi_needs[cls] = {
                'needed': needed, 'target': target, 'current': current,
                'deficit_score': deficit_score, 'priority_score': priority_score,
                'diversity_score': diversity_score, 'combined_score': combined_score
            }
        
        return multi_needs

# One-liner utilities menggunakan SRP modules
calculate_balancing_needs = lambda class_counts, target_count=1000, strategy='basic': (
    AdvancedBalancingStrategy().calculate_balancing_needs(class_counts, target_count) if strategy == 'advanced'
    else ClassBalancingStrategy().calculate_balancing_needs(class_counts, target_count)
)

calculate_split_balancing_needs = lambda data_dir, target_split, target_count=1000: ClassBalancingStrategy().calculate_balancing_needs_split_aware(data_dir, target_split, target_count)

get_target_classes = lambda layer='all': (
    get_layer1_classes() if layer == 'layer1' 
    else get_layer2_classes() if layer == 'layer2' 
    else get_all_target_classes()
)

analyze_class_distribution = lambda class_counts: {
    'total_samples': sum(class_counts.get(cls, 0) for cls in get_all_target_classes()),
    'layer1_samples': sum(class_counts.get(cls, 0) for cls in get_layer1_classes()),
    'layer2_samples': sum(class_counts.get(cls, 0) for cls in get_layer2_classes()),
    'imbalance_ratio': max([class_counts.get(cls, 0) for cls in get_all_target_classes()]) / max(min([class_counts.get(cls, 0) for cls in get_all_target_classes()]), 1),
    'distribution_quality': 'good' if max([class_counts.get(cls, 0) for cls in get_all_target_classes()]) / max(min([class_counts.get(cls, 0) for cls in get_all_target_classes()]), 1) < 3 else 'poor'
}

analyze_split_balance = lambda data_dir, target_split: ClassBalancingStrategy().analyze_split_balance(data_dir, target_split)