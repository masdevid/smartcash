"""
File: smartcash/dataset/augmentor/strategies/balancer.py
Deskripsi: Strategy untuk class balancing dengan fokus Layer 1 & 2 currency detection SmartCash
"""

from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter
from smartcash.common.logger import get_logger

# One-liner constants untuk SmartCash currency classes
get_layer1_classes = lambda: {'0', '1', '2', '3', '4', '5', '6'}  # Denominasi utama
get_layer2_classes = lambda: {'7', '8', '9', '10', '11', '12', '13'}  # Denominasi sekunder  
get_all_target_classes = lambda: get_layer1_classes().union(get_layer2_classes())
is_target_class = lambda cls: cls in get_all_target_classes()

# One-liner calculators
calc_need = lambda current, target: max(0, target - current)
calc_priority_multiplier = lambda cls: 1.5 if cls in get_layer1_classes() else 1.2 if cls in get_layer2_classes() else 1.0
calc_deficit_ratio = lambda current, target: (target - current) / target if target > 0 else 0.0

class ClassBalancingStrategy:
    """Strategy untuk class balancing dengan prioritas Layer 1 & 2 currency SmartCash"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger(__name__)
        self.target_weights = self._setup_target_weights()
    
    def _setup_target_weights(self) -> Dict[str, float]:
        """Setup bobot target untuk setiap layer currency"""
        return {
            **{cls: 1.0 for cls in get_layer1_classes()},  # Layer 1 priority tinggi
            **{cls: 0.8 for cls in get_layer2_classes()},  # Layer 2 priority medium
        }
    
    def calculate_balancing_needs(self, class_counts: Dict[str, int], target_count: int = 1000) -> Dict[str, int]:
        """Hitung kebutuhan balancing per kelas dengan weighted priority"""
        needs = {}
        target_classes = get_all_target_classes()
        
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            weight = self.target_weights.get(cls, 1.0)
            weighted_target = int(target_count * weight)
            needs[cls] = calc_need(current, weighted_target)
        
        # Log summary
        needy_classes = sum(1 for need in needs.values() if need > 0)
        total_needed = sum(needs.values())
        
        self.logger.info(f"💰 Class balancing: {needy_classes}/{len(target_classes)} kelas butuh total {total_needed} sampel")
        return needs
    
    def get_balancing_priority_order(self, class_needs: Dict[str, int]) -> List[str]:
        """Dapatkan urutan prioritas kelas untuk balancing"""
        # Filter hanya kelas yang butuh augmentasi
        needy_classes = {cls: need for cls, need in class_needs.items() if need > 0}
        
        # Sort berdasarkan kombinasi deficit dan priority multiplier
        priority_scores = {
            cls: need * calc_priority_multiplier(cls) 
            for cls, need in needy_classes.items()
        }
        
        sorted_classes = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        priority_order = [cls for cls, _ in sorted_classes]
        
        self.logger.info(f"🎯 Priority order: {' > '.join(priority_order[:5])}{'...' if len(priority_order) > 5 else ''}")
        return priority_order
    
    def is_balanced_enough(self, class_counts: Dict[str, int], target_count: int = 1000, threshold: float = 0.8) -> bool:
        """Cek apakah dataset sudah cukup balance"""
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
        
        self.logger.info(f"⚖️ Balance status: {balanced_classes}/{len(target_classes)} kelas balanced ({balance_ratio:.1%})")
        return is_balanced

class AdvancedBalancingStrategy(ClassBalancingStrategy):
    """Advanced balancing dengan adaptive thresholds dan multi-criteria optimization"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        super().__init__(config, logger)
        self.adaptive_thresholds = True
        self.multi_criteria_weights = {
            'deficit': 0.5,      # Bobot untuk deficit ratio
            'priority': 0.3,     # Bobot untuk class priority  
            'diversity': 0.2     # Bobot untuk diversity needs
        }
    
    def calculate_adaptive_targets(self, class_counts: Dict[str, int], base_target: int = 1000) -> Dict[str, int]:
        """Hitung target adaptif berdasarkan distribusi existing data"""
        target_classes = get_all_target_classes()
        total_samples = sum(class_counts.get(cls, 0) for cls in target_classes)
        
        if total_samples == 0:
            return {cls: base_target for cls in target_classes}
        
        # Adaptive targets berdasarkan proporsi existing + minimum thresholds
        adaptive_targets = {}
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            proportion = current / total_samples if total_samples > 0 else 0
            weight = self.target_weights.get(cls, 1.0)
            
            # Balanced antara proporsi existing dan target ideal
            adaptive_target = max(
                int(base_target * weight * 0.5),  # Minimum berdasarkan weight
                int(base_target * proportion * 1.5),  # Proportional scaling
                current  # Tidak kurangi yang sudah ada
            )
            
            adaptive_targets[cls] = adaptive_target
        
        self.logger.info(f"🔄 Adaptive targets: Layer1 avg={sum(adaptive_targets[c] for c in get_layer1_classes())//len(get_layer1_classes())}")
        return adaptive_targets
    
    def calculate_multi_criteria_needs(self, class_counts: Dict[str, int], base_target: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Hitung kebutuhan berdasarkan multi-criteria (deficit, priority, diversity)"""
        adaptive_targets = self.calculate_adaptive_targets(class_counts, base_target)
        target_classes = get_all_target_classes()
        
        multi_needs = {}
        for cls in target_classes:
            current = class_counts.get(cls, 0)
            target = adaptive_targets[cls]
            
            # Multi-criteria scores
            deficit_score = calc_deficit_ratio(current, target)
            priority_score = calc_priority_multiplier(cls)
            diversity_score = 1.0 / (current + 1)  # Inverse frequency untuk diversity
            
            # Weighted combined score
            combined_score = (
                deficit_score * self.multi_criteria_weights['deficit'] +
                priority_score * self.multi_criteria_weights['priority'] +
                diversity_score * self.multi_criteria_weights['diversity']
            )
            
            needed = calc_need(current, target)
            
            multi_needs[cls] = {
                'needed': needed,
                'target': target,
                'current': current,
                'deficit_score': deficit_score,
                'priority_score': priority_score,
                'diversity_score': diversity_score,
                'combined_score': combined_score
            }
        
        return multi_needs

# Utility functions untuk reusability
def calculate_balancing_needs(class_counts: Dict[str, int], target_count: int = 1000, 
                            strategy: str = 'basic') -> Dict[str, int]:
    """Utility function untuk menghitung kebutuhan balancing"""
    if strategy == 'advanced':
        balancer = AdvancedBalancingStrategy()
        return {cls: data['needed'] for cls, data in balancer.calculate_multi_criteria_needs(class_counts, target_count).items()}
    else:
        balancer = ClassBalancingStrategy()
        return balancer.calculate_balancing_needs(class_counts, target_count)

def get_target_classes(layer: str = 'all') -> Set[str]:
    """Utility function untuk mendapatkan target classes"""
    if layer == 'layer1':
        return get_layer1_classes()
    elif layer == 'layer2':
        return get_layer2_classes()
    else:
        return get_all_target_classes()

def analyze_class_distribution(class_counts: Dict[str, int]) -> Dict[str, Any]:
    """Analyze distribusi kelas untuk insights balancing"""
    target_classes = get_all_target_classes()
    
    # Basic stats
    total_samples = sum(class_counts.get(cls, 0) for cls in target_classes)
    layer1_samples = sum(class_counts.get(cls, 0) for cls in get_layer1_classes())
    layer2_samples = sum(class_counts.get(cls, 0) for cls in get_layer2_classes())
    
    # Distribution metrics
    class_values = [class_counts.get(cls, 0) for cls in target_classes]
    avg_samples = sum(class_values) / len(class_values) if class_values else 0
    max_samples = max(class_values) if class_values else 0
    min_samples = min(class_values) if class_values else 0
    
    # Imbalance ratio
    imbalance_ratio = max_samples / max(min_samples, 1)
    
    return {
        'total_samples': total_samples,
        'layer1_samples': layer1_samples,
        'layer2_samples': layer2_samples,
        'avg_samples_per_class': avg_samples,
        'max_samples': max_samples,
        'min_samples': min_samples,
        'imbalance_ratio': imbalance_ratio,
        'distribution_quality': 'good' if imbalance_ratio < 3 else 'poor' if imbalance_ratio > 10 else 'moderate'
    }