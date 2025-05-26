"""
File: smartcash/dataset/augmentor/strategies/priority.py
Deskripsi: Strategy untuk prioritization calculation dan ranking system untuk augmentasi files
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import math
from pathlib import Path
from smartcash.common.logger import get_logger

# One-liner priority calculators
calc_deficit_priority = lambda current, target: (target - current) / target if target > 0 else 0
calc_density_priority = lambda instances, max_instances: min(instances / max_instances, 1.0) if max_instances > 0 else 0
calc_diversity_priority = lambda num_classes, max_classes: min(num_classes / max_classes, 1.0) if max_classes > 0 else 0
calc_rarity_priority = lambda frequency, max_frequency: 1.0 - (frequency / max_frequency) if max_frequency > 0 else 0

# Currency-specific priority weights
get_currency_priority_weight = lambda cls: 2.0 if cls in {'0', '1', '2', '3', '4', '5', '6'} else 1.5 if cls in {'7', '8', '9', '10', '11', '12', '13'} else 1.0

class PriorityCalculator:
    """Calculator untuk menentukan prioritas augmentasi berdasarkan multiple criteria"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger(__name__)
        self.priority_weights = self._setup_priority_weights()
        self.normalization_cache = {}
    
    def _setup_priority_weights(self) -> Dict[str, float]:
        """Setup bobot untuk berbagai faktor prioritas"""
        return self.config.get('priority_weights', {
            'deficit': 0.4,        # Kekurangan sampel
            'density': 0.2,        # Kepadatan informasi
            'diversity': 0.2,      # Keragaman kelas
            'rarity': 0.1,         # Kelangkaan file
            'currency_type': 0.1   # Tipe mata uang (Layer 1/2)
        })
    
    def calculate_augmentation_priority(self, files_metadata: Dict[str, Dict[str, Any]], 
                                     class_needs: Dict[str, int],
                                     current_class_counts: Dict[str, int]) -> Dict[str, float]:
        """Hitung prioritas augmentasi untuk setiap file"""
        self.logger.info(f"🎯 Menghitung prioritas untuk {len(files_metadata)} file")
        
        # Calculate normalization factors
        norm_factors = self._calculate_normalization_factors(files_metadata, class_needs)
        
        # Calculate priority for each file
        file_priorities = {}
        for file_path, metadata in files_metadata.items():
            priority_score = self._calculate_file_priority(
                file_path, metadata, class_needs, current_class_counts, norm_factors
            )
            file_priorities[file_path] = priority_score
        
        self.logger.info(f"✅ Prioritas dihitung: avg={sum(file_priorities.values())/len(file_priorities):.3f}")
        return file_priorities
    
    def _calculate_normalization_factors(self, files_metadata: Dict[str, Dict[str, Any]], 
                                       class_needs: Dict[str, int]) -> Dict[str, float]:
        """Hitung faktor normalisasi untuk scoring"""
        if 'norm_factors' in self.normalization_cache:
            return self.normalization_cache['norm_factors']
        
        # Extract values untuk normalisasi
        all_instances = [meta.get('total_instances', 0) for meta in files_metadata.values()]
        all_classes = [meta.get('num_classes', 0) for meta in files_metadata.values()]
        all_needs = list(class_needs.values())
        
        # Calculate max values untuk normalisasi
        factors = {
            'max_instances': max(all_instances) if all_instances else 1,
            'max_classes': max(all_classes) if all_classes else 1,
            'max_need': max(all_needs) if all_needs else 1,
            'file_frequency': defaultdict(int)  # Will be populated per file
        }
        
        # File frequency untuk rarity calculation
        for file_path in files_metadata.keys():
            file_prefix = Path(file_path).stem[:5]  # First 5 chars as similarity key
            factors['file_frequency'][file_prefix] += 1
        
        factors['max_frequency'] = max(factors['file_frequency'].values()) if factors['file_frequency'] else 1
        
        self.normalization_cache['norm_factors'] = factors
        return factors
    
    def _calculate_file_priority(self, file_path: str, metadata: Dict[str, Any],
                               class_needs: Dict[str, int], current_class_counts: Dict[str, int],
                               norm_factors: Dict[str, Any]) -> float:
        """Hitung prioritas untuk single file"""
        # Extract metadata
        classes = metadata.get('classes', set())
        class_counts = metadata.get('class_counts', {})
        total_instances = metadata.get('total_instances', 1)
        num_classes = metadata.get('num_classes', 1)
        primary_class = metadata.get('primary_class', '')
        
        # Calculate component scores
        deficit_score = self._calc_deficit_score(classes, class_needs, current_class_counts, norm_factors)
        density_score = calc_density_priority(total_instances, norm_factors['max_instances'])
        diversity_score = calc_diversity_priority(num_classes, norm_factors['max_classes'])
        rarity_score = self._calc_rarity_score(file_path, norm_factors)
        currency_score = self._calc_currency_type_score(primary_class)
        
        # Weighted combination
        priority_score = (
            deficit_score * self.priority_weights['deficit'] +
            density_score * self.priority_weights['density'] +
            diversity_score * self.priority_weights['diversity'] +
            rarity_score * self.priority_weights['rarity'] +
            currency_score * self.priority_weights['currency_type']
        )
        
        return max(priority_score, 0.01)  # Minimum priority
    
    def _calc_deficit_score(self, classes: set, class_needs: Dict[str, int], 
                          current_counts: Dict[str, int], norm_factors: Dict[str, Any]) -> float:
        """Hitung score berdasarkan deficit kelas"""
        if not classes:
            return 0.0
        
        # Calculate weighted deficit untuk semua kelas dalam file
        total_deficit = 0.0
        total_weight = 0.0
        
        for cls in classes:
            need = class_needs.get(cls, 0)
            current = current_counts.get(cls, 0)
            
            if need > 0:  # Only consider classes yang butuh augmentasi
                deficit_ratio = calc_deficit_priority(current, current + need)
                currency_weight = get_currency_priority_weight(cls)
                
                total_deficit += deficit_ratio * currency_weight
                total_weight += currency_weight
        
        return total_deficit / total_weight if total_weight > 0 else 0.0
    
    def _calc_rarity_score(self, file_path: str, norm_factors: Dict[str, Any]) -> float:
        """Hitung score berdasarkan rarity file"""
        file_prefix = Path(file_path).stem[:5]
        frequency = norm_factors['file_frequency'].get(file_prefix, 1)
        return calc_rarity_priority(frequency, norm_factors['max_frequency'])
    
    def _calc_currency_type_score(self, primary_class: str) -> float:
        """Hitung score berdasarkan tipe currency"""
        if not primary_class:
            return 0.5  # Neutral score
        
        weight = get_currency_priority_weight(primary_class)
        return min(weight / 2.0, 1.0)  # Normalize ke 0-1
    
    def rank_files_by_priority(self, file_priorities: Dict[str, float], 
                             top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Rank files berdasarkan priority score"""
        sorted_files = sorted(file_priorities.items(), key=lambda x: x[1], reverse=True)
        
        if top_k:
            sorted_files = sorted_files[:top_k]
        
        self.logger.info(f"📊 Ranking: top file priority={sorted_files[0][1]:.3f}, bottom={sorted_files[-1][1]:.3f}")
        return sorted_files
    
    def get_priority_distribution_analysis(self, file_priorities: Dict[str, float]) -> Dict[str, Any]:
        """Analyze distribusi prioritas untuk insights"""
        if not file_priorities:
            return {'status': 'no_data'}
        
        priorities = list(file_priorities.values())
        
        # Statistical measures
        mean_priority = sum(priorities) / len(priorities)
        sorted_priorities = sorted(priorities, reverse=True)
        median_priority = sorted_priorities[len(sorted_priorities)//2]
        
        # Distribution quartiles
        q1_idx, q3_idx = len(sorted_priorities)//4, 3*len(sorted_priorities)//4
        q1_priority = sorted_priorities[q1_idx] if q1_idx < len(sorted_priorities) else sorted_priorities[-1]
        q3_priority = sorted_priorities[q3_idx] if q3_idx < len(sorted_priorities) else sorted_priorities[0]
        
        # Priority categories
        high_priority = sum(1 for p in priorities if p > mean_priority + 0.2)
        medium_priority = sum(1 for p in priorities if mean_priority - 0.1 <= p <= mean_priority + 0.2)
        low_priority = len(priorities) - high_priority - medium_priority
        
        return {
            'total_files': len(priorities),
            'mean_priority': mean_priority,
            'median_priority': median_priority,
            'max_priority': max(priorities),
            'min_priority': min(priorities),
            'q1_priority': q1_priority,
            'q3_priority': q3_priority,
            'high_priority_files': high_priority,
            'medium_priority_files': medium_priority,
            'low_priority_files': low_priority,
            'distribution_quality': self._assess_distribution_quality(priorities)
        }
    
    def _assess_distribution_quality(self, priorities: List[float]) -> str:
        """Assess kualitas distribusi prioritas"""
        if not priorities:
            return 'no_data'
        
        # Calculate coefficient of variation
        mean_p = sum(priorities) / len(priorities)
        variance = sum((p - mean_p)**2 for p in priorities) / len(priorities)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_p if mean_p > 0 else 0
        
        # Assess berdasarkan variability
        if cv > 0.5:
            return 'excellent'  # High variability = good discrimination
        elif cv > 0.3:
            return 'good'
        elif cv > 0.1:
            return 'moderate'
        else:
            return 'poor'  # Low variability = poor discrimination

class AdaptivePriorityCalculator(PriorityCalculator):
    """Advanced priority calculator dengan adaptive learning"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        super().__init__(config, logger)
        self.learning_rate = 0.1
        self.success_history = defaultdict(list)
        self.adaptive_weights = dict(self.priority_weights)
    
    def update_from_results(self, augmentation_results: List[Dict[str, Any]]) -> None:
        """Update priority calculation berdasarkan hasil augmentasi"""
        successful_files = []
        failed_files = []
        
        for result in augmentation_results:
            source_file = result.get('source_file', '')
            if result.get('status') == 'success':
                successful_files.append((source_file, result.get('generated', 0)))
            else:
                failed_files.append(source_file)
        
        # Analyze correlation antara priority dan success
        self._analyze_priority_success_correlation(successful_files, failed_files)
        
        # Update adaptive weights
        self._update_adaptive_weights(successful_files, failed_files)
    
    def _analyze_priority_success_correlation(self, successful_files: List[Tuple[str, int]], 
                                            failed_files: List[str]) -> None:
        """Analyze korelasi antarra priority score dan success rate"""
        if not hasattr(self, '_last_priorities'):
            return
        
        success_priorities = [self._last_priorities.get(f, 0) for f, _ in successful_files]
        failure_priorities = [self._last_priorities.get(f, 0) for f in failed_files]
        
        avg_success_priority = sum(success_priorities) / len(success_priorities) if success_priorities else 0
        avg_failure_priority = sum(failure_priorities) / len(failure_priorities) if failure_priorities else 0
        
        correlation_strength = abs(avg_success_priority - avg_failure_priority)
        
        self.logger.info(f"📈 Priority-Success correlation: {correlation_strength:.3f} (success={avg_success_priority:.3f}, fail={avg_failure_priority:.3f})")
    
    def _update_adaptive_weights(self, successful_files: List[Tuple[str, int]], failed_files: List[str]) -> None:
        """Update bobot prioritas berdasarkan performance"""
        if len(successful_files) < 5:  # Need minimum samples
            return
        
        # Simple adaptive adjustment
        success_rate = len(successful_files) / (len(successful_files) + len(failed_files))
        
        if success_rate > 0.8:  # High success rate - maintain weights
            return
        elif success_rate < 0.5:  # Low success rate - adjust weights
            # Increase deficit weight, decrease others
            self.adaptive_weights['deficit'] = min(0.6, self.adaptive_weights['deficit'] + self.learning_rate)
            self.adaptive_weights['diversity'] = max(0.1, self.adaptive_weights['diversity'] - self.learning_rate/2)
            
            self.logger.info(f"🔄 Adaptive weights updated: deficit={self.adaptive_weights['deficit']:.2f}")
    
    def calculate_augmentation_priority(self, files_metadata: Dict[str, Dict[str, Any]], 
                                     class_needs: Dict[str, int],
                                     current_class_counts: Dict[str, int]) -> Dict[str, float]:
        """Override dengan adaptive weights"""
        # Temporarily use adaptive weights
        original_weights = self.priority_weights
        self.priority_weights = self.adaptive_weights
        
        # Calculate dengan adaptive weights
        priorities = super().calculate_augmentation_priority(files_metadata, class_needs, current_class_counts)
        
        # Store untuk correlation analysis
        self._last_priorities = priorities
        
        # Restore original weights
        self.priority_weights = original_weights
        
        return priorities

# Utility functions untuk external usage
def calculate_augmentation_priority(files_metadata: Dict[str, Dict[str, Any]], 
                                  class_needs: Dict[str, int],
                                  current_class_counts: Dict[str, int],
                                  strategy: str = 'basic') -> Dict[str, float]:
    """Utility function untuk menghitung priority"""
    if strategy == 'adaptive':
        calculator = AdaptivePriorityCalculator()
    else:
        calculator = PriorityCalculator()
    
    return calculator.calculate_augmentation_priority(files_metadata, class_needs, current_class_counts)

def rank_files_by_priority(file_priorities: Dict[str, float], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Utility function untuk ranking files"""
    calculator = PriorityCalculator()
    return calculator.rank_files_by_priority(file_priorities, top_k)

def get_top_priority_files(files_metadata: Dict[str, Dict[str, Any]], class_needs: Dict[str, int],
                          current_class_counts: Dict[str, int], top_k: int = 100) -> List[str]:
    """Get top-k files dengan prioritas tertinggi"""
    priorities = calculate_augmentation_priority(files_metadata, class_needs, current_class_counts)
    ranked_files = rank_files_by_priority(priorities, top_k)
    return [file_path for file_path, _ in ranked_files]

def analyze_priority_clusters(file_priorities: Dict[str, float], 
                            files_metadata: Dict[str, Dict[str, Any]], 
                            num_clusters: int = 3) -> Dict[str, List[str]]:
    """Cluster files berdasarkan priority score"""
    if not file_priorities:
        return {}
    
    # Sort files by priority
    sorted_files = sorted(file_priorities.items(), key=lambda x: x[1], reverse=True)
    
    # Simple clustering by priority ranges
    cluster_size = len(sorted_files) // num_clusters
    clusters = {}
    
    for i in range(num_clusters):
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size if i < num_clusters - 1 else len(sorted_files)
        
        cluster_name = f'high_priority' if i == 0 else f'medium_priority' if i == 1 else 'low_priority'
        clusters[cluster_name] = [file_path for file_path, _ in sorted_files[start_idx:end_idx]]
    
    return clusters