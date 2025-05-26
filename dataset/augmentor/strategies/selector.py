"""
File: smartcash/dataset/augmentor/strategies/selector.py
Deskripsi: Strategy untuk file selection dengan prioritas multi-class dan diversity optimization
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import random
from smartcash.common.logger import get_logger

# One-liner utilities untuk file selection
dedupe_files = lambda files: list(set(files))
sort_by_priority = lambda items, key_func: sorted(items, key=key_func, reverse=True)
filter_existing = lambda files, existing: [f for f in files if f not in existing]
shuffle_preserving_order = lambda files, preserve_ratio=0.7: files[:int(len(files)*preserve_ratio)] + random.sample(files[int(len(files)*preserve_ratio):], len(files) - int(len(files)*preserve_ratio))

class FileSelectionStrategy:
    """Strategy untuk pemilihan file optimal dengan multi-class consideration"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger(__name__)
        self.processed_files: Set[str] = set()
        self.selection_metrics = defaultdict(int)
    
    def select_prioritized_files(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]], 
                               max_files_per_class: int = 500) -> List[str]:
        """Pilih file dengan prioritas berdasarkan kebutuhan kelas dan metadata"""
        self.logger.info(f"🎯 Memulai seleksi file prioritas untuk {len(class_needs)} kelas")
        
        # Reset metrics
        self.selection_metrics.clear()
        
        # Group files by primary class
        files_by_class = self._group_files_by_class(files_metadata)
        
        # Select files per class based on needs
        selected_files = []
        for cls, needed in sorted(class_needs.items(), key=lambda x: x[1], reverse=True):
            if needed <= 0:
                continue
                
            class_files = self._select_files_for_class(
                cls, needed, files_by_class.get(cls, []), 
                files_metadata, max_files_per_class
            )
            
            selected_files.extend(class_files)
            self.selection_metrics[f'class_{cls}'] = len(class_files)
        
        # Dedupe dan optimize selection
        selected_files = self._optimize_final_selection(selected_files, files_metadata)
        
        self.logger.info(f"✅ Seleksi selesai: {len(selected_files)} file terpilih dari {len(files_metadata)} total")
        return selected_files
    
    def _group_files_by_class(self, files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group files berdasarkan primary class"""
        files_by_class = defaultdict(list)
        
        for file_path, metadata in files_metadata.items():
            primary_class = metadata.get('primary_class')
            if primary_class:
                files_by_class[primary_class].append(file_path)
        
        return dict(files_by_class)
    
    def _select_files_for_class(self, target_class: str, needed: int, available_files: List[str],
                              files_metadata: Dict[str, Dict[str, Any]], max_files: int) -> List[str]:
        """Pilih file untuk kelas tertentu dengan scoring"""
        if not available_files:
            return []
        
        # Score files berdasarkan utilitas untuk target class
        scored_files = []
        for file_path in available_files:
            if file_path in self.processed_files:
                continue
                
            metadata = files_metadata.get(file_path, {})
            score = self._calculate_file_score(file_path, target_class, metadata)
            scored_files.append((file_path, score))
        
        # Sort dan ambil top files
        scored_files = sort_by_priority(scored_files, lambda x: x[1])
        
        # Limit selection
        max_select = min(needed, max_files, len(scored_files))
        selected = [file_path for file_path, _ in scored_files[:max_select]]
        
        # Add to processed set
        self.processed_files.update(selected)
        
        self.logger.info(f"📁 Kelas {target_class}: dipilih {len(selected)}/{len(available_files)} file (butuh {needed})")
        return selected
    
    def _calculate_file_score(self, file_path: str, target_class: str, metadata: Dict[str, Any]) -> float:
        """Hitung score file untuk target class"""
        score = 0.0
        
        # Base score dari target class count
        class_counts = metadata.get('class_counts', {})
        target_instances = class_counts.get(target_class, 0)
        score += target_instances * 10.0  # High weight untuk target class
        
        # Bonus untuk multi-class files (diversity)
        num_classes = metadata.get('num_classes', 1)
        score += (num_classes - 1) * 2.0  # Bonus untuk diversity
        
        # Bonus untuk total instances (information density)
        total_instances = metadata.get('total_instances', 1)
        score += min(total_instances, 10) * 1.0  # Cap at 10 untuk avoid outliers
        
        # Penalty untuk already processed similar files (diversity)
        file_name = Path(file_path).stem
        similar_processed = sum(1 for p in self.processed_files if Path(p).stem.startswith(file_name[:5]))
        score -= similar_processed * 5.0
        
        return max(score, 0.1)  # Minimum score
    
    def _optimize_final_selection(self, selected_files: List[str], files_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Optimize final selection untuk balance dan diversity"""
        # Deduplicate
        unique_files = dedupe_files(selected_files)
        
        # Analyze selection untuk optimization
        class_distribution = self._analyze_selection_distribution(unique_files, files_metadata)
        
        # Apply diversity optimization jika needed
        if len(unique_files) > 100:  # Only for large selections
            optimized_files = self._apply_diversity_optimization(unique_files, files_metadata, class_distribution)
        else:
            optimized_files = unique_files
        
        self.logger.info(f"🔄 Optimisasi: {len(selected_files)} → {len(unique_files)} → {len(optimized_files)} file")
        return optimized_files
    
    def _analyze_selection_distribution(self, files: List[str], files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribusi kelas dalam selection"""
        class_counts = Counter()
        
        for file_path in files:
            metadata = files_metadata.get(file_path, {})
            for cls, count in metadata.get('class_counts', {}).items():
                class_counts[cls] += count
        
        return dict(class_counts)
    
    def _apply_diversity_optimization(self, files: List[str], files_metadata: Dict[str, Dict[str, Any]], 
                                   class_distribution: Dict[str, int]) -> List[str]:
        """Apply diversity optimization untuk prevent over-selection"""
        # Find over-represented classes
        avg_count = sum(class_distribution.values()) / len(class_distribution) if class_distribution else 0
        over_represented = {cls: count for cls, count in class_distribution.items() if count > avg_count * 2}
        
        if not over_represented:
            return files
        
        # Remove some files from over-represented classes
        optimized_files = []
        removed_count = 0
        
        for file_path in files:
            metadata = files_metadata.get(file_path, {})
            primary_class = metadata.get('primary_class')
            
            # Keep file jika tidak over-represented atau random selection
            if primary_class not in over_represented or random.random() < 0.7:
                optimized_files.append(file_path)
            else:
                removed_count += 1
        
        self.logger.info(f"🎲 Diversity optimization: removed {removed_count} files dari over-represented classes")
        return optimized_files
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Dapatkan ringkasan proses selection"""
        return {
            'total_processed': len(self.processed_files),
            'class_metrics': dict(self.selection_metrics),
            'selection_quality': self._assess_selection_quality()
        }
    
    def _assess_selection_quality(self) -> str:
        """Assess kualitas selection berdasarkan metrics"""
        total_classes = len([k for k in self.selection_metrics.keys() if k.startswith('class_')])
        
        if total_classes >= 10:
            return 'excellent'
        elif total_classes >= 7:
            return 'good'
        elif total_classes >= 4:
            return 'moderate'
        else:
            return 'poor'

class SmartFileSelector(FileSelectionStrategy):
    """Advanced file selector dengan machine learning inspired heuristics"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        super().__init__(config, logger)
        self.learning_factor = 0.1  # Learning rate untuk adaptive scoring
        self.file_scores_history = defaultdict(list)
    
    def select_with_adaptive_learning(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]],
                                    previous_results: List[Dict[str, Any]] = None) -> List[str]:
        """Selection dengan adaptive learning dari hasil sebelumnya"""
        # Update scoring berdasarkan previous results
        if previous_results:
            self._update_scoring_from_results(previous_results)
        
        # Use enhanced scoring
        return self._select_with_enhanced_scoring(class_needs, files_metadata)
    
    def _update_scoring_from_results(self, results: List[Dict[str, Any]]) -> None:
        """Update scoring weights berdasarkan hasil augmentasi sebelumnya"""
        for result in results:
            if result.get('status') == 'success':
                file_path = result.get('source_file', '')
                success_score = result.get('generated', 0)
                
                # Update score history
                self.file_scores_history[file_path].append(success_score)
                
                # Keep only recent scores
                if len(self.file_scores_history[file_path]) > 5:
                    self.file_scores_history[file_path] = self.file_scores_history[file_path][-5:]
    
    def _select_with_enhanced_scoring(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Selection dengan enhanced scoring yang consider history"""
        enhanced_metadata = {}
        
        # Enhance metadata dengan historical performance
        for file_path, metadata in files_metadata.items():
            enhanced_meta = dict(metadata)
            
            # Add historical score
            historical_scores = self.file_scores_history.get(file_path, [])
            avg_historical_score = sum(historical_scores) / len(historical_scores) if historical_scores else 1.0
            enhanced_meta['historical_score'] = avg_historical_score
            
            enhanced_metadata[file_path] = enhanced_meta
        
        # Use parent selection dengan enhanced metadata
        return self.select_prioritized_files(class_needs, enhanced_metadata)
    
    def _calculate_file_score(self, file_path: str, target_class: str, metadata: Dict[str, Any]) -> float:
        """Enhanced scoring dengan historical data"""
        base_score = super()._calculate_file_score(file_path, target_class, metadata)
        
        # Add historical performance bonus
        historical_score = metadata.get('historical_score', 1.0)
        enhanced_score = base_score * (1.0 + self.learning_factor * (historical_score - 1.0))
        
        return enhanced_score

# Utility functions untuk external usage
def select_prioritized_files(class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]],
                           strategy: str = 'basic', **kwargs) -> List[str]:
    """Utility function untuk file selection"""
    if strategy == 'smart':
        selector = SmartFileSelector()
        return selector.select_with_adaptive_learning(class_needs, files_metadata, kwargs.get('previous_results'))
    else:
        selector = FileSelectionStrategy()
        return selector.select_prioritized_files(class_needs, files_metadata)

def dedupe_and_sort_files(files: List[str], files_metadata: Dict[str, Dict[str, Any]] = None, 
                         sort_by: str = 'name') -> List[str]:
    """Utility untuk dedupe dan sort files"""
    unique_files = dedupe_files(files)
    
    if sort_by == 'priority' and files_metadata:
        # Sort by total instances (information density)
        return sorted(unique_files, key=lambda f: files_metadata.get(f, {}).get('total_instances', 0), reverse=True)
    elif sort_by == 'diversity' and files_metadata:
        # Sort by class diversity
        return sorted(unique_files, key=lambda f: files_metadata.get(f, {}).get('num_classes', 0), reverse=True)
    else:
        # Sort by name (default)
        return sorted(unique_files)

def analyze_file_selection_quality(selected_files: List[str], files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze kualitas file selection"""
    if not selected_files or not files_metadata:
        return {'quality': 'no_data', 'metrics': {}}
    
    # Calculate metrics
    total_instances = sum(files_metadata.get(f, {}).get('total_instances', 0) for f in selected_files)
    total_classes = len(set().union(*[files_metadata.get(f, {}).get('classes', set()) for f in selected_files]))
    avg_instances_per_file = total_instances / len(selected_files) if selected_files else 0
    
    # Diversity metrics
    class_distribution = Counter()
    for file_path in selected_files:
        metadata = files_metadata.get(file_path, {})
        for cls in metadata.get('classes', set()):
            class_distribution[cls] += 1
    
    # Quality assessment
    diversity_score = len(class_distribution) / max(len(selected_files), 1)
    density_score = min(avg_instances_per_file / 10.0, 1.0)  # Normalize to 0-1
    balance_score = 1.0 - (max(class_distribution.values()) - min(class_distribution.values())) / max(max(class_distribution.values()), 1)
    
    overall_quality = (diversity_score * 0.4 + density_score * 0.3 + balance_score * 0.3)
    
    return {
        'quality': 'excellent' if overall_quality > 0.8 else 'good' if overall_quality > 0.6 else 'moderate' if overall_quality > 0.4 else 'poor',
        'metrics': {
            'total_files': len(selected_files),
            'total_instances': total_instances,
            'total_classes': total_classes,
            'avg_instances_per_file': avg_instances_per_file,
            'diversity_score': diversity_score,
            'density_score': density_score,
            'balance_score': balance_score,
            'overall_score': overall_quality
        }
    }