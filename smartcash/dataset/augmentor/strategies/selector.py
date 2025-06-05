"""
File: smartcash/dataset/augmentor/strategies/selector.py
Deskripsi: Fixed selector strategy menggunakan SRP utils modules dengan one-liner style
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import random

# Updated imports dari SRP utils modules
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.file_operations import smart_find_images_split_aware
from smartcash.dataset.augmentor.utils.bbox_operations import load_yolo_labels

# One-liner utilities untuk file selection
dedupe_files = lambda files: list(set(files))
sort_by_priority = lambda items, key_func: sorted(items, key=key_func, reverse=True)
filter_existing = lambda files, existing: [f for f in files if f not in existing]
shuffle_preserving_order = lambda files, preserve_ratio=0.7: files[:int(len(files)*preserve_ratio)] + random.sample(files[int(len(files)*preserve_ratio):], len(files) - int(len(files)*preserve_ratio))

class FileSelectionStrategy:
    """Updated file selector menggunakan SRP utils modules"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        self.config = config or {}
        self.progress = create_progress_tracker(communicator)
        self.processed_files: Set[str] = set()
        self.selection_metrics = defaultdict(int)
    
    def select_prioritized_files_split_aware(self, data_dir: str, target_split: str, 
                                           class_needs: Dict[str, int], 
                                           max_files_per_class: int = 500) -> List[str]:
        """Select files menggunakan SRP file operations"""
        try:
            source_files = smart_find_images_split_aware(data_dir, target_split)
            if not source_files:
                self.progress.log_warning(f"Tidak ada file ditemukan untuk selection pada split {target_split}")
                return []
            
            files_metadata = self._extract_files_metadata_for_selection(source_files)
            return self.select_prioritized_files(class_needs, files_metadata, max_files_per_class)
            
        except Exception as e:
            self.progress.log_error(f"Error selecting files for split {target_split}: {str(e)}")
            return []
    
    def select_prioritized_files(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]], 
                               max_files_per_class: int = 500) -> List[str]:
        """Select files dengan multi-class consideration"""
        self.progress.log_info(f"🎯 Memulai seleksi file prioritas untuk {len(class_needs)} kelas")
        
        self.selection_metrics.clear()
        files_by_class = self._group_files_by_primary_class(files_metadata)
        
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
        
        selected_files = self._optimize_final_selection(selected_files, files_metadata)
        
        self.progress.log_info(f"✅ Seleksi selesai: {len(selected_files)} file terpilih dari {len(files_metadata)} total")
        return selected_files
    
    def _extract_files_metadata_for_selection(self, source_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract metadata menggunakan SRP bbox operations"""
        files_metadata = {}
        
        for img_file in source_files:
            try:
                label_path = str(Path(img_file).parent.parent / 'labels' / f"{Path(img_file).stem}.txt")
                bboxes, class_labels = load_yolo_labels(label_path)
                
                class_counts = defaultdict(int)
                for cls in class_labels:
                    class_counts[str(cls)] += 1
                
                primary_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else '0'
                
                files_metadata[img_file] = {
                    'classes': set(map(str, class_labels)),
                    'class_counts': dict(class_counts),
                    'total_instances': len(bboxes),
                    'num_classes': len(set(class_labels)),
                    'primary_class': primary_class
                }
                
            except Exception:
                files_metadata[img_file] = {
                    'classes': set(), 'class_counts': {}, 'total_instances': 0,
                    'num_classes': 0, 'primary_class': '0'
                }
        
        return files_metadata
    
    def _group_files_by_primary_class(self, files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group files berdasarkan primary class"""
        files_by_class = defaultdict(list)
        
        for file_path, metadata in files_metadata.items():
            primary_class = metadata.get('primary_class')
            if primary_class:
                files_by_class[primary_class].append(file_path)
        
        return dict(files_by_class)
    
    def _select_files_for_class(self, target_class: str, needed: int, available_files: List[str],
                              files_metadata: Dict[str, Dict[str, Any]], max_files: int) -> List[str]:
        """Select files untuk kelas tertentu dengan scoring"""
        if not available_files:
            return []
        
        scored_files = []
        for file_path in available_files:
            if file_path in self.processed_files:
                continue
                
            metadata = files_metadata.get(file_path, {})
            score = self._calculate_file_score(file_path, target_class, metadata)
            scored_files.append((file_path, score))
        
        scored_files = sort_by_priority(scored_files, lambda x: x[1])
        max_select = min(needed, max_files, len(scored_files))
        selected = [file_path for file_path, _ in scored_files[:max_select]]
        
        self.processed_files.update(selected)
        
        self.progress.log_info(f"📁 Kelas {target_class}: dipilih {len(selected)}/{len(available_files)} file")
        return selected
    
    def _calculate_file_score(self, file_path: str, target_class: str, metadata: Dict[str, Any]) -> float:
        """Calculate score file untuk target class"""
        score = 0.0
        
        class_counts = metadata.get('class_counts', {})
        target_instances = class_counts.get(target_class, 0)
        score += target_instances * 10.0
        
        num_classes = metadata.get('num_classes', 1)
        score += (num_classes - 1) * 2.0
        
        total_instances = metadata.get('total_instances', 1)
        score += min(total_instances, 10) * 1.0
        
        file_name = Path(file_path).stem
        similar_processed = sum(1 for p in self.processed_files if Path(p).stem.startswith(file_name[:5]))
        score -= similar_processed * 5.0
        
        return max(score, 0.1)
    
    def _optimize_final_selection(self, selected_files: List[str], files_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Optimize final selection untuk balance dan diversity"""
        unique_files = dedupe_files(selected_files)
        class_distribution = self._analyze_selection_distribution(unique_files, files_metadata)
        
        if len(unique_files) > 100:
            optimized_files = self._apply_diversity_optimization(unique_files, files_metadata, class_distribution)
        else:
            optimized_files = unique_files
        
        self.progress.log_info(f"🔄 Optimisasi: {len(selected_files)} → {len(unique_files)} → {len(optimized_files)} file")
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
        avg_count = sum(class_distribution.values()) / len(class_distribution) if class_distribution else 0
        over_represented = {cls: count for cls, count in class_distribution.items() if count > avg_count * 2}
        
        if not over_represented:
            return files
        
        optimized_files = []
        removed_count = 0
        
        for file_path in files:
            metadata = files_metadata.get(file_path, {})
            primary_class = metadata.get('primary_class')
            
            if primary_class not in over_represented or random.random() < 0.7:
                optimized_files.append(file_path)
            else:
                removed_count += 1
        
        self.progress.log_info(f"🎲 Diversity optimization: removed {removed_count} files")
        return optimized_files
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get ringkasan proses selection"""
        total_classes = len([k for k in self.selection_metrics.keys() if k.startswith('class_')])
        quality = 'excellent' if total_classes >= 10 else 'good' if total_classes >= 7 else 'moderate' if total_classes >= 4 else 'poor'
        
        return {
            'total_processed': len(self.processed_files),
            'class_metrics': dict(self.selection_metrics),
            'selection_quality': quality
        }

class SmartFileSelector(FileSelectionStrategy):
    """Advanced file selector dengan adaptive learning"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        super().__init__(config, communicator)
        self.learning_factor = 0.1
        self.file_scores_history = defaultdict(list)
    
    def select_with_adaptive_learning(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]],
                                    previous_results: List[Dict[str, Any]] = None) -> List[str]:
        """Selection dengan adaptive learning"""
        if previous_results:
            self._update_scoring_from_results(previous_results)
        
        return self._select_with_scoring(class_needs, files_metadata)
    
    def _update_scoring_from_results(self, results: List[Dict[str, Any]]) -> None:
        """Update scoring weights berdasarkan hasil"""
        for result in results:
            if result.get('status') == 'success':
                file_path = result.get('source_file', '')
                success_score = result.get('generated', 0)
                
                self.file_scores_history[file_path].append(success_score)
                
                if len(self.file_scores_history[file_path]) > 5:
                    self.file_scores_history[file_path] = self.file_scores_history[file_path][-5:]
    
    def _select_with_scoring(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Selection dengan enhanced scoring"""
        enhanced_metadata = {}
        
        for file_path, metadata in files_metadata.items():
            enhanced_meta = dict(metadata)
            
            historical_scores = self.file_scores_history.get(file_path, [])
            avg_historical_score = sum(historical_scores) / len(historical_scores) if historical_scores else 1.0
            enhanced_meta['historical_score'] = avg_historical_score
            
            enhanced_metadata[file_path] = enhanced_meta
        
        return self.select_prioritized_files(class_needs, enhanced_metadata)
    
    def _calculate_file_score(self, file_path: str, target_class: str, metadata: Dict[str, Any]) -> float:
        """Enhanced scoring dengan historical data"""
        base_score = super()._calculate_file_score(file_path, target_class, metadata)
        
        historical_score = metadata.get('historical_score', 1.0)
        enhanced_score = base_score * (1.0 + self.learning_factor * (historical_score - 1.0))
        
        return enhanced_score

# One-liner utilities menggunakan SRP modules
select_prioritized_files = lambda class_needs, files_metadata, strategy='basic', **kwargs: (
    SmartFileSelector().select_with_adaptive_learning(class_needs, files_metadata, kwargs.get('previous_results')) if strategy == 'smart'
    else FileSelectionStrategy().select_prioritized_files(class_needs, files_metadata)
)

select_split_files = lambda data_dir, target_split, class_needs, max_files=500: FileSelectionStrategy().select_prioritized_files_split_aware(data_dir, target_split, class_needs, max_files)

dedupe_and_sort_files = lambda files, files_metadata=None, sort_by='name': (
    sorted(dedupe_files(files), key=lambda f: files_metadata.get(f, {}).get('total_instances', 0), reverse=True) if sort_by == 'priority' and files_metadata
    else sorted(dedupe_files(files), key=lambda f: files_metadata.get(f, {}).get('num_classes', 0), reverse=True) if sort_by == 'diversity' and files_metadata
    else sorted(dedupe_files(files))
)

analyze_selection_quality = lambda selected_files, files_metadata: {
    'quality': 'excellent' if len(selected_files) > 50 and len(set().union(*[files_metadata.get(f, {}).get('classes', set()) for f in selected_files])) > 8 else 'good',
    'total_files': len(selected_files),
    'total_classes': len(set().union(*[files_metadata.get(f, {}).get('classes', set()) for f in selected_files])),
    'avg_instances': sum(files_metadata.get(f, {}).get('total_instances', 0) for f in selected_files) / len(selected_files) if selected_files else 0
} if selected_files and files_metadata else {'quality': 'no_data'}