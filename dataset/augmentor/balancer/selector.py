"""
File: smartcash/dataset/augmentor/balancer/selector.py
Deskripsi: File selector dengan preserved logic dan missing imports
"""

from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter
import random
from pathlib import Path
from smartcash.common.logger import get_logger

class FileSelectionStrategy:
    """📁 File selector dengan preserved business logic"""
    
    def __init__(self, config: Dict[str, Any] = None, communicator=None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.communicator = communicator
        self.processed_files: Set[str] = set()
        self.selection_metrics = defaultdict(int)
    
    def select_prioritized_files_split_aware(self, data_dir: str, target_split: str, 
                                           class_needs: Dict[str, int], 
                                           max_files_per_class: int = 500) -> List[str]:
        """Select files dengan preserved logic"""
        try:
            from smartcash.dataset.augmentor.utils.file_processor import FileProcessor
            
            file_processor = FileProcessor({'data': {'dir': data_dir}})
            source_files = file_processor.get_split_files(target_split)
            
            if not source_files:
                self.logger.warning(f"Tidak ada file ditemukan untuk selection pada split {target_split}")
                return []
            
            files_metadata = self._extract_files_metadata_for_selection(source_files)
            return self.select_prioritized_files(class_needs, files_metadata, max_files_per_class)
            
        except Exception as e:
            self.logger.error(f"Error selecting files for split {target_split}: {str(e)}")
            return []
    
    def select_prioritized_files(self, class_needs: Dict[str, int], files_metadata: Dict[str, Dict[str, Any]], 
                               max_files_per_class: int = 500) -> List[str]:
        """Select files dengan preserved logic"""
        self.logger.info(f"🎯 Memulai seleksi file prioritas untuk {len(class_needs)} kelas")
        
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
        
        self.logger.info(f"✅ Seleksi selesai: {len(selected_files)} file terpilih dari {len(files_metadata)} total")
        return selected_files
    
    def _extract_files_metadata_for_selection(self, source_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract metadata dengan preserved logic"""
        files_metadata = {}
        
        for img_file in source_files:
            try:
                label_path = str(Path(img_file).parent.parent / 'labels' / f"{Path(img_file).stem}.txt")
                bboxes, class_labels = self._load_yolo_labels(label_path)
                
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
    
    def _load_yolo_labels(self, label_path: str) -> tuple:
        """Load labels dengan preserved logic"""
        bboxes, class_labels = [], []
        
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
        except Exception:
            pass
        
        return bboxes, class_labels
    
    def _group_files_by_primary_class(self, files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group files dengan preserved logic"""
        files_by_class = defaultdict(list)
        
        for file_path, metadata in files_metadata.items():
            primary_class = metadata.get('primary_class')
            if primary_class:
                files_by_class[primary_class].append(file_path)
        
        return dict(files_by_class)
    
    def _select_files_for_class(self, target_class: str, needed: int, available_files: List[str],
                              files_metadata: Dict[str, Dict[str, Any]], max_files: int) -> List[str]:
        """Select files untuk kelas dengan preserved logic"""
        if not available_files:
            return []
        
        scored_files = []
        for file_path in available_files:
            if file_path in self.processed_files:
                continue
                
            metadata = files_metadata.get(file_path, {})
            score = self._calculate_file_score(file_path, target_class, metadata)
            scored_files.append((file_path, score))
        
        scored_files = sorted(scored_files, key=lambda x: x[1], reverse=True)
        max_select = min(needed, max_files, len(scored_files))
        selected = [file_path for file_path, _ in scored_files[:max_select]]
        
        self.processed_files.update(selected)
        
        self.logger.info(f"📁 Kelas {target_class}: dipilih {len(selected)}/{len(available_files)} file")
        return selected
    
    def _calculate_file_score(self, file_path: str, target_class: str, metadata: Dict[str, Any]) -> float:
        """Calculate score dengan preserved logic"""
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
        """Optimize selection dengan preserved logic"""
        unique_files = list(set(selected_files))
        
        if len(unique_files) > 100:
            optimized_files = self._apply_diversity_optimization(unique_files, files_metadata)
        else:
            optimized_files = unique_files
        
        self.logger.info(f"🔄 Optimisasi: {len(selected_files)} → {len(unique_files)} → {len(optimized_files)} file")
        return optimized_files
    
    def _apply_diversity_optimization(self, files: List[str], files_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Apply diversity dengan preserved logic"""
        class_counts = Counter()
        
        for file_path in files:
            metadata = files_metadata.get(file_path, {})
            for cls, count in metadata.get('class_counts', {}).items():
                class_counts[cls] += count
        
        avg_count = sum(class_counts.values()) / len(class_counts) if class_counts else 0
        over_represented = {cls: count for cls, count in class_counts.items() if count > avg_count * 2}
        
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
        
        self.logger.info(f"🎲 Diversity optimization: removed {removed_count} files")
        return optimized_files