"""
File: smartcash/dataset/preprocessor/utils/preprocessing_stats.py
Deskripsi: Statistics collection dan aggregation untuk preprocessing operations
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from smartcash.common.logger import get_logger


@dataclass
class SplitStats:
    """Statistics untuk single split processing."""
    split: str
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    processing_time: float = 0.0
    success_rate: float = 0.0
    avg_processing_time_per_image: float = 0.0


@dataclass
class ProcessingStats:
    """Comprehensive preprocessing statistics."""
    total_images: int = 0
    total_processed: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    total_processing_time: float = 0.0
    overall_success_rate: float = 0.0
    splits_processed: int = 0
    split_stats: Dict[str, SplitStats] = None
    
    def __post_init__(self):
        if self.split_stats is None:
            self.split_stats = {}


class PreprocessingStats:
    """Statistics collector dan aggregator untuk preprocessing operations."""
    
    def __init__(self, logger=None):
        """Initialize stats collector dengan logging."""
        self.logger = logger or get_logger()
        self.collection_start_time = time.time()
        self.active_collections = {}
        
    def collect_processing_stats(self, split_results: Dict[str, Dict[str, Any]]) -> ProcessingStats:
        """
        Collect dan validate statistics dari split results.
        
        Args:
            split_results: Dictionary hasil processing per split
            
        Returns:
            ProcessingStats object dengan comprehensive metrics
        """
        stats = ProcessingStats()
        
        try:
            for split_name, split_result in split_results.items():
                if not isinstance(split_result, dict):
                    continue
                
                # Create split statistics
                split_stats = SplitStats(
                    split=split_name,
                    processed=split_result.get('processed', 0),
                    skipped=split_result.get('skipped', 0),
                    failed=split_result.get('failed', 0),
                    processing_time=split_result.get('processing_time', 0.0)
                )
                
                # Calculate derived metrics
                total_attempted = split_stats.processed + split_stats.skipped + split_stats.failed
                if total_attempted > 0:
                    split_stats.success_rate = (split_stats.processed / total_attempted) * 100
                
                if split_stats.processed > 0 and split_stats.processing_time > 0:
                    split_stats.avg_processing_time_per_image = split_stats.processing_time / split_stats.processed
                
                stats.split_stats[split_name] = split_stats
                
                # Aggregate totals
                stats.total_processed += split_stats.processed
                stats.total_skipped += split_stats.skipped
                stats.total_failed += split_stats.failed
                stats.total_processing_time += split_stats.processing_time
            
            # Calculate overall metrics
            stats.total_images = stats.total_processed + stats.total_skipped + stats.total_failed
            stats.splits_processed = len([s for s in stats.split_stats.values() if s.processed > 0])
            
            if stats.total_images > 0:
                stats.overall_success_rate = (stats.total_processed / stats.total_images) * 100
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting stats: {str(e)}")
        
        return stats
    
    def aggregate_processing_results(self, split_results: Dict[str, Dict[str, Any]], 
                                   processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Aggregate processing results ke format yang user-friendly.
        
        Args:
            split_results: Dictionary hasil processing per split
            processing_time: Total processing time
            
        Returns:
            Dictionary aggregated results
        """
        stats = self.collect_processing_stats(split_results)
        stats.total_processing_time = processing_time
        
        # Create user-friendly aggregated result
        aggregated = {
            'total_images': stats.total_images,
            'total_processed': stats.total_processed,
            'total_skipped': stats.total_skipped,
            'total_failed': stats.total_failed,
            'processing_time': stats.total_processing_time,
            'overall_success_rate': round(stats.overall_success_rate, 2),
            'splits_processed': stats.splits_processed,
            'successful_splits': [name for name, s in stats.split_stats.items() if s.processed > 0]
        }
        
        # Add detailed split statistics
        aggregated['split_stats'] = {}
        for split_name, split_stat in stats.split_stats.items():
            aggregated['split_stats'][split_name] = {
                'images': split_stat.processed,
                'labels': split_stat.processed,  # Assume 1:1 ratio untuk labels
                'skipped': split_stat.skipped,
                'failed': split_stat.failed,
                'success_rate': round(split_stat.success_rate, 1),
                'processing_time': round(split_stat.processing_time, 2),
                'complete': split_stat.failed == 0 and split_stat.processed > 0
            }
        
        # Add performance metrics
        if stats.total_processed > 0 and stats.total_processing_time > 0:
            aggregated['performance_metrics'] = {
                'images_per_second': round(stats.total_processed / stats.total_processing_time, 2),
                'avg_time_per_image': round(stats.total_processing_time / stats.total_processed, 3),
                'throughput_classification': self._classify_throughput(
                    stats.total_processed / stats.total_processing_time
                )
            }
        
        return aggregated
    
    def format_stats_summary(self, stats: ProcessingStats) -> str:
        """
        Format statistics ke human-readable summary.
        
        Args:
            stats: ProcessingStats object
            
        Returns:
            String formatted summary
        """
        lines = [
            "📊 Preprocessing Statistics Summary",
            "=" * 45
        ]
        
        # Overall stats
        lines.extend([
            f"\n🎯 Overall Results:",
            f"   • Total Images: {stats.total_images:,}",
            f"   • Processed: {stats.total_processed:,} ({stats.overall_success_rate:.1f}%)",
            f"   • Skipped: {stats.total_skipped:,}",
            f"   • Failed: {stats.total_failed:,}",
            f"   • Processing Time: {stats.total_processing_time:.1f}s"
        ])
        
        # Performance metrics
        if stats.total_processed > 0 and stats.total_processing_time > 0:
            throughput = stats.total_processed / stats.total_processing_time
            lines.extend([
                f"\n⚡ Performance:",
                f"   • Throughput: {throughput:.1f} images/sec",
                f"   • Avg Time/Image: {stats.total_processing_time/stats.total_processed:.3f}s"
            ])
        
        # Split breakdown
        if stats.split_stats:
            lines.append(f"\n📂 Split Breakdown:")
            for split_name, split_stat in stats.split_stats.items():
                if split_stat.processed > 0:
                    lines.append(
                        f"   • {split_name}: {split_stat.processed:,} processed "
                        f"({split_stat.success_rate:.1f}% success) in {split_stat.processing_time:.1f}s"
                    )
        
        return "\n".join(lines)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of stats collector status."""
        return {
            'collector_ready': True,
            'collection_uptime': time.time() - self.collection_start_time,
            'active_collections': len(self.active_collections),
            'supported_metrics': [
                'processing_stats', 'performance_metrics', 'split_breakdown',
                'success_rates', 'throughput_analysis'
            ]
        }
    
    def reset_stats_collection(self) -> None:
        """Reset stats collection state."""
        self.active_collections.clear()
        self.collection_start_time = time.time()
        self.logger.debug("🔄 Stats collection state reset")
    
    def _classify_throughput(self, images_per_second: float) -> str:
        """Classify throughput performance."""
        if images_per_second >= 50:
            return "Excellent"
        elif images_per_second >= 20:
            return "Good"
        elif images_per_second >= 10:
            return "Fair"
        elif images_per_second >= 5:
            return "Slow"
        else:
            return "Very Slow"
    
    def export_stats_to_dict(self, stats: ProcessingStats) -> Dict[str, Any]:
        """Export ProcessingStats ke dictionary untuk serialization."""
        stats_dict = asdict(stats)
        
        # Convert SplitStats objects ke dictionaries
        if 'split_stats' in stats_dict and stats_dict['split_stats']:
            converted_split_stats = {}
            for split_name, split_stat in stats_dict['split_stats'].items():
                if hasattr(split_stat, '__dict__'):
                    converted_split_stats[split_name] = asdict(split_stat)
                else:
                    converted_split_stats[split_name] = split_stat
            stats_dict['split_stats'] = converted_split_stats
        
        return stats_dict