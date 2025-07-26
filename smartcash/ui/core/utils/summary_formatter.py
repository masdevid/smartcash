"""
Unified summary formatter for dataset operations across all modules.

This module provides consistent formatting for summaries across downloader,
preprocessing, and augmentation modules with standardized metrics.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path


class UnifiedSummaryFormatter:
    """Unified formatter for creating consistent summaries across all dataset modules."""
    
    @staticmethod
    def format_dataset_summary(
        module_name: str,
        operation_type: str,
        result: Dict[str, Any],
        include_paths: bool = True
    ) -> str:
        """
        Format a unified dataset operation summary.
        
        Args:
            module_name: Name of the module (downloader, preprocessing, augmentation)
            operation_type: Type of operation (check, process, cleanup, etc.)
            result: Operation result dictionary
            include_paths: Whether to include data paths in summary
            
        Returns:
            Formatted markdown summary string
        """
        success = result.get('success', False)
        status_icon = "✅" if success else "❌"
        status_text = "Berhasil" if success else "Gagal"
        
        # Get statistics based on module type
        stats = UnifiedSummaryFormatter._extract_statistics(result, module_name)
        
        # Build summary sections
        sections = []
        
        # Header section
        sections.append(f"## {UnifiedSummaryFormatter._get_module_icon(module_name)} Ringkasan {operation_type.title()}")
        sections.append("### Status Operasi")
        sections.append(f"{status_icon} **{status_text}**")
        sections.append("")
        
        # Statistics section
        if stats:
            sections.append("### Statistik Dataset")
            sections.extend(UnifiedSummaryFormatter._format_statistics_section(stats, module_name))
            sections.append("")
        
        # Per-split breakdown if available
        split_stats = UnifiedSummaryFormatter._extract_split_statistics(result, module_name)
        if split_stats:
            sections.append("### Per Split Breakdown")
            sections.extend(UnifiedSummaryFormatter._format_split_section(split_stats))
            sections.append("")
        
        # Data paths section
        if include_paths:
            paths = UnifiedSummaryFormatter._extract_paths(result, module_name)
            if paths:
                sections.append("### Data Paths")
                sections.extend(UnifiedSummaryFormatter._format_paths_section(paths))
                sections.append("")
        
        # Operation details section
        details = UnifiedSummaryFormatter._extract_operation_details(result, module_name)
        if details:
            sections.append("### Detail Operasi")
            sections.extend(UnifiedSummaryFormatter._format_details_section(details))
            sections.append("")
        
        # Footer with message
        message = result.get('message', f'{operation_type.title()} selesai.')
        sections.append("---")
        sections.append(f"**Pesan:** *{message}*")
        
        return "\n".join(sections)
    
    @staticmethod
    def _get_module_icon(module_name: str) -> str:
        """Get icon for module."""
        icons = {
            'downloader': '📥',
            'preprocessing': '⚙️',
            'augmentation': '🎨',
            'visualization': '📊'
        }
        return icons.get(module_name.lower(), '📋')
    
    @staticmethod
    def _extract_statistics(result: Dict[str, Any], module_name: str) -> Dict[str, Any]:
        """Extract statistics based on module type."""
        stats = {}
        
        if module_name.lower() == 'downloader':
            # Downloader format
            stats['total_files'] = result.get('file_count', 0)
            summary = result.get('summary', {})
            stats['total_images'] = summary.get('total_images', 0)
            stats['total_labels'] = summary.get('total_labels', 0)
            stats['total_size'] = result.get('total_size', '0B')
            
        elif module_name.lower() == 'preprocessing':
            # Preprocessing format
            statistics = result.get('statistics', {})
            stats['files_processed'] = statistics.get('files_processed', 0)
            stats['files_skipped'] = statistics.get('files_skipped', 0)
            stats['files_failed'] = statistics.get('files_failed', 0)
            stats['files_missing'] = statistics.get('files_missing', 0)
            stats['raw_images'] = statistics.get('raw_images', 0)
            stats['files_deleted'] = statistics.get('files_deleted', 0)
            stats['space_reclaimed_mb'] = statistics.get('space_reclaimed_mb', 0)
            stats['total_time'] = result.get('total_time_seconds', 0)
            
            # Extract per-split data if available
            splits_data = result.get('splits_data', {})
            if splits_data:
                total_raw = sum(split.get('raw_count', 0) for split in splits_data.values())
                total_processed = sum(split.get('processed_count', 0) for split in splits_data.values())
                stats['total_raw_files'] = total_raw
                stats['total_preprocessed_images'] = total_processed
            
        elif module_name.lower() == 'augmentation':
            # Augmentation format
            statistics = result.get('statistics', {})
            splits_data = result.get('splits_data', {})
            
            if splits_data:
                total_raw = sum(split.get('raw_count', 0) for split in splits_data.values())
                total_aug = sum(split.get('augmented_count', 0) for split in splits_data.values())
                total_prep = sum(split.get('preprocessed_count', 0) for split in splits_data.values())
                
                stats['total_raw_files'] = total_raw
                stats['total_augmented_files'] = total_aug
                stats['total_preprocessed_files'] = total_prep
            
            stats['augmentation_factor'] = statistics.get('augmentation_factor', 1)
            stats['total_time'] = result.get('total_time_seconds', 0)
        
        return stats
    
    @staticmethod
    def _format_statistics_section(stats: Dict[str, Any], module_name: str) -> List[str]:
        """Format the statistics section."""
        lines = []
        
        if module_name.lower() == 'downloader':
            lines.append(f"- **Total Files**: 📁 {stats.get('total_files', 0):,} file")
            lines.append(f"- **Total Images**: 🖼️ {stats.get('total_images', 0):,}")
            lines.append(f"- **Total Labels**: 🏷️ {stats.get('total_labels', 0):,}")
            lines.append(f"- **Total Size**: 💾 {stats.get('total_size', '0B')}")
            
        elif module_name.lower() == 'preprocessing':
            if 'total_raw_files' in stats:
                lines.append(f"- **Total Raw Files**: 📁 {stats['total_raw_files']:,} file")
                lines.append(f"- **Total Preprocessed Images**: 🖼️ {stats['total_preprocessed_images']:,}")
            else:
                # Individual operation stats
                if stats.get('files_processed', 0) > 0:
                    lines.append(f"- **Files Processed**: ✔️ {stats['files_processed']:,}")
                if stats.get('files_skipped', 0) > 0:
                    lines.append(f"- **Files Skipped**: ⏭️ {stats['files_skipped']:,}")
                if stats.get('files_failed', 0) > 0:
                    lines.append(f"- **Files Failed**: ❌ {stats['files_failed']:,}")
                if stats.get('files_missing', 0) > 0:
                    lines.append(f"- **Files Missing**: ❓ {stats['files_missing']:,}")
                if stats.get('raw_images', 0) > 0:
                    lines.append(f"- **Raw Images**: 🖼️ {stats['raw_images']:,}")
                if stats.get('files_deleted', 0) > 0:
                    lines.append(f"- **Files Deleted**: 🗑️ {stats['files_deleted']:,}")
                if stats.get('space_reclaimed_mb', 0) > 0:
                    lines.append(f"- **Space Reclaimed**: 💾 {stats['space_reclaimed_mb']:.2f} MB")
            
            if stats.get('total_time', 0) > 0:
                lines.append(f"- **Total Time**: ⏱️ {stats['total_time']:.2f} detik")
            
        elif module_name.lower() == 'augmentation':
            lines.append(f"- **Total Raw Files**: 📁 {stats.get('total_raw_files', 0):,}")
            lines.append(f"- **Total Augmented Files**: 🎨 {stats.get('total_augmented_files', 0):,}")
            lines.append(f"- **Total Preprocessed Files**: ⚙️ {stats.get('total_preprocessed_files', 0):,}")
            
            if stats.get('augmentation_factor', 1) > 1:
                lines.append(f"- **Augmentation Factor**: ✖️ {stats['augmentation_factor']}x")
            
            if stats.get('total_time', 0) > 0:
                lines.append(f"- **Total Time**: ⏱️ {stats['total_time']:.2f} detik")
        
        return lines
    
    @staticmethod
    def _extract_split_statistics(result: Dict[str, Any], module_name: str) -> Dict[str, Dict[str, Any]]:
        """Extract per-split statistics."""
        splits_data = result.get('splits_data', {})
        
        if not splits_data:
            # Try alternative formats
            summary = result.get('summary', {})
            if 'by_split' in summary:
                splits_data = summary['by_split']
        
        return splits_data
    
    @staticmethod
    def _format_split_section(splits_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Format per-split statistics section."""
        lines = []
        
        for split_name, split_data in splits_data.items():
            if split_name.upper() in ['TRAIN', 'VALID', 'TEST']:
                split_display = split_name.upper()
                
                # Format based on available data
                parts = []
                
                if 'raw_count' in split_data or 'raw' in split_data:
                    raw_count = split_data.get('raw_count', split_data.get('raw', 0))
                    status = "available" if raw_count > 0 else "not_found"
                    parts.append(f"Raw={raw_count}({status})")
                
                if 'augmented_count' in split_data or 'augmented' in split_data:
                    aug_count = split_data.get('augmented_count', split_data.get('augmented', 0))
                    status = "available" if aug_count > 0 else "not_found"
                    parts.append(f"Aug={aug_count}({status})")
                
                if 'preprocessed_count' in split_data or 'preprocessed' in split_data:
                    prep_count = split_data.get('preprocessed_count', split_data.get('preprocessed', 0))
                    status = "available" if prep_count > 0 else "not_found"
                    parts.append(f"Prep={prep_count}({status})")
                
                if parts:
                    lines.append(f"- **{split_display}**: {', '.join(parts)}")
        
        return lines
    
    @staticmethod
    def _extract_paths(result: Dict[str, Any], module_name: str) -> Dict[str, str]:
        """Extract data paths."""
        paths = {}
        
        # Common paths
        if 'dataset_path' in result:
            paths['Dataset Path'] = result['dataset_path']
        
        if 'data_path' in result:
            paths['Data Path'] = result['data_path']
        
        # Module-specific paths
        if module_name.lower() == 'preprocessing':
            if 'output_path' in result:
                paths['Preprocessed Path'] = result['output_path']
                
        elif module_name.lower() == 'augmentation':
            if 'augmented_path' in result:
                paths['Augmented Path'] = result['augmented_path']
        
        return paths
    
    @staticmethod
    def _format_paths_section(paths: Dict[str, str]) -> List[str]:
        """Format paths section."""
        lines = []
        for path_name, path_value in paths.items():
            lines.append(f"- **{path_name}**: 📂 `{path_value}`")
        return lines
    
    @staticmethod
    def _extract_operation_details(result: Dict[str, Any], module_name: str) -> Dict[str, Any]:
        """Extract operation-specific details."""
        details = {}
        
        # Common details
        if 'operation_id' in result:
            details['Operation ID'] = result['operation_id']
        
        if 'timestamp' in result:
            details['Timestamp'] = result['timestamp']
        
        # Module-specific details
        if module_name.lower() == 'downloader':
            issues = result.get('issues', [])
            if issues:
                details['Issues Found'] = f"{len(issues)} masalah ditemukan"
            else:
                details['Data Quality'] = "✅ Tidak ada masalah"
        
        return details
    
    @staticmethod
    def _format_details_section(details: Dict[str, Any]) -> List[str]:
        """Format operation details section."""
        lines = []
        for detail_name, detail_value in details.items():
            lines.append(f"- **{detail_name}**: {detail_value}")
        return lines