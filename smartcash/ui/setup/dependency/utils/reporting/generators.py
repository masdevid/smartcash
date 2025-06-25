"""
File: smartcash/ui/setup/dependency/utils/reporting/generators.py

Report generation utilities for dependency management.

This module provides functions to generate various reports including system
compatibility, installation summaries, and package status reports.
"""

# Standard library imports
from typing import Any, Callable, Dict, List, Optional, Tuple

# Absolute imports
from smartcash.ui.setup.dependency.utils.package.categories import get_package_categories

__all__ = [
    'generate_comprehensive_status_report',
    'generate_installation_summary_report',
    'generate_analysis_summary_report',
    'generate_system_compatibility_report'
]

def generate_comprehensive_status_report(system_info: Dict[str, Any], 
                                        package_status: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive HTML report"""
    # Report sections
    system_section = _generate_system_info_section(system_info)
    summary_section = _generate_summary_section(package_status)
    categories_section = _generate_categories_section(package_status, get_package_categories)
    
    return f"""
    <div style="font-family:monospace;max-width:100%;overflow:hidden;">
        <h3 style="color:#2c3e50;margin:0 0 15px 0;">🔍 Comprehensive Package Status Report</h3>
        {system_section}
        {summary_section}
        {categories_section}
    </div>
    """

def generate_installation_summary_report(results: List[Dict[str, Any]]) -> str:
    """Generate installation summary report"""
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    failed_packages = [r.get('package', 'Unknown') for r in results if not r.get('success', False)]
    
    status_color = "#28a745" if success_count == total_count else "#ffc107" if success_count > 0 else "#dc3545"
    
    failed_section = ""
    if failed_packages:
        failed_list = "<br>".join([f"   • {pkg}" for pkg in failed_packages[:5]])
        if len(failed_packages) > 5:
            failed_list += f"<br>   • ... and {len(failed_packages) - 5} more"
        failed_section = f"""
        <div style="background:#fff3cd;padding:8px;border-radius:4px;margin:8px 0;">
            <strong>⚠️ Failed Packages:</strong><br>
            {failed_list}
        </div>
        """
    
    return f"""
    <div style="font-family:monospace;max-width:100%;overflow:hidden;">
        <h4 style="color:{status_color};margin:0 0 10px 0;">📦 Installation Summary</h4>
        <p><strong>Success Rate:</strong> {success_count}/{total_count} packages installed</p>
        {failed_section}
    </div>
    """

def generate_analysis_summary_report(analysis_results: Dict[str, Any]) -> str:
    """Generate analysis summary report"""
    installed = analysis_results.get('installed', [])
    not_installed = analysis_results.get('not_installed', [])
    errors = analysis_results.get('errors', [])
    compatibility_issues = analysis_results.get('compatibility_issues', [])
    
    return f"""
    <div style="font-family:monospace;max-width:100%;overflow:hidden;">
        <h4 style="color:#2c3e50;margin:0 0 10px 0;">🔍 Analysis Summary</h4>
        <div style="background:#f8f9fa;padding:8px;border-radius:4px;">
            <strong>✅ Installed:</strong> {len(installed)} packages<br>
            <strong>❌ Not Installed:</strong> {len(not_installed)} packages<br>
            <strong>🔥 Errors:</strong> {len(errors)} packages<br>
            <strong>⚠️ Compatibility Issues:</strong> {len(compatibility_issues)} found
        </div>
    </div>
    """

def generate_system_compatibility_report(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate system compatibility report"""
    warnings = []
    recommendations = []
    
    # Check Python version
    python_version = system_info.get('python_version', '')
    if python_version and python_version < '3.8':
        warnings.append("Python version < 3.8 may have compatibility issues")
        recommendations.append("Consider upgrading to Python 3.8+")
    
    # Check memory
    memory_gb = system_info.get('memory_info', {}).get('available_gb', 0)
    if memory_gb < 2:
        warnings.append("Low available memory (< 2GB)")
        recommendations.append("Close other applications to free memory")
    
    # Check CUDA
    if not system_info.get('gpu_info', {}).get('cuda_available', False):
        recommendations.append("CUDA not available - CPU-only mode for deep learning")
    
    return {
        'warnings': warnings,
        'recommendations': recommendations,
        'compatible': len(warnings) == 0
    }

def _generate_system_info_section(system_info: Dict[str, Any]) -> str:
    """Generate system info section"""
    
    info_items = [
        ('Environment', system_info.get('environment', 'Unknown')),
        ('Platform', f"{system_info.get('platform', 'Unknown')} {system_info.get('platform_release', '')}"),
        ('Python', system_info.get('python_version', 'Unknown')),
        ('Architecture', system_info.get('architecture', 'Unknown')),
        ('Memory', f"{system_info.get('memory_info', {}).get('available_gb', 0):.1f}GB available"),
        ('CUDA', '✅ Available' if system_info.get('gpu_info', {}).get('cuda_available') else '❌ Not Available')
    ]
    
    table_rows = ''.join(f"<tr><td><strong>{label}:</strong></td><td>{value}</td></tr>" 
                        for label, value in info_items)
    
    return f"""
    <div style="background:#f0f8ff;padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#2c3e50;">💻 System Information</h4>
        <table style="width:100%;font-size:12px;">{table_rows}</table>
    </div>
    """

def _generate_summary_section(package_status: Dict[str, Dict[str, Any]]) -> str:
    """Generate package summary section"""
    installed_count = sum(1 for status in package_status.values() if status.get('installed', False))
    total_count = len(package_status)
    
    return f"""
    <div style="background:#f8f9fa;padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#2c3e50;">📦 Package Summary</h4>
        <p><strong>Total Packages:</strong> {total_count}</p>
        <p><strong>Installed:</strong> {installed_count} / {total_count}</p>
        <p><strong>Installation Rate:</strong> {(installed_count/total_count*100):.1f}%</p>
    </div>
    """

def _generate_categories_section(package_status: Dict[str, Dict[str, Any]], get_package_categories_func) -> str:
    """Generate categories section"""
    try:
        categories = get_package_categories_func()
        
        category_html = ""
        for category, packages in categories.items():
            installed_in_category = sum(1 for pkg in packages 
                                      if package_status.get(pkg, {}).get('installed', False))
            total_in_category = len(packages)
            
            category_html += f"""
            <div style="margin:8px 0;padding:8px;background:#f8f9fa;border-radius:4px;">
                <strong>{category}:</strong> {installed_in_category}/{total_in_category} installed
            </div>
            """
        
        return f"""
        <div style="margin:10px 0;">
            <h4 style="margin:0 0 8px 0;color:#2c3e50;">📋 By Category</h4>
            {category_html}
        </div>
        """
    except:
        return "<div>Category information unavailable</div>"