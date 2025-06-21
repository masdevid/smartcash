"""
File: smartcash/ui/setup/dependency/utils/report_generator_utils.py
Deskripsi: Report generation utilities dengan consolidated HTML templates
"""

from typing import Dict, Any, List
from smartcash.ui.setup.dependency.components.package_selector import get_package_categories

def generate_comprehensive_status_report(system_info: Dict[str, Any], 
                                        package_status: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive HTML report dengan consolidated approach"""
    
    # Report sections
    system_section = _generate_system_info_section(system_info)
    summary_section = _generate_summary_section(package_status)
    categories_section = _generate_categories_section(package_status)
    
    return f"""
    <div style="font-family:monospace;max-width:100%;overflow:hidden;">
        <h3 style="color:#2c3e50;margin:0 0 15px 0;">🔍 Comprehensive Package Status Report</h3>
        {system_section}
        {summary_section}
        {categories_section}
    </div>
    """

def _generate_system_info_section(system_info: Dict[str, Any]) -> str:
    """Generate system info section - one-liner table generation"""
    
    info_items = [
        ('Environment', system_info.get('environment', 'Unknown')),
        ('Platform', f"{system_info.get('platform', 'Unknown')} {system_info.get('platform_release', '')}"),
        ('Python', system_info.get('python_version', 'Unknown')),
        ('Architecture', system_info.get('architecture', 'Unknown')),
        ('Memory', f"{system_info.get('memory_available_gb', 0):.1f}GB / {system_info.get('memory_total_gb', 0):.1f}GB"),
        ('CUDA', '✅ Available' if system_info.get('cuda_available') else '❌ Not Available')
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
    """Generate summary statistics section - one-liner calculations"""
    
    total_packages = len(package_status)
    installed_packages = sum(1 for pkg_info in package_status.values() if pkg_info['installed'])
    missing_packages = total_packages - installed_packages
    coverage_percent = (installed_packages/total_packages*100) if total_packages > 0 else 0
    
    return f"""
    <div style="background:#e8f5e9;padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#27ae60;">📊 Summary</h4>
        <div style="display:flex;justify-content:space-between;">
            <div><strong>Total Packages:</strong> {total_packages}</div>
            <div><strong>✅ Installed:</strong> {installed_packages}</div>
            <div><strong>❌ Missing:</strong> {missing_packages}</div>
            <div><strong>Coverage:</strong> {coverage_percent:.1f}%</div>
        </div>
    </div>
    """

def _generate_categories_section(package_status: Dict[str, Dict[str, Any]]) -> str:
    """Generate categories breakdown section"""
    
    package_categories = get_package_categories()
    categories_html = ""
    
    for category in package_categories:
        category_packages = [
            (pkg_key, pkg_info) for pkg_key, pkg_info in package_status.items() 
            if pkg_info['category'] == category['name']
        ]
        
        installed_count = sum(1 for _, pkg_info in category_packages if pkg_info['installed'])
        total_count = len(category_packages)
        
        # Package table rows - one-liner generation
        package_rows = ''.join(
            f"""<tr>
                <td style="width:20px;">{'✅' if pkg_info['installed'] else '❌'}</td>
                <td><strong>{pkg_info['name']}</strong></td>
                <td style="color:#666;">{'v' + pkg_info['version'] if pkg_info.get('version') else 'Not installed'}</td>
            </tr>"""
            for pkg_key, pkg_info in category_packages
        )
        
        categories_html += f"""
        <div style="background:#ffffff;padding:10px;border:1px solid #ddd;border-radius:5px;margin:8px 0;">
            <h5 style="margin:0 0 8px 0;color:#34495e;">{category['icon']} {category['name']} ({installed_count}/{total_count})</h5>
            <table style="width:100%;font-size:11px;">{package_rows}</table>
        </div>
        """
    
    return categories_html

def generate_installation_summary_report(installation_results: Dict[str, bool], 
                                        duration: float = 0) -> str:
    """Generate installation summary report"""
    
    total_packages = len(installation_results)
    successful_packages = sum(1 for success in installation_results.values() if success)
    failed_packages = total_packages - successful_packages
    
    # Success/failure lists
    success_list = [pkg for pkg, success in installation_results.items() if success]
    failure_list = [pkg for pkg, success in installation_results.items() if not success]
    
    return f"""
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;margin:10px 0;">
        <h4 style="margin:0 0 10px 0;color:#2c3e50;">📋 Installation Summary</h4>
        
        <div style="margin:10px 0;">
            <strong>Duration:</strong> {duration:.1f} seconds<br>
            <strong>Total Packages:</strong> {total_packages}<br>
            <strong>✅ Successful:</strong> {successful_packages}<br>
            <strong>❌ Failed:</strong> {failed_packages}<br>
            <strong>Success Rate:</strong> {(successful_packages/total_packages*100):.1f}%
        </div>
        
        {_generate_package_list_section("✅ Successfully Installed", success_list, "#e8f5e9")}
        {_generate_package_list_section("❌ Failed to Install", failure_list, "#f8d7da") if failure_list else ""}
    </div>
    """

def _generate_package_list_section(title: str, packages: List[str], bg_color: str) -> str:
    """Generate package list section - one-liner approach"""
    
    if not packages:
        return ""
    
    package_items = ''.join(f"<li>{pkg}</li>" for pkg in packages[:10])  # Limit to 10 items
    more_text = f"<li><em>... and {len(packages) - 10} more</em></li>" if len(packages) > 10 else ""
    
    return f"""
    <div style="background:{bg_color};padding:8px;border-radius:4px;margin:8px 0;">
        <strong>{title} ({len(packages)}):</strong>
        <ul style="margin:5px 0;padding-left:20px;font-size:11px;">
            {package_items}{more_text}
        </ul>
    </div>
    """

def generate_analysis_summary_report(analysis_results: Dict[str, Any]) -> str:
    """Generate analysis summary report"""
    
    installed_count = len(analysis_results.get('installed', []))
    missing_count = len(analysis_results.get('missing', []))
    upgrade_count = len(analysis_results.get('upgrade_needed', []))
    total_count = installed_count + missing_count + upgrade_count
    
    return f"""
    <div style="background:#fff3cd;padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#856404;">🔍 Analysis Summary</h4>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:12px;">
            <div><strong>Total:</strong> {total_count}</div>
            <div><strong>✅ Installed:</strong> {installed_count}</div>
            <div><strong>❌ Missing:</strong> {missing_count}</div>
            <div><strong>⚠️ Upgrade:</strong> {upgrade_count}</div>
        </div>
        
        {_generate_recommendation_section(analysis_results)}
    </div>
    """

def _generate_recommendation_section(analysis_results: Dict[str, Any]) -> str:
    """Generate recommendation section berdasarkan analysis"""
    
    missing_count = len(analysis_results.get('missing', []))
    upgrade_count = len(analysis_results.get('upgrade_needed', []))
    
    if missing_count == 0 and upgrade_count == 0:
        return """
        <div style="margin-top:8px;padding:6px;background:#d4edda;border-radius:3px;">
            <strong>✅ Recommendation:</strong> All packages are properly installed. No action required.
        </div>
        """
    
    recommendations = []
    if missing_count > 0:
        recommendations.append(f"Install {missing_count} missing packages")
    if upgrade_count > 0:
        recommendations.append(f"Upgrade {upgrade_count} packages to newer versions")
    
    return f"""
    <div style="margin-top:8px;padding:6px;background:#ffeaa7;border-radius:3px;">
        <strong>💡 Recommendation:</strong> {' and '.join(recommendations)}.
    </div>
    """

def generate_quick_status_badge(package_status: Dict[str, Dict[str, Any]]) -> str:
    """Generate quick status badge untuk UI - one-liner"""
    
    total = len(package_status)
    installed = sum(1 for pkg in package_status.values() if pkg['installed'])
    coverage = (installed / total * 100) if total > 0 else 0
    
    color = "#28a745" if coverage >= 90 else "#ffc107" if coverage >= 70 else "#dc3545"
    
    return f"""
    <span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:bold;">
        {installed}/{total} ({coverage:.0f}%)
    </span>
    """

def generate_system_compatibility_report(system_info: Dict[str, Any]) -> str:
    """Generate system compatibility report"""
    
    # Check requirements
    requirements = {
        'python_ok': _check_python_version(system_info.get('python_version', '')),
        'memory_ok': system_info.get('memory_total_gb', 0) >= 2.0,
        'platform_ok': system_info.get('platform', '') in ['Linux', 'Windows', 'Darwin']
    }
    
    compatibility_items = [
        ('Python 3.7+', requirements['python_ok']),
        ('Memory 2GB+', requirements['memory_ok']),
        ('Supported Platform', requirements['platform_ok'])
    ]
    
    check_rows = ''.join(
        f"<tr><td>{item}</td><td>{'✅ OK' if status else '❌ Fail'}</td></tr>"
        for item, status in compatibility_items
    )
    
    overall_ok = all(requirements.values())
    bg_color = "#e8f5e9" if overall_ok else "#f8d7da"
    status_text = "✅ System Compatible" if overall_ok else "⚠️ Compatibility Issues"
    
    return f"""
    <div style="background:{bg_color};padding:10px;border-radius:5px;margin:8px 0;">
        <h5 style="margin:0 0 5px 0;">🖥️ System Compatibility</h5>
        <table style="width:100%;font-size:11px;">{check_rows}</table>
        <div style="margin-top:5px;font-weight:bold;">{status_text}</div>
    </div>
    """

def _check_python_version(python_version: str) -> bool:
    """Check Python version compatibility - one-liner"""
    try:
        major, minor = map(int, python_version.split('.')[:2])
        return major >= 3 and minor >= 7
    except Exception:
        return False

def format_duration(seconds: float) -> str:
    """Format duration dalam human-readable format - one-liner"""
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds//60)}m {int(seconds%60)}s"

def create_status_summary_text(package_status: Dict[str, Dict[str, Any]]) -> str:
    """Create concise status summary text - one-liner"""
    total = len(package_status)
    installed = sum(1 for pkg in package_status.values() if pkg['installed'])
    return f"{installed}/{total} packages installed ({(installed/total*100):.0f}% coverage)"