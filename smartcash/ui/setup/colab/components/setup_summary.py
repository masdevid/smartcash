"""
file_path: smartcash/ui/setup/colab/components/setup_summary.py

Komponen untuk menampilkan ringkasan setup environment dengan pembaruan status.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Tuple, List, Union

# Tipe status untuk type hinting yang lebih baik
StatusType = str  # 'success', 'warning', 'error', 'info', dll.

# Pemetaan warna untuk tiap tipe status
STATUS_COLORS = {
    'success': '#4caf50',  # Hijau
    'warning': '#ff9800',  # Oranye
    'error': '#f44336',    # Merah
    'info': '#2196f3',     # Biru
    'default': '#9e9e9e'   # Abu-abu
}

def create_setup_summary(initial_message: Optional[str] = None) -> widgets.HTML:
    """Buat widget ringkasan setup dengan pesan awal opsional.
    
    Args:
        initial_message: Pesan awal yang akan ditampilkan (opsional)
        
    Returns:
        Widget HTML untuk menampilkan ringkasan setup
    """
    return widgets.HTML(
        value=_get_initial_summary_content(initial_message),
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            margin='10px 0',
            background='#f9f9f9',
            overflow='auto'
        )
    )

def update_setup_summary(
    summary_widget: widgets.HTML, 
    status_message: str, 
    status_type: StatusType = 'info',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Perbarui ringkasan setup dengan informasi status terbaru.
    
    Args:
        summary_widget: Widget HTML yang akan diperbarui
        status_message: Pesan status utama yang akan ditampilkan
        status_type: Tipe status (mempengaruhi warna dan ikon)
        details: Kamus opsional dengan detail tambahan untuk ditampilkan
    """
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['default'])
    
    # Buat baris status utama
    icon = _get_status_icon(status_type)
    content = (
        f'<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">\n'
        f'    <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">\n'
        f'        <h3 style="margin: 0; color: {color};">{icon} {status_message}</h3>\n'
        '    </div>\n'
    )
    
    # Tambahkan detail jika disediakan
    if details:
        content += _format_enhanced_summary_content(details)
    
    content += '</div>'
    summary_widget.value = content

def update_setup_summary_with_verification(
    summary_widget: widgets.HTML,
    verification_results: Dict[str, Any],
    system_info: Optional[Dict[str, Any]] = None
) -> None:
    """Perbarui ringkasan setup dengan hasil verifikasi komprehensif.
    
    Args:
        summary_widget: Widget HTML yang akan diperbarui
        verification_results: Hasil dari operasi verifikasi
        system_info: Informasi sistem opsional dari env_detector
    """
    success = verification_results.get('success', False)
    issues = verification_results.get('issues', [])
    verification = verification_results.get('verification', {})
    
    # Tentukan status keseluruhan
    status_type = 'success' if success else 'error'
    main_message = "Setup Environment Selesai" if success else f"Ditemukan Masalah ({len(issues)} masalah)"
    
    # Warna dan ikon status
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['default'])
    icon = _get_status_icon(status_type)
    
    # Format konten HTML
    content = [
        '<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">',
        '    <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">',
        f'        <h3 style="margin: 0; color: {color};">{icon} {main_message}</h3>',
        '    </div>',
        '    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">',
        '        <div>',
        '            <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #2196f3;">',
        '                🔧 Komponen Setup',
        '            </h4>',
        f'            {_format_verification_status(verification)}',
        '        </div>',
        '        <div>',
        '            <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #4caf50;">',
        '                💻 Status Sistem',
        '            </h4>',
        f'            {_format_system_status(system_info)}',
        '        </div>',
        '    </div>'
    ]
    
    # Tambahkan bagian masalah jika ada
    if issues:
        content.append(_format_issues_section(issues))
    
    content.append('</div>')
    summary_widget.value = '\n'.join(content)

def _get_initial_summary_content(message: Optional[str] = None) -> str:
    """Hasilkan konten ringkasan awal.
    
    Args:
        message: Pesan kustom opsional untuk ditampilkan
        
    Returns:
        Konten HTML untuk status awal
    """
    default_parts = [
        '<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">',
        '    <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">',
        '        <h3 style="margin: 0; color: #9e9e9e;">ℹ️ Menunggu setup dimulai...</h3>',
        '    </div>',
        '    <p>Klik tombol "Setup Environment" untuk memulai proses konfigurasi.</p>',
        '</div>'
    ]
    
    if message:
        return '\n'.join([
            '<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">',
            '    <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">',
            f'        <h3 style="margin: 0; color: #9e9e9e;">ℹ️ {message}</h3>',
            '    </div>',
            '    <p>Klik tombol "Setup Environment" untuk memulai proses konfigurasi.</p>',
            '</div>'
        ])
    
    return '\n'.join(default_parts)

def _get_status_icon(status_type: StatusType) -> str:
    """Dapatkan ikon yang sesuai untuk tipe status.
    
    Args:
        status_type: Tipe status ('success', 'warning', 'error', 'info')
        
    Returns:
        String yang berisi emoji yang sesuai
    """
    if not status_type:
        return '🔹'  # Default icon for None/empty status
        
    status_type = str(status_type).lower()
    
    if status_type in ['success', 'ok', 'completed']:
        return '✅'
    elif status_type in ['warning', 'warn', 'partial']:
        return '⚠️'
    elif status_type in ['error', 'fail', 'failed']:
        return '❌'
    elif status_type in ['info', 'information', 'note']:
        return 'ℹ️'
    elif status_type in ['pending', 'in_progress', 'running']:
        return '⏳'
    else:
        return '🔹'  # Default icon: 'ℹ️'  # Default ke ikon informasi

def _format_enhanced_summary_content(data: Dict) -> str:
    """Format data ringkasan yang ditingkatkan untuk ditampilkan di widget.
    
    Args:
        data: Kamus yang berisi data ringkasan
        
    Returns:
        String HTML yang sudah diformat
    """
    if not isinstance(data, dict) or not data:
        return "<p style='color: #6c757d; padding: 10px;'>Tidak ada data ringkasan yang tersedia</p>"
    
    content = ['<div style="font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 14px; line-height: 1.6;">']
    
    try:
        # Format setiap bagian dengan gaya yang ditingkatkan
        for section, items in data.items():
            if not items:
                continue
                
            # Tambahkan header bagian dengan kode warna
            section_color = _get_section_color(section)
            content.append(
                f'<div style="margin: 15px 0 10px 0; padding: 0 0 0 10px; border-left: 3px solid {section_color};">'
                f'<h4 style="margin: 0 0 8px 0; color: {section_color}; font-size: 1.05em; font-weight: 600;">'
                f'{section}</h4>'
            )
            
            # Tambahkan item sebagai daftar yang distilisasi
            content.append('<div style="margin: 0 0 15px 10px; padding: 0;">')
        
            if isinstance(items, dict):
                for key, value in items.items():
                    # Format key-value pair
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        nested_items = []
                        for k, v in value.items():
                            if v is not None and v != '':
                                nested_items.append(f'<span style="display: block; margin: 2px 0;"><strong>{k}:</strong> {v}</span>')
                        content.append('<div style="margin: 5px 0 10px 0; padding: 8px; background: #f8f9fa; border-radius: 4px;">')
                        content.append(f'<div style="font-weight: 500; margin-bottom: 5px;">{key}:</div>')
                        content.append('<div style="margin-left: 10px;">' + ''.join(nested_items) + '</div>')
                        content.append('</div>')
                    elif value is not None and value != '':
                        # Handle simple key-value pairs
                        content.append(
                            f'<div style="margin: 6px 0; padding: 6px; background: #f8f9fa; border-radius: 4px; display: flex; align-items: flex-start;">'
                            f'<div style="font-weight: 500; min-width: 120px;">{key}:</div>'
                            f'<div style="flex: 1;">{value}</div>'
                            '</div>'
                        )
            
            # Close section divs
            content.append('</div>')
            content.append('</div>')
        
        content.append('</div>')
        return '\n'.join(str(item) for item in content if item is not None)
        
    except Exception as e:
        error_msg = f"<div style='color: #dc3545; padding: 10px; border: 1px solid #f5c6cb; background: #f8d7da; border-radius: 4px;'>"
        error_msg += "<strong>Error memformat ringkasan:</strong> "
        error_msg += f"{str(e)}\n\n{type(e).__name__}"
        error_msg += "</div>"
        return error_msg

def _format_verification_status(verification: Dict[str, Any]) -> str:
    """Format status verifikasi untuk ditampilkan.
    
    Args:
        verification: Kamus hasil verifikasi
        
    Returns:
        String HTML yang sudah diformat
    """
    if not verification:
        return "<p style='color: #6c757d;'>Tidak ada data verifikasi</p>"
    
    content = ['<div style="margin: 0; padding: 0; font-size: 0.92em;">']
    
    for key, status in verification.items():
        if isinstance(status, dict):
            # Tangani item verifikasi bersarang
            status_value = status.get('status', 'unknown')
            message = status.get('message', '')
            icon, color = _get_verification_status_icon(status_value)
            
            content.append(
                f'<div style="margin: 6px 0; padding: 4px 0; display: flex; align-items: center;">'
                f'<span style="color: {color}; margin-right: 8px; font-size: 1.1em;">{icon}</span>'
                f'<div><strong style="color: #343a40;">{key}:</strong> <span style="color: #495057;">{message}</span></div>'
                '</div>'
            )
        else:
            # Tangani status sederhana
            icon, color = _get_verification_status_icon(status)
            content.append(
                f'<div style="margin: 6px 0; padding: 4px 0; display: flex; align-items: center;">'
                f'<span style="color: {color}; margin-right: 8px; font-size: 1.1em;">{icon}</span>'
                f'<div><strong style="color: #343a40;">{key}:</strong> <span style="color: #495057;">{status}</span></div>'
                '</div>'
            )
    
    content.append('</div>')
    return '\n'.join(content)

def _format_system_status(system_info: Optional[Dict[str, Any]]) -> str:
    """Format status sistem untuk ditampilkan.
    
    Args:
        system_info: Kamus informasi sistem
        
    Returns:
        String HTML yang sudah diformat
    """
    if not system_info:
        return "<p style='color: #6c757d;'>Informasi sistem tidak tersedia</p>"
    
    content = ['<div style="margin: 0; padding: 0; font-size: 0.92em;">']
    
    # Tambahkan info dasar sistem
    if 'platform' in system_info or 'python_version' in system_info:
        platform = system_info.get('platform', 'Tidak diketahui')
        python_version = system_info.get('python_version', 'Tidak diketahui')
        
        content.extend([
            '<div style="margin: 6px 0; padding: 4px 0;">',
            '    <div style="margin-bottom: 6px;">',
            f'        <strong style="color: #343a40;">🖥️ Platform:</strong> <span style="color: #495057;">{platform}</span>',
            '    </div>',
            '    <div>',
            f'        <strong style="color: #343a40;">🐍 Python:</strong> <span style="color: #495057;">{python_version}</span>',
            '    </div>',
            '</div>'
        ])
    
    # Tambahkan penggunaan sumber daya jika tersedia
    if 'cpu_percent' in system_info or 'memory_percent' in system_info:
        cpu = system_info.get('cpu_percent', 'N/A')
        memory = system_info.get('memory_percent', 'N/A')
        
        # Tentukan warna berdasarkan penggunaan
        cpu_color = '#28a745' if isinstance(cpu, (int, float)) and cpu < 80 else '#dc3545'
        mem_color = '#28a745' if isinstance(memory, (int, float)) and memory < 80 else '#dc3545'
        
        content.extend([
            '<div style="margin: 12px 0 8px 0; padding: 8px 0; border-top: 1px dashed #e0e0e0;">',
            '    <div style="margin-bottom: 8px; color: #495057; font-weight: 500;">',
            '        📊 Penggunaan Sumber Daya',
            '    </div>',
            '    <div style="display: flex; flex-direction: column; gap: 4px;">',
            f'        <div><span style="display: inline-block; width: 80px;">CPU:</span> <span style="color: {cpu_color};">{cpu}%</span></div>',
            f'        <div><span style="display: inline-block; width: 80px;">Memori:</span> <span style="color: {mem_color};">{memory}%</span></div>',
            '    </div>',
            '</div>'
        ])
    
    # Tambahkan informasi penyimpanan jika tersedia
    storage = system_info.get('storage', {})
    if storage and storage.get('total_gb', 0) > 0:
        free_gb = storage.get('free_gb', 0)
        total_gb = storage.get('total_gb', 1)  # Hindari pembagian dengan nol
        usage_percent = (1 - (free_gb / total_gb)) * 100
        usage_color = "#4caf50" if usage_percent < 80 else "#ff9800" if usage_percent < 90 else "#f44336"
        
        content.extend([
            '<div style="margin: 12px 0 8px 0; padding: 8px 0; border-top: 1px dashed #e0e0e0;">',
            '    <div style="margin-bottom: 8px; color: #495057; font-weight: 500;">',
            '        💾 Penyimpanan',
            '    </div>',
            '    <div style="margin-top: 6px;">',
            f'        <div style="margin-bottom: 4px; font-size: 0.9em;">Total: {total_gb:.1f}GB (Tersedia: {free_gb:.1f}GB)</div>',
            '        <div style="width: 100%; background: #e9ecef; border-radius: 4px; overflow: hidden;">',
            f'            <div style="width: {usage_percent:.1f}%; height: 8px; background: {usage_color};"></div>',
            '        </div>',
            f'        <div style="font-size: 0.85em; color: #6c757d; margin-top: 4px; text-align: right;">',
            f'            {usage_percent:.1f}% digunakan',
            '        </div>',
            '    </div>',
            '</div>'
        ])
    
    content.append('</div>')
    return '\n'.join(content)

def _format_issues_section(issues: list) -> str:
    """Format bagian masalah untuk ditampilkan.
    
    Args:
        issues: Daftar string masalah
        
    Returns:
        String HTML yang sudah diformat untuk bagian masalah
    """
    if not issues:
        return ""
    
    content = [
        '<div style="margin-top: 20px; padding: 15px; background: #fff3f3; border-radius: 6px; border: 1px solid #ffd6d6;">',
        '    <div style="display: flex; align-items: center; margin-bottom: 12px; color: #721c24;">',
        '        <span style="font-size: 1.2em; margin-right: 8px;">⚠️</span>',
        '        <h4 style="margin: 0; font-size: 1.1em;">Masalah yang Ditemukan</h4>',
        '    </div>',
        '    <ul style="margin: 0; padding-left: 24px; color: #721c24;">'
    ]
    
    for issue in issues:
        content.append(f'<li style="margin-bottom: 6px; padding-left: 4px;">{issue}</li>')
    
    content.extend([
        '    </ul>',
        '    <div style="margin-top: 10px; font-size: 0.9em; color: #856404;">',
        '        Silakan perbaiki masalah di atas sebelum melanjutkan.',
        '    </div>',
        '</div>'
    ])
    
    return '\n'.join(content)

def _get_section_color(section: str) -> str:
    """Dapatkan warna untuk header bagian.
    
    Args:
        section: Nama bagian
        
    Returns:
        Kode warna untuk bagian tersebut
    """
    colors = {
        'Environment': '#2196f3',  # Biru
        'Storage': '#4caf50',     # Hijau
        'GPU': '#9c27b0',         # Ungu
        'Network': '#ff9800',     # Oranye
        'System': '#607d8b',      # Biru Abu
        'Verification': '#009688', # Hijau Kebiruan
        'Setup': '#673ab7',       # Ungu Tua
        'Status': '#03a9f4',      # Biru Muda
        'Komponen': '#ff5722',    # Jingga
        'default': '#9e9e9e'      # Abu-abu
    }
    return colors.get(section, colors['default'])

def _get_verification_status_icon(status: str) -> tuple:
    """Dapatkan ikon dan warna untuk status verifikasi.
    
    Args:
        status: Status verifikasi ('success', 'warning', 'error', dll)
        
    Returns:
        Tuple berisi (icon, color)
    """
    status = str(status).lower()
    
    if status in ['ok', 'success', 'true', 'yes', 'completed']:
        return '✅', '#4caf50'  # Hijau
    elif status in ['warning', 'warn', 'partial']:
        return '⚠️', '#ff9800'  # Kuning/oranye
    elif status in ['error', 'fail', 'failed', 'false', 'no']:
        return '❌', '#f44336'  # Merah
    elif status in ['info', 'information', 'note']:
        return 'ℹ️', '#2196f3'  # Biru
    elif status in ['pending', 'in_progress', 'running']:
        return '⏳', '#ffc107'  # Kuning
    else:
        return '🔹', '#9e9e9e'  # Abu-abu

def _format_summary_content(data: Dict) -> str:
    """Format data ringkasan untuk ditampilkan di widget (fungsi lama).
    
    Catatan:
        Fungsi ini dipertahankan untuk kompatibilitas ke belakang.
        Gunakan _format_enhanced_summary_content() untuk fungsionalitas yang lebih baru.
    
    Args:
        data: Kamus yang berisi data ringkasan
        
    Returns:
        String HTML yang sudah diformat
    """
    return _format_enhanced_summary_content(data)
