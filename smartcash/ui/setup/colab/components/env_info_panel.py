"""
file_path: smartcash/ui/setup/colab/components/env_info_panel.py

Environment Information Panel for Google Colab.

Menampilkan informasi lengkap tentang environment Colab seperti:
- Informasi sistem dan runtime
- Penggunaan resource (CPU, RAM, GPU)
- Status penyimpanan dan akses
- Variabel environment penting
"""

from typing import Dict, Any, Optional, Union
import ipywidgets as widgets
from smartcash.ui.setup.colab.utils.env_detector import detect_environment_info

def create_env_info_panel(env_info: Optional[Dict[str, Any]] = None, lazy_load: bool = True) -> widgets.Widget:
    """Buat panel informasi environment Colab dengan opsi lazy loading.
    
    Args:
        env_info: Informasi environment yang sudah diambil sebelumnya.
        lazy_load: Jika True, akan menampilkan tombol untuk memuat environment.
                 Jika False, akan langsung memuat informasi environment.
                 
    Returns:
        Widget yang menampilkan informasi environment atau tombol untuk memuat
    """
    container = widgets.VBox(layout={'width': '100%'})
    
    def load_environment_info(button=None):
        try:
            with container.hold_trait_notifications():
                container.children = [widgets.HTML(
                    value='<div style="text-align: center; padding: 20px;">Memuat informasi environment...</div>',
                    layout={'width': '100%'}
                )]
                
                # Deteksi environment
                env_data = detect_environment_info()
                
                # Tampilkan informasi
                container.children = [widgets.HTML(
                    value=_format_env_info_content(env_data),
                    layout=widgets.Layout(
                        width='100%',
                        padding='15px',
                        border='1px solid #e0e0e0',
                        border_radius='6px',
                        margin='10px 0',
                        background='#f9f9f9',
                        overflow='hidden'
                    )
                )]
        except Exception as e:
            container.children = [widgets.HTML(
                value=f'<div style="color: #d32f2f; padding: 10px;">Gagal memuat informasi environment: {str(e)}</div>',
                layout={'width': '100%'}
            )]
    
    if env_info is not None and not lazy_load:
        # Jika sudah ada data environment dan tidak lazy load, tampilkan langsung
        container.children = [widgets.HTML(
            value=_format_env_info_content(env_info),
            layout=widgets.Layout(
                width='100%',
                padding='15px',
                border='1px solid #e0e0e0',
                border_radius='6px',
                margin='10px 0',
                background='#f9f9f9',
                overflow='hidden'
            )
        )]
    elif lazy_load:
        # Tampilkan tombol untuk memuat environment
        load_button = widgets.Button(
            description='🔍 Muat Informasi Environment',
            button_style='info',
            layout={'width': 'auto', 'margin': '10px 0'}
        )
        load_button.on_click(load_environment_info)
        
        container.children = [
            widgets.HTML(
                value='<div style="text-align: center; padding: 15px; border: 1px dashed #e0e0e0; border-radius: 6px; margin: 10px 0;">'
                     '<p>Informasi environment belum dimuat</p>',
                layout={'width': '100%'}
            ),
            load_button
        ]
    
    return container

def _format_env_info_content(env_info: Any) -> str:
    """Format informasi environment menjadi HTML.
    
    Args:
        env_info: Dictionary berisi informasi environment atau string error
        
    Returns:
        String HTML yang sudah diformat
    """
    if not isinstance(env_info, dict):
        return '''
        <div style="color: #d32f2f; padding: 15px; border: 1px solid #ffcdd2; border-radius: 6px; 
                   margin: 10px 0; background-color: #ffebee; font-family: monospace;">
            <strong>⚠️ Gagal memuat informasi environment:</strong><br>
            <div style="margin-top: 8px;">{}</div>
        </div>
        '''.format(str(env_info) if env_info else "Tidak ada informasi yang tersedia")
    
    try:
        # Dapatkan informasi runtime
        runtime_info = env_info.get('runtime', {}) or {}
        runtime_display = runtime_info.get('display', 'Environment Tidak Dikenal')
        
        # Dapatkan informasi sistem
        python_version = env_info.get('python_version', 'Tidak Tersedia')
        os_info = env_info.get('os', {}) or {}
        os_name = os_info.get('system', 'Tidak Tersedia')
        os_release = os_info.get('release', '')
        
        # Dapatkan informasi memori
        memory_info = env_info.get('memory_info', {}) or {}
        total_ram = memory_info.get('total_gb', 
            float(env_info.get('total_ram', 0)) / (1024**3) if env_info.get('total_ram') else 0
        )
        available_ram = memory_info.get('available_gb', 0)
        ram_usage = memory_info.get('percent_used', 0)
        
        # Dapatkan informasi resource
        cpu_cores = env_info.get('cpu_cores', 'Tidak Tersedia')
        
        # Dapatkan informasi penyimpanan
        storage_info = env_info.get('storage_info', {}) or {}
        storage_status = _format_enhanced_storage_info(storage_info)
        
        # Dapatkan informasi GPU
        gpu_data = env_info.get('gpu', {})
        gpu_info = _format_gpu_info(gpu_data)
        
        # Dapatkan status drive
        drive_status = _get_enhanced_drive_status(env_info)
        
        # Dapatkan informasi jaringan
        network_info = env_info.get('network_info', {}) or {}
        hostname = network_info.get('hostname', 'Tidak Tersedia')
        
        # Dapatkan status environment SmartCash
        env_vars = env_info.get('environment_variables', {}) or {}
        smartcash_status = _get_smartcash_env_status(env_vars)
        
        # Tentukan warna RAM usage
        ram_color = "#4caf50"
        ram_usage = float(ram_usage) if ram_usage else 0
        if ram_usage >= 90:
            ram_color = "#f44336"
        elif ram_usage >= 70:
            ram_color = "#ff9800"
        
        # Bangun HTML secara bertahap
        html_parts = [
            '<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">',
            '    <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">',
            f'        <h3 style="margin: 0; color: #333;">🌐 {runtime_display}</h3>',
            '    </div>',
            '    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">',
            '        <div>',
            '            <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #2196f3;">',
            '                🖥️ Informasi Sistem',
            '            </h4>',
            f'            <p><strong>Sistem Operasi:</strong> {os_name} {os_release}</p>',
            f'            <p><strong>Versi Python:</strong> {python_version}</p>',
            f'            <p><strong>Tipe Environment:</strong> {_get_env_type(env_info)}</p>',
            f'            <p><strong>Nama Host:</strong> {hostname}</p>',
            f'            <p><strong>Status SmartCash:</strong> {smartcash_status}</p>',
            '        </div>',
            '        <div>',
            '            <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #4caf50;">',
            '                💾 Penggunaan Resource',
            '            </h4>',
            f'            <p><strong>Core CPU:</strong> {cpu_cores}</p>',
            f'            <p><strong>Total RAM:</strong> {total_ram:.1f} GB</p>',
            f'            <p><strong>RAM Tersedia:</strong> {available_ram:.1f} GB</p>',
            '            <div style="margin: 5px 0 10px 0; background: #e0e0e0; border-radius: 3px; height: 20px; overflow: hidden;">',
            f'                <div style="height: 100%; width: {ram_usage}%; background: {ram_color}; color: white; text-align: center; line-height: 20px; font-size: 12px;"',
            f'                     title="Penggunaan RAM: {ram_usage:.1f}%">',
            f'                    {ram_usage:.1f}% Digunakan',
            '                </div>',
            '            </div>',
            f'            {gpu_info}',
            '        </div>',
            '        <div>',
            '            <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #9c27b0;">',
            '                💽 Informasi Penyimpanan',
            '            </h4>',
            f'            {storage_status}',
            f'            {drive_status}',
            '        </div>',
            '    </div>'
        ]
        
        # Tambahkan info tambahan jika diperlukan
        if env_info.get('show_additional_info', True):
            additional_info = _format_additional_info(env_info)
            if additional_info:
                html_parts.append(f'    {additional_info}')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
        
    except Exception as e:
        return f'<div style="color: #d32f2f; padding: 15px; background: #ffebee; border-radius: 4px; margin: 10px 0;">Error memformat konten environment: {str(e)}</div>'

def _format_gpu_info(gpu_info: Any) -> str:
    """Format informasi GPU untuk ditampilkan.
    
    Args:
        gpu_info: Dictionary berisi informasi GPU atau string error
        
    Returns:
        String informasi GPU yang sudah diformat
    """
    try:
        # Handle case where gpu_info is a string (error message)
        if not isinstance(gpu_info, dict):
            if isinstance(gpu_info, str) and gpu_info != 'No GPU available':
                return f'<p><em>{gpu_info}</em></p>'
            return '<p><em>Tidak terdeteksi GPU</em></p>'
        
        if not gpu_info or not gpu_info.get('available', False):
            return '<p><em>Tidak terdeteksi GPU</em></p>'
            
        gpu_name = gpu_info.get('name', 'GPU Tidak Dikenal')
        gpu_memory = gpu_info.get('memory', {})
        total_memory = gpu_memory.get('total_gb', 0)
        used_memory = gpu_memory.get('used_gb', 0)
        memory_usage = gpu_memory.get('percent_used', 0)
        
        return f"""
        <div style="margin-top: 10px;">
            <p><strong>GPU:</strong> {gpu_name}</p>
            <p><strong>VRAM:</strong> {used_memory:.1f} / {total_memory:.1f} GB</p>
            <div style="margin: 5px 0 10px 0; background: #e0e0e0; border-radius: 3px; height: 20px; overflow: hidden;">
                <div style="height: 100%; width: {memory_usage}%; background: {'#4caf50' if memory_usage < 70 else '#ff9800' if memory_usage < 90 else '#f44336'}; color: white; text-align: center; line-height: 20px; font-size: 12px;"
                     title="Penggunaan VRAM: {memory_usage:.1f}%">
                    {memory_usage:.1f}% Digunakan
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f'<p style="color: #d32f2f;">Gagal memuat info GPU: {str(e)}</p>'

def _format_storage_info(storage_info: Any) -> str:
    """Format informasi penyimpanan untuk ditampilkan.
    
    Args:
        storage_info: Dictionary berisi informasi penyimpanan atau string error
        
    Returns:
        String informasi penyimpanan yang sudah diformat
    """
    if not storage_info or isinstance(storage_info, str):
        return f'<p><em>{storage_info or "Informasi penyimpanan tidak tersedia"}</em></p>'
        
    try:
        total_gb = storage_info.get('total_gb', 0)
        used_gb = storage_info.get('used_gb', 0)
        free_gb = storage_info.get('free_gb', 0)
        usage_percent = (used_gb / total_gb * 100) if total_gb > 0 else 0
        
        return f"""
        <p><strong>Total:</strong> {total_gb:.1f} GB</p>
        <p><strong>Digunakan:</strong> {used_gb:.1f} GB</p>
        <p><strong>Tersisa:</strong> {free_gb:.1f} GB</p>
        <div style="margin: 5px 0 15px 0; background: #e0e0e0; border-radius: 3px; height: 20px; overflow: hidden;">
            <div style="height: 100%; width: {usage_percent}%; background: {'#4caf50' if usage_percent < 70 else '#ff9800' if usage_percent < 90 else '#f44336'}; color: white; text-align: center; line-height: 20px; font-size: 12px;"
                 title="Penggunaan penyimpanan: {usage_percent:.1f}%">
                {usage_percent:.1f}% Digunakan
            </div>
        </div>
        """
    except Exception as e:
        return f'<p style="color: #d32f2f;">Gagal memuat info penyimpanan: {str(e)}</p>'

def _get_drive_status(env_info: Dict[str, Any]) -> str:
    """Get formatted drive status.
    
    Args:
        env_info: Dictionary containing drive information
        
    Returns:
        Formatted drive status string
    """
    if env_info.get('drive_mounted'):
        mount_path = env_info.get('drive_mount_path', '')
        return f"✅ Mounted at {mount_path}" if mount_path else "✅ Mounted"
    return "❌ Not mounted"

def _get_env_type(env_info: Dict[str, Any]) -> str:
    """Tentukan tipe environment.
    
    Args:
        env_info: Dictionary berisi informasi environment
        
    Returns:
        String tipe environment
    """
    try:
        env_type = env_info.get('runtime', {}).get('type', 'Tidak Dikenal')
        env_map = {
            'colab': 'Google Colab',
            'jupyter': 'Jupyter Notebook',
            'local': 'Lokal',
            'kaggle': 'Kaggle',
            'sagemaker': 'Amazon SageMaker',
            'azureml': 'Azure ML',
            'vertex': 'Google Vertex AI'
        }
        return env_map.get(str(env_type).lower(), str(env_type))
    except Exception:
        return 'Tidak Dikenal'

def _format_enhanced_storage_info(storage_info: Dict[str, Any]) -> str:
    """Format informasi penyimpanan yang ditingkatkan untuk ditampilkan.
    
    Args:
        storage_info: Dictionary berisi informasi penyimpanan yang ditingkatkan
        
    Returns:
        String informasi penyimpanan yang sudah diformat
    """
    if not storage_info:
        return '<p><em>Informasi penyimpanan tidak tersedia</em></p>'
    
    try:
        total_gb = storage_info.get('total_gb', 0)
        used_gb = storage_info.get('used_gb', 0)
        free_gb = storage_info.get('free_gb', 0)
        percent_used = storage_info.get('percent_used', 0)
        
        if total_gb <= 0:
            return f'<p><strong>Digunakan:</strong> {used_gb:.1f} GB</p>'
        
        # Tentukan warna berdasarkan persentase penggunaan
        if percent_used >= 90:
            usage_color = "#f44336"  # Merah
        elif percent_used >= 70:
            usage_color = "#ff9800"  # Oranye
        else:
            usage_color = "#4caf50"  # Hijau
        
        # Bangun HTML secara bertahap
        html_parts = [
            '<div style="margin-bottom: 10px;">',
            f'    <p><strong>Total:</strong> {total_gb:.1f} GB</p>',
            f'    <p><strong>Digunakan:</strong> {used_gb:.1f} GB</p>',
            f'    <p><strong>Tersedia:</strong> {free_gb:.1f} GB</p>',
            '    <div style="margin: 5px 0; background: #e0e0e0; border-radius: 3px; height: 20px; overflow: hidden;">',
            f'        <div style="height: 100%; width: {percent_used}%; background: {usage_color}; color: white; text-align: center; line-height: 20px; font-size: 12px;"',
            f'             title="Penggunaan penyimpanan: {percent_used:.1f}%">',
            f'            {percent_used:.1f}% Digunakan',
            '        </div>',
            '    </div>',
            '</div>'
        ]
        
        return '\n'.join(html_parts)
        
    except Exception as e:
        return f'<p style="color: #d32f2f;">Gagal memuat info penyimpanan: {str(e)}</p>'

def _get_enhanced_drive_status(env_info: Dict[str, Any]) -> str:
    """Dapatkan status drive yang ditingkatkan dengan informasi akses tulis.
    
    Args:
        env_info: Dictionary berisi informasi drive
        
    Returns:
        String status drive yang sudah ditingkatkan dalam format HTML
    """
    if not env_info.get('is_colab', False):
        return '<span style="color: #666;">Tidak tersedia di luar Google Colab</span>'
        
    drive_mounted = env_info.get('drive_mounted')
    drive_path = env_info.get('drive_mount_path', '')
    drive_status = env_info.get('drive_status', 'unknown')
    
    # Handle case where drive status hasn't been checked yet
    if drive_status == 'not_checked' or drive_mounted is None:
        return ('<div style="color: #666; margin: 5px 0;">'
                '❓ Status drive belum diperiksa. Klik tombol "Muat Informasi Environment" untuk memeriksa.'
                '</div>')
                
    # Handle case where drive is not mounted
    if not drive_mounted:
        return ('<div style="color: #d32f2f; margin: 5px 0;">'
                '❌ Google Drive belum di-mount. Gunakan tombol "Mount Drive" untuk melanjutkan.'
                '</div>')
    
    # Handle case where drive is mounted
    try:
        import os
        test_file = os.path.join(drive_path, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        write_access = True
    except (IOError, OSError):
        write_access = False
    
    status_icon = '✅' if write_access else '⚠️'
    status_text = 'Akses tulis diizinkan' if write_access else 'Akses tulis ditolak - Periksa izin folder'
    status_color = '#28a745' if write_access else '#ff9800'
    
    return (f'<div style="margin: 5px 0;">'
            f'<p style="margin: 0 0 5px 0;">'
            f'<strong>📁 Google Drive:</strong> Terpasang di <code>{drive_path}</code>'
            f'</p>'
            f'<p style="margin: 0; color: {status_color};">'
            f'{status_icon} {status_text}'
            f'</p>'
            f'</div>')

    try:
        if not env_info.get('drive_mounted', False):
            return "<p>❌ <strong>Drive:</strong> Tidak terpasang</p>"
        
        mount_path = env_info.get('drive_mount_path', '/content/drive')
        drive_info = env_info.get('drive_info', {})
        
        # Mulai membangun HTML secara bertahap
        html_parts = [
            '<div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px;">',
            f'    <p style="margin: 0 0 10px 0; font-weight: 500;">',
            '        <span style="font-size: 1.1em;">📁 Drive</span>',
            f'        <span style="color: #666; font-size: 0.9em;">(Terpasang di {mount_path})</span>',
            '    </p>'
        ]
        
        # Tambahkan info ukuran jika tersedia
        if 'total_size' in drive_info and 'used_size' in drive_info:
            total_gb = drive_info['total_size'] / (1024**3)
            used_gb = drive_info['used_size'] / (1024**3)
            free_gb = max(0, total_gb - used_gb)  # Pastikan tidak negatif
            usage_percent = min(100, max(0, (used_gb / total_gb * 100) if total_gb > 0 else 0))
            
            # Tentukan warna berdasarkan persentase penggunaan
            if usage_percent >= 90:
                usage_color = "#f44336"  # Merah
            elif usage_percent >= 70:
                usage_color = "#ff9800"  # Oranye
            else:
                usage_color = "#4caf50"  # Hijau
            
            html_parts.extend([
                '    <div style="margin: 10px 0;">',
                '        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em;">',
                f'            <span>Kapasitas Penyimpanan</span>',
                f'            <span>{used_gb:.1f} GB / {total_gb:.1f} GB</span>',
                '        </div>',
                '        <div style="background: #e9ecef; border-radius: 3px; height: 20px; overflow: hidden;">',
                f'            <div style="height: 100%; width: {usage_percent:.1f}%; background: {usage_color}; color: white;',
                f'                     display: flex; align-items: center; justify-content: center; font-size: 12px;"',
                f'                 title="Penggunaan penyimpanan: {usage_percent:.1f}%">',
                f'                {usage_percent:.1f}%',
                '            </div>',
                '        </div>',
                '        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.85em; color: #666;">',
                f'            <span>Tersedia: {free_gb:.1f} GB</span>',
                f'            <span>Terpakai: {used_gb:.1f} GB</span>',
                '        </div>',
                '    </div>'
            ])
        
        # Tandai akses tulis jika tersedia
        if 'writable' in drive_info:
            status = '✅ Dapat menulis' if drive_info['writable'] else '❌ Hanya baca'
            status_style = 'color: #28a745;' if drive_info['writable'] else 'color: #dc3545;'
            html_parts.extend([
                '    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e0e0e0;">',
                f'        <p style="margin: 0; font-size: 0.95em;">',
                f'            <strong>Status Akses:</strong> <span style="{status_style}">{status}</span>',
                '        </p>',
                '    </div>'
            ])
        
        # Tutup div utama
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
        
    except Exception as e:
        error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
        return (
            '<div style="margin-top: 15px; padding: 10px; background: #f8d7da; border: 1px solid #f5c6cb; '
            'border-radius: 4px; color: #721c24;">'
            f'⚠️ <strong>Error:</strong> Gagal memuat informasi drive: {error_msg}'
            '</div>'
        )

def _get_smartcash_env_status(env_vars: Dict[str, str]) -> str:
    """Dapatkan status konfigurasi environment SmartCash.
    
    Fungsi ini memeriksa variabel environment yang diperlukan untuk SmartCash
    dan mengembalikan status konfigurasi dalam format yang mudah dibaca.
    
    Args:
        env_vars: Dictionary yang berisi variabel-variabel environment
        
    Returns:
        String HTML yang menampilkan status konfigurasi SmartCash
        
    Contoh:
        >>> env_vars = {
        ...     'SMARTCASH_ROOT': '/path/to/root',
        ...     'SMARTCASH_ENV': 'development',
        ...     'SMARTCASH_DATA_ROOT': '/path/to/data'
        ... }
        >>> _get_smartcash_env_status(env_vars)
        '<span style="color: #28a745;">✅ Development</span> (3/3 variabel terkonfigurasi)'
    """
    try:
        # Daftar variabel environment yang diperlukan
        required_vars = [
            'SMARTCASH_ROOT',      # Root direktori SmartCash
            'SMARTCASH_ENV',       # Environment (development/production/staging)
            'SMARTCASH_DATA_ROOT'  # Root direktori data
        ]
        
        # Hitung variabel yang sudah terkonfigurasi
        set_vars = [var for var in required_vars if var in env_vars and env_vars[var]]
        set_count = len(set_vars)
        total_count = len(required_vars)
        
        # Dapatkan tipe environment jika tersedia
        env_type = env_vars.get('SMARTCASH_ENV', 'tidak dikonfigurasi').capitalize()
        
        # Tentukan status berdasarkan kelengkapan konfigurasi
        if set_count == total_count:
            # Semua variabel terpenuhi
            status_icon = '✅'
            status_color = '#28a745'  # Hijau
            status_text = f"{env_type}"
            detail = f"({set_count}/{total_count} variabel terkonfigurasi)"
        elif set_count > 0:
            # Sebagian variabel terpenuhi
            status_icon = '⚠️'
            status_color = '#ffc107'  # Kuning
            status_text = f"Konfigurasi Parsial - {env_type}"
            missing_vars = [var for var in required_vars if var not in env_vars]
            detail = f"({set_count}/{total_count} variabel terkonfigurasi)"
        else:
            # Tidak ada variabel yang terpenuhi
            status_icon = '❌'
            status_color = '#dc3545'  # Merah
            status_text = "Tidak Terkonfigurasi"
            detail = "(0/3 variabel terkonfigurasi)"
        
        # Format output HTML
        return (
            f'<div style="margin-top: 5px;">'
            f'<span style="color: {status_color}; font-weight: 500;">'
            f'{status_icon} {status_text} <span style="font-size: 0.9em; color: #6c757d;">{detail}</span>'
            f'</span>'
            f'</div>'
        )
        
    except Exception as e:
        error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
        return (
            '<div style="color: #dc3545; margin-top: 5px;">'
            f'⚠️ <strong>Error:</strong> Gagal memeriksa status SmartCash: {error_msg}'
            '</div>'
        )

def _format_additional_info(env_info: Any) -> str:
    """Format dan tampilkan informasi tambahan tentang environment.
    
    Fungsi ini mengumpulkan dan memformat berbagai informasi tambahan
    seperti jaringan, CUDA, runtime, dan informasi Python ke dalam
    tampilan HTML yang rapi dan informatif.
    
    Args:
        env_info: Dictionary berisi informasi lengkap environment atau string error
        
    Returns:
        String HTML yang berisi informasi tambahan yang diformat
        atau string kosong jika tidak ada informasi tambahan yang tersedia.
    """
    try:
        # Handle case where env_info is not a dictionary
        if not isinstance(env_info, dict):
            return ''  # Return empty string if not a valid dict
            
        additional_info = []
        
        # 1. Informasi Jaringan
        network_info = env_info.get('network_info', {})
        if not isinstance(network_info, dict):
            network_info = {}
        interfaces = network_info.get('interfaces', [])
        if interfaces and isinstance(interfaces, list):
            interface_count = len(interfaces)
            active_interfaces = [i for i in interfaces if i.get('is_up', False)]
            active_count = len(active_interfaces)
            
            # Dapatkan alamat IP dari interface aktif pertama (jika ada)
            ip_address = "Tidak terdeteksi"
            if active_interfaces:
                addresses = active_interfaces[0].get('addresses', [])
                if addresses:
                    ip_address = addresses[0].get('addr', 'Tidak terdeteksi')
            
            network_html = [
                '<div style="margin-bottom: 10px;">',
                '    <div style="font-weight: 500; margin-bottom: 5px;">🌐 Jaringan</div>',
                '    <div style="margin-left: 15px; font-size: 0.9em;">',
                f'        <div>• <strong>Status:</strong> {active_count} dari {interface_count} antarmuka aktif</div>',
                f'        <div>• <strong>Alamat IP:</strong> {ip_address}</div>',
                '    </div>',
                '</div>'
            ]
            additional_info.append('\n'.join(network_html))
        
        # 2. Informasi CUDA dan GPU
        cuda_version = env_info.get('cuda_version')
        gpu_info = env_info.get('gpu', {})
        if not isinstance(gpu_info, dict):
            gpu_info = {}
        
        if cuda_version or gpu_info.get('available', False):
            gpu_html = [
                '<div style="margin: 15px 0 10px 0;">',
                '    <div style="font-weight: 500; margin-bottom: 5px;">🎮 GPU & CUDA</div>',
                '    <div style="margin-left: 15px; font-size: 0.9em;">'
            ]
            
            if cuda_version:
                gpu_html.append(f'<div>• <strong>CUDA Version:</strong> {cuda_version}</div>')
            
            if gpu_info.get('available', False):
                gpu_name = gpu_info.get('name', 'Tidak Dikenal')
                gpu_memory = gpu_info.get('memory', {})
                memory_used = gpu_memory.get('used_gb', 0)
                memory_total = gpu_memory.get('total_gb', 0)
                memory_percent = gpu_memory.get('percent_used', 0)
                
                gpu_html.extend([
                    f'<div>• <strong>GPU:</strong> {gpu_name}</div>',
                    f'<div>• <strong>VRAM:</strong> {memory_used:.1f} / {memory_total:.1f} GB ({memory_percent:.1f}%)</div>'
                ])
            
            gpu_html.extend([
                '    </div>',
                '</div>'
            ])
            additional_info.append('\n'.join(gpu_html))
        
        # 3. Informasi Runtime
        runtime_info = env_info.get('runtime', {})
        if not isinstance(runtime_info, dict):
            runtime_info = {}
        if 'start_time' in runtime_info:
            try:
                from datetime import datetime
                start_time = datetime.fromisoformat(runtime_info['start_time'].replace('Z', '+00:00'))
                uptime = datetime.now() - start_time
                
                # Format durasi
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                uptime_parts = []
                if days > 0:
                    uptime_parts.append(f"{days} hari")
                if hours > 0 or days > 0:
                    uptime_parts.append(f"{hours} jam")
                uptime_parts.append(f"{minutes} menit")
                
                runtime_html = [
                    '<div style="margin: 10px 0;">',
                    '    <div style="font-weight: 500; margin-bottom: 5px;">⏱️ Runtime</div>',
                    '    <div style="margin-left: 15px; font-size: 0.9em;">',
                    f'        <div>• <strong>Dimulai pada:</strong> {start_time.strftime("%d %b %Y, %H:%M:%S")}</div>',
                    f'        <div>• <strong>Uptime:</strong> {", ".join(uptime_parts)}</div>',
                    '    </div>',
                    '</div>'
                ]
                additional_info.append('\n'.join(runtime_html))
            except (ValueError, TypeError, AttributeError) as e:
                # Jika terjadi kesalahan parsing waktu, abaikan bagian ini
                pass
        
        # 4. Informasi Python
        python_info = env_info.get('python_info', {})
        if not isinstance(python_info, dict):
            python_info = {}
        if python_info:
            implementation = python_info.get('implementation', 'Python')
            version = python_info.get('version', '')
            executable = python_info.get('executable', '')
            
            python_html = [
                '<div style="margin: 10px 0 5px 0;">',
                '    <div style="font-weight: 500; margin-bottom: 5px;">🐍 Python</div>',
                '    <div style="margin-left: 15px; font-size: 0.9em;">',
                f'        <div>• <strong>Versi:</strong> {implementation} {version}</div>'
            ]
            
            if executable:
                python_html.append(f'        <div>• <strong>Lokasi:</strong> <code style="font-size: 0.85em;">{executable}</code></div>')
            
            python_html.extend([
                '    </div>',
                '</div>'
            ])
            additional_info.append('\n'.join(python_html))
        
        # Jika tidak ada informasi tambahan, kembalikan string kosong
        if not additional_info:
            return ''
        
        # Gabungkan semua bagian informasi tambahan
        info_content = '\n'.join(additional_info)
        return f'''
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; border: 1px solid #e0e0e0;">
            <div style="display: flex; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #dee2e6;">
                <span style="font-size: 1.1em; font-weight: 600; color: #495057;">ℹ️ Informasi Tambahan</span>
            </div>
            {info_content}
        </div>
        '''
        
    except Exception as e:
        error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
        return (
            '<div style="margin-top: 15px; padding: 10px; background: #f8d7da; border: 1px solid #f5c6cb; '
            'border-radius: 4px; color: #721c24;">'
            f'⚠️ <strong>Error:</strong> Gagal memuat informasi tambahan: {error_msg}'
            '</div>'
        )
