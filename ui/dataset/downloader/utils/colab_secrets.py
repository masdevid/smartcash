"""
File: smartcash/ui/dataset/downloader/utils/colab_secrets.py
Deskripsi: Utility untuk mengambil API key dari Google Colab Secrets dengan fallback
"""

from typing import Optional, Dict, Any

def get_api_key_from_secrets(secret_names: list = None) -> Optional[str]:
    """
    Ambil API key dari Google Colab Secrets dengan multiple fallback names.
    
    Args:
        secret_names: List nama secret yang akan dicoba
        
    Returns:
        API key string atau None jika tidak ditemukan
    """
    # Default secret names untuk dicoba
    if secret_names is None:
        secret_names = [
            'ROBOFLOW_API_KEY',
            'roboflow_api_key', 
            'API_KEY',
            'api_key',
            'SMARTCASH_API_KEY',
            'smartcash_api_key'
        ]
    
    try:
        # Import Google Colab userdata
        from google.colab import userdata
        
        # Coba setiap secret name
        for secret_name in secret_names:
            try:
                api_key = userdata.get(secret_name)
                if api_key and api_key.strip():
                    return api_key.strip()
            except Exception:
                continue
        
        return None
        
    except ImportError:
        # Tidak di Colab environment
        return None
    except Exception:
        # Error lainnya
        return None

def validate_api_key(api_key: str) -> Dict[str, Any]:
    """
    Validate format API key dengan one-liner checks.
    
    Args:
        api_key: API key yang akan divalidate
        
    Returns:
        Dictionary dengan validation result
    """
    if not api_key or not api_key.strip():
        return {'valid': False, 'message': 'API key kosong'}
    
    key = api_key.strip()
    
    # Basic format validation
    validation_checks = [
        (len(key) >= 10, 'API key terlalu pendek (minimal 10 karakter)'),
        (len(key) <= 200, 'API key terlalu panjang (maksimal 200 karakter)'),
        (not any(char.isspace() for char in key), 'API key tidak boleh mengandung spasi'),
        (key.replace('-', '').replace('_', '').isalnum(), 'API key hanya boleh alphanumeric, dash, dan underscore')
    ]
    
    for check, message in validation_checks:
        if not check:
            return {'valid': False, 'message': message}
    
    return {'valid': True, 'message': 'Format API key valid'}

def get_available_secrets() -> Dict[str, Any]:
    """
    Get daftar secrets yang tersedia di Colab (tanpa value).
    
    Returns:
        Dictionary dengan informasi secrets
    """
    try:
        from google.colab import userdata
        
        # Coba detect secrets dengan common names
        common_secrets = [
            'ROBOFLOW_API_KEY', 'roboflow_api_key', 'API_KEY', 'api_key',
            'SMARTCASH_API_KEY', 'smartcash_api_key', 'GITHUB_TOKEN', 'HF_TOKEN'
        ]
        
        available = []
        for secret in common_secrets:
            try:
                value = userdata.get(secret)
                if value:
                    available.append({
                        'name': secret,
                        'length': len(value),
                        'preview': f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
                    })
            except Exception:
                continue
        
        return {
            'colab_available': True,
            'secrets_found': len(available),
            'secrets': available
        }
        
    except ImportError:
        return {
            'colab_available': False,
            'message': 'Tidak berjalan di Google Colab'
        }
    except Exception as e:
        return {
            'colab_available': True,
            'error': str(e),
            'message': 'Error mengakses Colab secrets'
        }

def set_api_key_to_config(config: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
    """
    Set API key ke config dengan automatic detection dan validation.
    
    Args:
        config: Config dictionary
        force_refresh: Force refresh dari secrets meskipun sudah ada
        
    Returns:
        Updated config dengan API key info
    """
    current_key = config.get('api_key', '').strip()
    
    # Skip jika sudah ada key dan tidak force refresh
    if current_key and not force_refresh:
        validation = validate_api_key(current_key)
        config['_api_key_source'] = 'existing'
        config['_api_key_valid'] = validation['valid']
        return config
    
    # Coba ambil dari secrets
    secret_key = get_api_key_from_secrets()
    
    if secret_key:
        validation = validate_api_key(secret_key)
        config['api_key'] = secret_key
        config['_api_key_source'] = 'colab_secret'
        config['_api_key_valid'] = validation['valid']
        config['_api_key_message'] = validation['message']
    else:
        # Fallback ke manual input
        config['_api_key_source'] = 'manual_required'
        config['_api_key_valid'] = False
        config['_api_key_message'] = 'API key tidak ditemukan di Colab Secret'
        
        if not current_key:
            config['api_key'] = ''
    
    return config

def create_api_key_info_html(config: Dict[str, Any]) -> str:
    """
    Create HTML info untuk API key status dengan one-liner formatting.
    
    Args:
        config: Config dengan API key info
        
    Returns:
        HTML string untuk display
    """
    source = config.get('_api_key_source', 'unknown')
    valid = config.get('_api_key_valid', False)
    message = config.get('_api_key_message', '')
    
    source_icons = {
        'colab_secret': '🔑 Colab Secret',
        'existing': '📝 Manual Input', 
        'manual_required': '⚠️ Input Required',
        'unknown': '❓ Unknown'
    }
    
    status_colors = {
        'colab_secret': '#28a745',
        'existing': '#007bff',
        'manual_required': '#ffc107',
        'unknown': '#6c757d'
    }
    
    source_text = source_icons.get(source, source)
    color = status_colors.get(source, '#6c757d')
    
    return f"""
    <div style="padding: 8px 12px; background: rgba(248, 249, 250, 0.8); 
                border-left: 3px solid {color}; border-radius: 4px; margin: 5px 0;">
        <small style="color: {color}; font-weight: 500;">
            {source_text} {'✅' if valid else '❌'} {message}
        </small>
    </div>
    """

# One-liner utilities
is_colab_environment = lambda: get_available_secrets()['colab_available']
has_roboflow_secret = lambda: any('roboflow' in s['name'].lower() for s in get_available_secrets().get('secrets', []))
get_secret_count = lambda: get_available_secrets().get('secrets_found', 0)


def update_config_with_api_key(config: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Helper function untuk memperbarui config dengan API key.
    
    Args:
        config: Config dictionary yang akan diperbarui
        api_key: API key yang akan dimasukkan ke config
        
    Returns:
        Config dictionary yang sudah diperbarui dengan API key
    """
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    if 'roboflow' not in config['data']:
        config['data']['roboflow'] = {}
    config['data']['roboflow']['api_key'] = api_key
    return config


def get_api_key_from_ui(ui_components: Dict[str, Any]) -> str:
    """
    Helper function untuk mendapatkan API key dari UI components.
    
    Args:
        ui_components: Dictionary UI components yang berisi api_key_input
        
    Returns:
        API key string atau empty string jika tidak ditemukan
    """
    api_key = ''
    api_key_widget = ui_components.get('api_key_input')
    if api_key_widget and hasattr(api_key_widget, 'value'):
        api_key = api_key_widget.value.strip()
    return api_key


def detect_and_set_api_key(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Helper function untuk mendeteksi dan mengatur API key dari berbagai sumber.
    
    Prioritas:
    1. UI components (jika disediakan)
    2. Colab secrets
    3. Konfigurasi yang ada
    
    Args:
        config: Config dictionary yang akan diperbarui
        ui_components: Optional UI components untuk mengambil API key
        
    Returns:
        Config dictionary yang sudah diperbarui dengan API key
    """
    # Coba dapatkan dari UI jika disediakan
    api_key = ''
    if ui_components:
        api_key = get_api_key_from_ui(ui_components)
    
    # Jika tidak ada di UI, coba dari Colab secrets
    if not api_key:
        detected_key = get_api_key_from_secrets()
        if detected_key:
            api_key = detected_key
    
    # Jika API key ditemukan, perbarui config
    if api_key:
        config = update_config_with_api_key(config, api_key)
    else:
        # Coba auto-deteksi dari Colab secrets sebagai fallback
        try:
            config = set_api_key_to_config(config, force_refresh=False)
        except Exception:
            pass
    
    return config