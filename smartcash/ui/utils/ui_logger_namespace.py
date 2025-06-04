"""
File: smartcash/ui/utils/ui_logger_namespace.py
Deskripsi: Fixed UI logger namespace untuk prevent duplicate formatting dan clean message output
"""

from typing import Dict, Any, Optional

# Daftar namespace yang diketahui dan ID unik mereka
KNOWN_NAMESPACES = {
    # Setup & Environment
    "smartcash.setup.dependency": "DEPS",
    "smartcash.ui.env_config": "ENV",
    "smartcash.setup.environment": "SETUP",
    
    # Dataset modules
    "smartcash.dataset.download": "DOWNLOAD",
    "smartcash.dataset.preprocessing": "PREPROC",
    "smartcash.dataset.augmentation": "AUGMENT",
    "smartcash.dataset.validation": "VALID",
    "smartcash.dataset.analysis": "ANALYZE",
    
    # Model modules
    "smartcash.model.training": "TRAIN",
    "smartcash.model.evaluation": "EVAL",
    "smartcash.model.inference": "INFER",
    "smartcash.model.export": "EXPORT",
    "smartcash.model.config": "CONFIG",
    
    # UI modules
    "smartcash.ui.main": "MAIN",
    "smartcash.ui.dashboard": "DASH",
    "smartcash.ui.components": "COMP",
    "smartcash.ui.handlers": "HAND",
    "smartcash.ui.observers": "OBS",
    
    # Common modules
    "smartcash.common.logger": "LOG",
    "smartcash.common.config": "CFG",
    "smartcash.common.utils": "UTILS",
    
    # Detection modules
    "smartcash.detection.service": "DETECT",
    "smartcash.detection.postprocess": "POST",
    
    # Pretrained model modules
    "smartcash.ui.pretrained_model": "PRETRAIN"
}

# Export konstanta untuk backward compatibility
DEPENDENCY_LOGGER_NAMESPACE = "smartcash.setup.dependency"
DOWNLOAD_LOGGER_NAMESPACE = "smartcash.dataset.download"
ENV_CONFIG_LOGGER_NAMESPACE = "smartcash.ui.env_config"
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"
TRAINING_LOGGER_NAMESPACE = "smartcash.model.training"
EVALUATION_LOGGER_NAMESPACE = "smartcash.model.evaluation"
PRETRAINED_MODEL_LOGGER_NAMESPACE = "smartcash.ui.pretrained_model"

# Color mapping untuk setiap namespace
NAMESPACE_COLORS = {
    "DEPS": "#FF6B6B",      # Red untuk dependency
    "ENV": "#4ECDC4",       # Teal untuk environment
    "SETUP": "#45B7D1",     # Blue untuk setup
    "DOWNLOAD": "#96CEB4",  # Green untuk download
    "PREPROC": "#FFEAA7",   # Yellow untuk preprocessing
    "AUGMENT": "#DDA0DD",   # Plum untuk augmentation
    "VALID": "#98D8C8",     # Mint untuk validation
    "ANALYZE": "#F7DC6F",   # Light yellow untuk analysis
    "TRAIN": "#85C1E9",     # Light blue untuk training
    "EVAL": "#F8C471",      # Orange untuk evaluation
    "INFER": "#BB8FCE",     # Purple untuk inference
    "EXPORT": "#82E0AA",    # Light green untuk export
    "CONFIG": "#F1948A",    # Pink untuk config
    "MAIN": "#AED6F1",      # Light blue untuk main
    "DASH": "#D5DBDB",      # Gray untuk dashboard
    "COMP": "#E8DAEF",      # Light purple untuk components
    "HAND": "#FADBD8",      # Light pink untuk handlers
    "OBS": "#D4EDDA",       # Light green untuk observers
    "LOG": "#FCF3CF",       # Light yellow untuk logger
    "CFG": "#EBDEF0",       # Light lavender untuk config
    "UTILS": "#E5E7E9",     # Light gray untuk utils
    "DETECT": "#ABEBC6",    # Light green untuk detection
    "POST": "#F9E79F",      # Light yellow untuk postprocess
    "PRETRAIN": "#D7BDE2"   # Light purple untuk pretrained model
}

def get_namespace_id(ui_components: Dict[str, Any]) -> Optional[str]:
    """
    Dapatkan ID namespace dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ID namespace atau None jika tidak ditemukan
    """
    # Cek namespace tersedia di ui_components
    namespace = ui_components.get('logger_namespace')
    if namespace:
        return KNOWN_NAMESPACES.get(namespace, namespace)
    
    # Cek flag spesifik yang disetel oleh inisializer
    flag_mapping = {
        'dependency_initialized': "smartcash.setup.dependency",
        'download_initialized': "smartcash.dataset.download",
        'env_config_initialized': "smartcash.ui.env_config",
        'augmentation_initialized': "smartcash.dataset.augmentation",
        'preprocessing_initialized': "smartcash.dataset.preprocessing",
        'training_initialized': "smartcash.model.training",
        'evaluation_initialized': "smartcash.model.evaluation",
        'main_ui_initialized': "smartcash.ui.main",
        'dashboard_initialized': "smartcash.ui.dashboard"
    }
    
    for flag, namespace in flag_mapping.items():
        if ui_components.get(flag, False):
            return KNOWN_NAMESPACES.get(namespace)
    
    return None

def get_namespace_color(namespace_id: str) -> str:
    """
    Dapatkan warna untuk namespace ID.
    
    Args:
        namespace_id: ID namespace
        
    Returns:
        Hex color string
    """
    return NAMESPACE_COLORS.get(namespace_id, "#6c757d")

def format_log_message(ui_components: Dict[str, Any], message: str, 
                      level: str = "info", timestamp: bool = False) -> str:
    """
    Format pesan log TANPA timestamp dan namespace untuk prevent duplicate.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan diformat (raw message saja)
        level: Level log (debug, info, success, warning, error, critical)
        timestamp: Selalu False untuk prevent duplicate timestamp
        
    Returns:
        Raw message tanpa formatting tambahan
    """
    # Return raw message saja - formatting akan dilakukan di UI layer
    return message.strip()

def format_log_message_html(ui_components: Dict[str, Any], message: str, 
                           level: str = "info", icon: str = None) -> str:
    """
    Format pesan log sebagai HTML dengan styling namespace yang clean.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan diformat
        level: Level log
        icon: Custom icon
        
    Returns:
        HTML string yang diformat
    """
    try:
        from smartcash.ui.utils.constants import COLORS
    except ImportError:
        COLORS = {
            "primary": "#007bff", "success": "#28a745", "warning": "#ffc107", 
            "danger": "#dc3545", "muted": "#6c757d", "text": "#212529"
        }
    
    # Map level ke emoji dan warna
    level_config = {
        "debug": {"emoji": "🔍", "color": COLORS.get("muted", "#6c757d")},
        "info": {"emoji": "ℹ️", "color": COLORS.get("primary", "#007bff")},
        "success": {"emoji": "✅", "color": COLORS.get("success", "#28a745")},
        "warning": {"emoji": "⚠️", "color": COLORS.get("warning", "#ffc107")},
        "error": {"emoji": "❌", "color": COLORS.get("danger", "#dc3545")},
        "critical": {"emoji": "🔥", "color": COLORS.get("danger", "#dc3545")}
    }
    
    config = level_config.get(level, level_config["info"])
    emoji = icon or config["emoji"]
    color = config["color"]
    
    # Clean message - remove any existing formatting
    clean_message = _clean_message(message)
    
    # Dapatkan namespace untuk styling
    namespace_id = get_namespace_id(ui_components)
    namespace_color = get_namespace_color(namespace_id) if namespace_id else color
    
    # HTML dengan styling yang clean
    html = f"""
    <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
               background-color:rgba(248,249,250,0.8);
               border-left:3px solid {namespace_color};">
        <span style="color:{color};margin-left:4px;">
            {emoji} {clean_message}
        </span>
    </div>
    """
    return html.strip()

def _clean_message(message: str) -> str:
    """
    Clean message dari duplicate formatting.
    
    Args:
        message: Raw message yang mungkin sudah terformat
        
    Returns:
        Clean message tanpa duplicate formatting
    """
    import re
    
    # Remove timestamp patterns [HH:MM:SS]
    message = re.sub(r'\[(\d{2}:\d{2}:\d{2})\]\s*', '', message)
    
    # Remove namespace patterns [NAMESPACE]
    message = re.sub(r'\[([A-Z]+)\]\s*', '', message)
    
    # Remove leading emoji + space jika ada
    message = re.sub(r'^[🔍ℹ️✅⚠️❌🔥]\s*', '', message)
    
    # Clean multiple spaces
    message = re.sub(r'\s+', ' ', message)
    
    return message.strip()

def create_namespace_badge(namespace_id: str) -> str:
    """
    Buat badge HTML untuk namespace.
    
    Args:
        namespace_id: ID namespace
        
    Returns:
        HTML badge string
    """
    color = get_namespace_color(namespace_id)
    
    return f"""
    <span style="background-color:{color};color:white;
                 padding:2px 6px;border-radius:3px;
                 font-size:0.75em;font-weight:bold;">
        {namespace_id}
    </span>
    """

def register_namespace(namespace: str, namespace_id: str, color: str = None) -> None:
    """
    Register namespace baru secara dinamis.
    
    Args:
        namespace: Full namespace string
        namespace_id: Short ID untuk namespace
        color: Optional hex color untuk namespace
    """
    KNOWN_NAMESPACES[namespace] = namespace_id
    
    if color:
        NAMESPACE_COLORS[namespace_id] = color

def get_all_namespaces() -> Dict[str, str]:
    """
    Dapatkan semua namespace yang terdaftar.
    
    Returns:
        Dictionary mapping namespace -> namespace_id
    """
    return KNOWN_NAMESPACES.copy()

def get_namespace_summary() -> Dict[str, Any]:
    """
    Dapatkan ringkasan semua namespace yang terdaftar.
    
    Returns:
        Dictionary dengan informasi namespace
    """
    return {
        'total_namespaces': len(KNOWN_NAMESPACES),
        'categories': {
            'setup': [ns for ns in KNOWN_NAMESPACES if 'setup' in ns or 'env' in ns],
            'dataset': [ns for ns in KNOWN_NAMESPACES if 'dataset' in ns],
            'model': [ns for ns in KNOWN_NAMESPACES if 'model' in ns],
            'ui': [ns for ns in KNOWN_NAMESPACES if 'ui' in ns],
            'common': [ns for ns in KNOWN_NAMESPACES if 'common' in ns],
            'detection': [ns for ns in KNOWN_NAMESPACES if 'detection' in ns]
        },
        'namespace_ids': list(KNOWN_NAMESPACES.values())
    }