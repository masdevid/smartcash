"""
File: smartcash/ui/utils/ui_logger_namespace.py
Deskripsi: Fixed UI logger namespace untuk prevent duplicate formatting dan clean message output
"""

from typing import Dict, Any, Optional

# ENHANCED KNOWN_NAMESPACES dengan UUID support
KNOWN_NAMESPACES = {
    # Setup & Environment (existing)
    "smartcash.ui.setup.dependency_installer": "DEPS",
    "smartcash.ui.setup.env_config": "ENV",
    "smartcash.ui.setup": "SETUP",
    # Dataset modules (existing + enhanced)
    "smartcash.dataset.download": "DOWNLOAD",
    "smartcash.dataset.preprocessing": "PREPROC",
    "smartcash.dataset.augmentation": "AUGMENT",
    "smartcash.dataset.validation": "VALID",
    "smartcash.dataset.analysis": "ANALYZE",
    
    # NEW: Enhanced dataset modules dengan UUID support
    "smartcash.dataset.downloader": "DL_ENH",
    "smartcash.dataset.downloader.enhanced": "DL_ENH",
    "smartcash.dataset.file_renamer": "RENAME",
    "smartcash.dataset.organizer.enhanced": "ORG_ENH",
    "smartcash.common.file_naming": "NAMING",
    "smartcash.common.threadpools": "THREAD",
    
    # Model modules (existing)
    "smartcash.model.training": "TRAIN",
    "smartcash.model.evaluation": "EVAL",
    "smartcash.model.inference": "INFER",
    "smartcash.model.export": "EXPORT",
    "smartcash.model.config": "CONFIG",
    
    # UI modules (existing)
    "smartcash.ui.main": "MAIN",
    "smartcash.ui.dashboard": "DASH",
    "smartcash.ui.components": "COMP",
    "smartcash.ui.handlers": "HAND",
    "smartcash.ui.observers": "OBS",
    
    # Common modules (existing + enhanced)
    "smartcash.common.logger": "LOG",
    "smartcash.common.config": "CFG",
    "smartcash.common.utils": "UTILS",
    
    # Detection modules (existing)
    "smartcash.detection.service": "DETECT",
    "smartcash.detection.postprocess": "POST",
    
    # Pretrained model modules (existing)
    "smartcash.ui.pretrained_model": "PRETRAIN"
}

# ENHANCED NAMESPACE_COLORS dengan UUID modules
NAMESPACE_COLORS = {
    # Existing colors tetap sama...
    "DEPS": "#FF6B6B",      # Red untuk dependency
    "ENV": "#4ECDC4",       # Teal untuk environment
    "SETUP": "#45B7D1",     # Blue untuk setup
    "DOWNLOAD": "#96CEB4",  # Green untuk download
    "PREPROC": "#FFEAA7",   # Yellow untuk preprocessing
    "AUGMENT": "#DDA0DD",   # Plum untuk augmentation
    "VALID": "#98D8C8",     # Mint untuk validation
    "ANALYZE": "#F7DC6F",   # Light yellow untuk analysis
    
    # NEW: Enhanced dataset colors
    "DL_ENH": "#2ECC71",    # Green untuk enhanced downloader
    "RENAME": "#E74C3C",    # Red untuk renaming operations
    "ORG_ENH": "#9B59B6",   # Purple untuk enhanced organizer
    "NAMING": "#3498DB",    # Blue untuk naming operations
    "THREAD": "#F39C12",    # Orange untuk threading operations
    
    # Existing model colors tetap sama...
    "TRAIN": "#85C1E9",     # Light blue untuk training
    "EVAL": "#F8C471",      # Orange untuk evaluation
    "INFER": "#BB8FCE",     # Purple untuk inference
    "EXPORT": "#82E0AA",    # Light green untuk export
    "CONFIG": "#F1948A",    # Pink untuk config
    
    # Existing UI colors tetap sama...
    "MAIN": "#AED6F1",      # Light blue untuk main
    "DASH": "#D5DBDB",      # Gray untuk dashboard
    "COMP": "#E8DAEF",      # Light purple untuk components
    "HAND": "#FADBD8",      # Light pink untuk handlers
    "OBS": "#D4EDDA",       # Light green untuk observers
    
    # Existing common colors tetap sama...
    "LOG": "#FCF3CF",       # Light yellow untuk logger
    "CFG": "#EBDEF0",       # Light lavender untuk config
    "UTILS": "#E5E7E9",     # Light gray untuk utils
    
    # Existing detection colors tetap sama...
    "DETECT": "#ABEBC6",    # Light green untuk detection
    "POST": "#F9E79F",      # Light yellow untuk postprocess
    "PRETRAIN": "#D7BDE2"   # Light purple untuk pretrained model
}

# ENHANCED namespace color helper function
def get_namespace_color(namespace_id: str) -> str:
    """Mendapatkan warna untuk namespace ID yang diberikan"""
    if not namespace_id:
        return "#007bff"  # Default blue
    
    return NAMESPACE_COLORS.get(namespace_id, "#007bff")

# ENHANCED flag mapping dengan UUID support
def get_namespace_id(ui_components: Dict[str, Any]) -> Optional[str]:
    """Enhanced namespace ID detection dengan UUID support"""
    # Cek namespace tersedia di ui_components
    namespace = ui_components.get('logger_namespace')
    if namespace:
        return KNOWN_NAMESPACES.get(namespace, namespace)
    
    # Enhanced flag mapping dengan UUID support
    flag_mapping = {
        # Existing flags
        'dependency_installer_initialized': "smartcash.setup.dependency_installer",
        'download_initialized': "smartcash.dataset.download",
        'env_config_initialized': "smartcash.ui.env_config",
        'augmentation_initialized': "smartcash.dataset.augmentation",
        'preprocessing_initialized': "smartcash.dataset.preprocessing",
        'training_initialized': "smartcash.model.training",
        'evaluation_initialized': "smartcash.model.evaluation",
        'main_ui_initialized': "smartcash.ui.main",
        'dashboard_initialized': "smartcash.ui.dashboard",
        
        # NEW: Enhanced flags untuk UUID support
        'file_renamer_initialized': "smartcash.dataset.file_renamer",
        'organizer_initialized': "smartcash.dataset.organizer",
        'uuid_naming_initialized': "smartcash.common.file_naming",
        'threading_initialized': "smartcash.common.threadpools"
    }
    
    for flag, namespace in flag_mapping.items():
        if ui_components.get(flag, False):
            return KNOWN_NAMESPACES.get(namespace)
    
    return None

# ENHANCED format_log_message_html dengan UUID context awareness
def format_log_message_html(ui_components: Dict[str, Any], message: str, 
                           level: str = "info", icon: str = None) -> str:
    """Enhanced HTML formatting dengan UUID operation context"""
    try:
        from smartcash.ui.utils.constants import COLORS
    except ImportError:
        COLORS = {
            "primary": "#007bff", "success": "#28a745", "warning": "#ffc107", 
            "danger": "#dc3545", "muted": "#6c757d", "text": "#212529"
        }
    
    # Enhanced level config dengan UUID operations
    level_config = {
        "debug": {"emoji": "🔍", "color": COLORS.get("muted", "#6c757d")},
        "info": {"emoji": "ℹ️", "color": COLORS.get("primary", "#007bff")},
        "success": {"emoji": "✅", "color": COLORS.get("success", "#28a745")},
        "warning": {"emoji": "⚠️", "color": COLORS.get("warning", "#ffc107")},
        "error": {"emoji": "❌", "color": COLORS.get("danger", "#dc3545")},
        "critical": {"emoji": "🔥", "color": COLORS.get("danger", "#dc3545")},
        
        # NEW: UUID-specific levels
        "uuid_info": {"emoji": "🔤", "color": "#3498DB"},
        "rename_progress": {"emoji": "🔄", "color": "#E74C3C"},
        "threading": {"emoji": "⚡", "color": "#F39C12"}
    }
    
    config = level_config.get(level, level_config["info"])
    emoji = icon or config["emoji"]
    color = config["color"]
    
    # Enhanced message cleaning dengan UUID patterns
    clean_message = _clean_message_enhanced(message)
    
    # Enhanced namespace untuk UUID operations
    namespace_id = get_namespace_id(ui_components)
    namespace_color = get_namespace_color(namespace_id) if namespace_id else color
    
    # Enhanced HTML dengan UUID operation indicators
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

def _clean_message_enhanced(message: str) -> str:
    """Enhanced message cleaning dengan UUID patterns"""
    import re
    
    # Remove timestamp patterns [HH:MM:SS]
    message = re.sub(r'\[(\d{2}:\d{2}:\d{2})\]\s*', '', message)
    
    # Remove namespace patterns [NAMESPACE]
    message = re.sub(r'\[([A-Z]+)\]\s*', '', message)
    
    # Remove leading emoji + space jika ada
    message = re.sub(r'^[🔍ℹ️✅⚠️❌🔥🔤🔄⚡]\s*', '', message)
    
    # NEW: Clean UUID patterns untuk readability
    message = re.sub(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', '[UUID]', message)
    
    # Clean multiple spaces
    message = re.sub(r'\s+', ' ', message)
    
    return message.strip()

# NEW: UUID-specific helper functions
def register_uuid_namespace(module_name: str, namespace_id: str, color: str = None) -> None:
    """Register UUID-related namespace dynamically"""
    full_namespace = f"smartcash.{module_name}" if not module_name.startswith('smartcash') else module_name
    KNOWN_NAMESPACES[full_namespace] = namespace_id
    
    if color:
        NAMESPACE_COLORS[namespace_id] = color
    
    # Auto-register common UUID operation namespaces
    uuid_variants = [f"{full_namespace}.uuid", f"{full_namespace}.renaming", f"{full_namespace}.enhanced"]
    for variant in uuid_variants:
        KNOWN_NAMESPACES[variant] = namespace_id

def get_uuid_operation_context(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get context information untuk UUID operations"""
    uuid_flags = [flag for flag in ui_components.keys() 
                 if any(uuid_term in flag.lower() for uuid_term in ['uuid', 'rename', 'naming', 'enhanced'])]
    
    return {
        'has_uuid_operations': bool(uuid_flags),
        'uuid_flags': uuid_flags,
        'namespace_id': get_namespace_id(ui_components),
        'is_enhanced_mode': any('enhanced' in flag for flag in uuid_flags)
    }

def format_uuid_progress_message(operation: str, current: int, total: int, additional_info: str = "") -> str:
    """Format progress message untuk UUID operations"""
    operation_emojis = {
        'discover': '🔍', 'backup': '💾', 'rename': '🔄', 
        'validate': '✅', 'cleanup': '🧹', 'organize': '📁'
    }
    
    emoji = operation_emojis.get(operation.lower(), '⚡')
    progress_text = f"{current}/{total}" if total > 0 else "Processing"
    
    message = f"{emoji} {operation.title()}: {progress_text}"
    if additional_info:
        message += f" - {additional_info}"
    
    return message

# Enhanced namespace summary dengan UUID support
def get_namespace_summary() -> Dict[str, Any]:
    """Enhanced namespace summary dengan UUID operations"""
    base_summary = {
        'total_namespaces': len(KNOWN_NAMESPACES),
        'categories': {
            'setup': [ns for ns in KNOWN_NAMESPACES if 'setup' in ns or 'env' in ns],
            'dataset': [ns for ns in KNOWN_NAMESPACES if 'dataset' in ns],
            'model': [ns for ns in KNOWN_NAMESPACES if 'model' in ns],
            'ui': [ns for ns in KNOWN_NAMESPACES if 'ui' in ns],
            'common': [ns for ns in KNOWN_NAMESPACES if 'common' in ns],
            'detection': [ns for ns in KNOWN_NAMESPACES if 'detection' in ns],
            
            # NEW: UUID-related categories
            'uuid_operations': [ns for ns in KNOWN_NAMESPACES if any(term in ns.lower() for term in ['uuid', 'rename', 'naming', 'enhanced'])],
            'file_operations': [ns for ns in KNOWN_NAMESPACES if any(term in ns.lower() for term in ['file', 'organizer', 'downloader'])],
            'threading': [ns for ns in KNOWN_NAMESPACES if 'thread' in ns.lower()]
        },
        'namespace_ids': list(KNOWN_NAMESPACES.values())
    }
    
    # Add UUID-specific statistics
    uuid_count = len(base_summary['categories']['uuid_operations'])
    base_summary['uuid_support'] = {
        'enabled': uuid_count > 0,
        'uuid_namespaces': uuid_count,
        'enhanced_modules': len([ns for ns in KNOWN_NAMESPACES if 'enhanced' in ns])
    }
    
    return base_summary

# Backward compatibility constants untuk module-specific namespaces
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"
DEPENDENCY_LOGGER_NAMESPACE = "smartcash.ui.setup.dependency"
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"
EVALUATION_LOGGER_NAMESPACE = "smartcash.model.evaluation"
DOWNLOAD_LOGGER_NAMESPACE = "smartcash.dataset.download"
TRAINING_LOGGER_NAMESPACE = "smartcash.model.training"
ENV_CONFIG_LOGGER_NAMESPACE = "smartcash.ui.setup.env_config"

# Export constants for backward compatibility
__all__ = [
    # Existing exports
    'KNOWN_NAMESPACES', 'NAMESPACE_COLORS', 'get_namespace_id', 'get_namespace_color',
    'format_log_message', 'format_log_message_html', 'create_namespace_badge',
    'register_namespace', 'get_all_namespaces', 'get_namespace_summary',
    
    # NEW: UUID-specific exports
    'register_uuid_namespace', 'get_uuid_operation_context', 'format_uuid_progress_message',
    
    # Backward compatibility constants
    'PREPROCESSING_LOGGER_NAMESPACE', 'DEPENDENCY_LOGGER_NAMESPACE', 'AUGMENTATION_LOGGER_NAMESPACE',
    'EVALUATION_LOGGER_NAMESPACE', 'DOWNLOAD_LOGGER_NAMESPACE', 'TRAINING_LOGGER_NAMESPACE',
    'ENV_CONFIG_LOGGER_NAMESPACE'
]