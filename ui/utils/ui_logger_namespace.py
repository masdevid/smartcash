"""
File: smartcash/ui/utils/ui_logger_namespace.py
Deskripsi: Fixed UI logger namespace untuk prevent duplicate formatting dan clean message output
"""

from typing import Dict, Any, Optional
# KNOWN_NAMESPACES dengan UUID support
KNOWN_NAMESPACES = {
    # Setup & Environment
    "smartcash.ui.setup.env_config": "ENV",
    "smartcash.ui.setup.dependency": "DEPS",
    "smartcash.ui.setup": "SETUP",
    
    # Dataset related
    "smartcash.ui.dataset.downloader": "DOWNLOAD",
    "smartcash.ui.dataset.split": "SPLIT",
    "smartcash.ui.dataset.preprocessing": "PREPROC",
    "smartcash.ui.dataset.augmentation": "AUGMENT",
    "smartcash.ui.pretrained_model": "PRETRAIN",
    "smartcash.ui.backbone": "BACKBONE",
    "smartcash.ui.hyperparameters": "HYPER",
    "smartcash.ui.strategy": "STRATEGY",
    "smartcash.ui.training": "TRAINING",
    "smartcash.ui.evaluation": "EVAL",

    "smartcash.dataset.preprocessing": "DATASET_PREPROC",
    "smartcash.dataset.augmentation": "DATASET_AUGMENT",
    "smartcash.dataset.validation": "DATASET_VALID",
    "smartcash.dataset.analysis": "DATASET_ANALYZE",
    "smartcash.dataset.downloader": "DATASET_DOWNLOADER",
    "smartcash.dataset.file_renamer": "DATASET_RENAME",
    "smartcash.dataset.organizer": "DATASET_ORG",
    # Model related
    "smartcash.model.training": "MOD_TRAIN",
    "smartcash.model.evaluation": "MOD_EVAL",
    "smartcash.model.inference": "MOD_INFER",
    "smartcash.model.export": "MOD_EXPORT",
    "smartcash.model.config": "MOD_CONFIG",
    
    # Common utilities
    "smartcash.common.file_naming": "NAMING",
    "smartcash.common.threadpools": "THREAD",
    "smartcash.common.logger": "LOG",
    "smartcash.common.config": "CFG",
    "smartcash.common.utils": "UTILS",
    
    # Detection
    "smartcash.detection.service": "DETECT",
    "smartcash.detection.postprocess": "POST"
}

# ENHANCED NAMESPACE_COLORS dengan UUID modules
NAMESPACE_COLORS = {
    # Setup & Environment colors
    "DEPS": "#FF6B6B",      # Red untuk dependency
    "ENV": "#4ECDC4",       # Teal untuk environment
    "SETUP": "#45B7D1",     # Blue untuk setup
    
    # Dataset related colors
    "DOWNLOAD": "#96CEB4",  # Green untuk download
    "SPLIT": "#7DCEA0",     # Medium green untuk split
    "PREPROC": "#FFEAA7",   # Yellow untuk preprocessing
    "AUGMENT": "#DDA0DD",   # Plum untuk augmentation
    "VALID": "#98D8C8",     # Mint untuk validation
    "ANALYZE": "#F7DC6F",   # Light yellow untuk analysis
    
    # Enhanced dataset colors
    "DATASET_DOWNLOADER": "#2ECC71",    # Green untuk enhanced downloader
    "DATASET_RENAME": "#E74C3C",    # Red untuk renaming operations
    "DATASET_ORG": "#9B59B6",   # Purple untuk enhanced organizer
    "DATASET_PREPROC": "#3498DB",    # Blue untuk naming operations
    "DATASET_AUGMENT": "#F39C12",    # Orange untuk threading operations
    
    # Model related colors
    "PRETRAIN": "#D7BDE2",  # Light purple untuk pretrained model
    "BACKBONE": "#A9CCE3",  # Sky blue untuk backbone
    "HYPER": "#A3E4D7",     # Aqua untuk hyperparameters
    "STRATEGY": "#FAD7A0",  # Light orange untuk strategy
    "TRAINING": "#D2B4DE",  # Lavender untuk training
    "MOD_TRAIN": "#85C1E9",     # Light blue untuk training
    "MOD_EVAL": "#F8C471",      # Orange untuk evaluation
    "MOD_INFER": "#BB8FCE",     # Purple untuk inference
    "MOD_EXPORT": "#82E0AA",    # Light green untuk export
    "MOD_CONFIG": "#F1948A",    # Pink untuk config
    
    # UI components colors
    "MAIN": "#AED6F1",      # Light blue untuk main
    "DASH": "#D5DBDB",      # Gray untuk dashboard
    "COMP": "#E8DAEF",      # Light purple untuk components
    "HAND": "#FADBD8",      # Light pink untuk handlers
    "OBS": "#D4EDDA",       # Light green untuk observers
    
    # Common utilities colors
    "LOG": "#FCF3CF",       # Light yellow untuk logger
    "CFG": "#EBDEF0",       # Light lavender untuk config
    "UTILS": "#E5E7E9",     # Light gray untuk utils
    
    # Detection colors
    "DETECT": "#ABEBC6",    # Light green untuk detection
    "POST": "#F9E79F"       # Light yellow untuk postprocess
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
        'env_config_initialized': "smartcash.ui.setup.env_config",
        'dependency_installer_initialized': "smartcash.ui.setup.dependency",
        'split_initialized': "smartcash.ui.split",
        'downloader_initialized': "smartcash.dataset.download",
        'augmentation_initialized': "smartcash.dataset.augmentation",
        'preprocessing_initialized': "smartcash.dataset.preprocessing",
        'backbone_initialized': "smartcash.ui.backbone",
        'hyperparameters_initialized': "smartcash.ui.hyperparameters",
        'pretrained_model': 'smartcash.ui.pretrained_model',
        'strategy_initialized': "smartcash.ui.strategy",
        'training_initialized': "smartcash.ui.training",
        'evaluation_initialized': "smartcash.ui.evaluation",
        
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



# Export constants for backward compatibility
__all__ = [
    # Existing exports
    'KNOWN_NAMESPACES', 'NAMESPACE_COLORS', 'get_namespace_id', 'get_namespace_color',
    'format_log_message', 'format_log_message_html', 'create_namespace_badge',
    'register_namespace', 'get_all_namespaces', 'get_namespace_summary',
    
    # NEW: UUID-specific exports
    'register_uuid_namespace', 'get_uuid_operation_context', 'format_uuid_progress_message',
]