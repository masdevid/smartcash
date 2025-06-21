"""
File: smartcash/ui/utils/ui_logger_namespace.py
Deskripsi: Fixed UI logger namespace dengan augmentation module registration
"""

from typing import Dict, Any, Optional

# ENHANCED KNOWN_NAMESPACES dengan augmentation module
KNOWN_NAMESPACES = {
    # Setup & Environment
    "smartcash.ui.setup.env_config": "ENV",
    "smartcash.ui.setup.dependency": "DEPS",
    "smartcash.ui.setup": "SETUP",
    
    # Dataset related - FIXED: Added augmentation namespace
    "smartcash.ui.dataset.downloader": "DOWNLOAD",
    "smartcash.ui.dataset.split": "SPLIT",
    "smartcash.ui.dataset.preprocessing": "PREPROC",
    "smartcash.ui.dataset.augmentation": "AUGMENT",  # FIXED: Added augmentation
    "smartcash.ui.dataset.validation": "VALID",
    
    # Backend dataset modules
    "smartcash.dataset.preprocessing": "DATASET_PREPROC",
    "smartcash.dataset.augmentation": "DATASET_AUGMENT",
    "smartcash.dataset.validation": "DATASET_VALID",
    "smartcash.dataset.analysis": "DATASET_ANALYZE",
    "smartcash.dataset.downloader": "DATASET_DOWNLOADER",
    "smartcash.dataset.file_renamer": "DATASET_RENAME",
    "smartcash.dataset.organizer": "DATASET_ORG",
    
    # Model related
    "smartcash.ui.pretrained_model": "PRETRAIN",
    "smartcash.ui.backbone": "BACKBONE",
    "smartcash.ui.hyperparameters": "HYPER",
    "smartcash.ui.strategy": "STRATEGY",
    "smartcash.ui.training": "TRAINING",
    "smartcash.ui.evaluation": "EVAL",
    
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

# ENHANCED NAMESPACE_COLORS dengan augmentation color
NAMESPACE_COLORS = {
    # Setup & Environment colors
    "DEPS": "#FF6B6B",
    "ENV": "#4ECDC4",
    "SETUP": "#45B7D1",
    
    # Dataset related colors - FIXED: Added AUGMENT color
    "DOWNLOAD": "#96CEB4",
    "SPLIT": "#7DCEA0",
    "PREPROC": "#FFEAA7",
    "AUGMENT": "#A29BFE",     # FIXED: Purple color for augmentation
    "VALID": "#98D8C8",
    "ANALYZE": "#F7DC6F",
    
    # Enhanced dataset colors
    "DATASET_DOWNLOADER": "#2ECC71",
    "DATASET_RENAME": "#E74C3C",
    "DATASET_ORG": "#9B59B6",
    "DATASET_PREPROC": "#3498DB",
    "DATASET_AUGMENT": "#8E44AD",    # FIXED: Different shade for backend augment
    
    # Model related colors
    "PRETRAIN": "#D7BDE2",
    "BACKBONE": "#A9CCE3",
    "HYPER": "#A3E4D7",
    "STRATEGY": "#FAD7A0",
    "TRAINING": "#D2B4DE",
    "MOD_TRAIN": "#85C1E9",
    "MOD_EVAL": "#F8C471",
    "MOD_INFER": "#BB8FCE",
    "MOD_EXPORT": "#82E0AA",
    "MOD_CONFIG": "#F1948A",
    
    # Common utilities colors
    "LOG": "#FCF3CF",
    "CFG": "#EBDEF0",
    "UTILS": "#E5E7E9",
    
    # Detection colors
    "DETECT": "#ABEBC6",
    "POST": "#F9E79F"
}

def get_namespace_color(namespace_id: str) -> str:
    """Mendapatkan warna untuk namespace ID yang diberikan"""
    if not namespace_id:
        return "#007bff"
    return NAMESPACE_COLORS.get(namespace_id, "#007bff")

def get_namespace_id(ui_components: Dict[str, Any]) -> Optional[str]:
    """Enhanced namespace ID detection dengan augmentation support"""
    # Cek namespace tersedia di ui_components
    namespace = ui_components.get('logger_namespace')
    if namespace:
        return KNOWN_NAMESPACES.get(namespace, namespace)
    
    # Enhanced flag mapping dengan augmentation support
    flag_mapping = {
        # Existing flags
        'env_config_initialized': "smartcash.ui.setup.env_config",
        'dependency_installer_initialized': "smartcash.ui.setup.dependency",
        'split_initialized': "smartcash.ui.split",
        'downloader_initialized': "smartcash.dataset.download",
        'preprocessing_initialized': "smartcash.dataset.preprocessing",
        'backbone_initialized': "smartcash.ui.backbone",
        'hyperparameters_initialized': "smartcash.ui.hyperparameters",
        'pretrained_model': 'smartcash.ui.pretrained_model',
        'strategy_initialized': "smartcash.ui.strategy",
        'training_initialized': "smartcash.ui.training",
        'evaluation_initialized': "smartcash.ui.evaluation",
        
        # FIXED: Added augmentation flag detection
        'augmentation_initialized': "smartcash.ui.dataset.augmentation",
        
        # Enhanced flags
        'file_renamer_initialized': "smartcash.dataset.file_renamer",
        'organizer_initialized': "smartcash.dataset.organizer",
        'uuid_naming_initialized': "smartcash.common.file_naming",
        'threading_initialized': "smartcash.common.threadpools"
    }
    
    for flag, namespace in flag_mapping.items():
        if ui_components.get(flag, False):
            return KNOWN_NAMESPACES.get(namespace)
    
    return None

def format_log_message_html(ui_components: Dict[str, Any], message: str, 
                           level: str = "info", icon: str = None) -> str:
    """Enhanced HTML formatting dengan augmentation context awareness"""
    try:
        from smartcash.ui.utils.constants import COLORS
    except ImportError:
        COLORS = {
            "primary": "#007bff", "success": "#28a745", "warning": "#ffc107", 
            "danger": "#dc3545", "muted": "#6c757d", "text": "#212529"
        }
    
    # Enhanced level config dengan augmentation operations
    level_config = {
        "debug": {"emoji": "🔍", "color": COLORS.get("muted", "#6c757d")},
        "info": {"emoji": "ℹ️", "color": COLORS.get("primary", "#007bff")},
        "success": {"emoji": "✅", "color": COLORS.get("success", "#28a745")},
        "warning": {"emoji": "⚠️", "color": COLORS.get("warning", "#ffc107")},
        "error": {"emoji": "❌", "color": COLORS.get("danger", "#dc3545")},
        "critical": {"emoji": "🔥", "color": COLORS.get("danger", "#dc3545")},
        
        # FIXED: Augmentation-specific levels
        "augment_info": {"emoji": "🔄", "color": "#A29BFE"},
        "augment_progress": {"emoji": "📊", "color": "#8E44AD"},
        "pipeline": {"emoji": "🚀", "color": "#6C5CE7"}
    }
    
    config = level_config.get(level, level_config["info"])
    emoji = icon or config["emoji"]
    color = config["color"]
    
    # Enhanced message cleaning
    clean_message = _clean_message(message)
    
    # Enhanced namespace untuk augmentation operations
    namespace_id = get_namespace_id(ui_components)
    namespace_color = get_namespace_color(namespace_id) if namespace_id else color
    
    # Enhanced HTML dengan augmentation operation indicators
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
    """Enhanced message cleaning dengan augmentation patterns"""
    import re
    
    # Remove timestamp patterns [HH:MM:SS]
    message = re.sub(r'\[(\d{2}:\d{2}:\d{2})\]\s*', '', message)
    
    # Remove namespace patterns [NAMESPACE]
    message = re.sub(r'\[([A-Z_]+)\]\s*', '', message)
    
    # Remove leading emoji + space jika ada
    message = re.sub(r'^[🔍ℹ️✅⚠️❌🔥🔄📊🚀]\s*', '', message)
    
    # FIXED: Clean augmentation patterns untuk readability
    message = re.sub(r'(augmentation_pipeline|augment_and_normalize)', 'pipeline', message)
    message = re.sub(r'(comprehensive_check|dataset_check)', 'check', message)
    
    # Clean multiple spaces
    message = re.sub(r'\s+', ' ', message)
    
    return message.strip()

def register_namespace(module_name: str, namespace_id: str, color: str = None) -> None:
    """Register namespace dynamically dengan augmentation support"""
    full_namespace = f"smartcash.{module_name}" if not module_name.startswith('smartcash') else module_name
    KNOWN_NAMESPACES[full_namespace] = namespace_id
    
    if color:
        NAMESPACE_COLORS[namespace_id] = color
    
    # Auto-register common variants
    variants = [f"{full_namespace}.handlers", f"{full_namespace}.utils", f"{full_namespace}.components"]
    for variant in variants:
        KNOWN_NAMESPACES[variant] = namespace_id

# FIXED: Auto-register augmentation variants
register_namespace("smartcash.ui.dataset.augmentation", "AUGMENT", "#A29BFE")

def get_all_namespaces() -> Dict[str, str]:
    """Get all registered namespaces"""
    return KNOWN_NAMESPACES.copy()

def get_namespace_summary() -> Dict[str, Any]:
    """Enhanced namespace summary dengan augmentation operations"""
    base_summary = {
        'total_namespaces': len(KNOWN_NAMESPACES),
        'categories': {
            'setup': [ns for ns in KNOWN_NAMESPACES if 'setup' in ns or 'env' in ns],
            'dataset': [ns for ns in KNOWN_NAMESPACES if 'dataset' in ns],
            'model': [ns for ns in KNOWN_NAMESPACES if 'model' in ns],
            'ui': [ns for ns in KNOWN_NAMESPACES if 'ui' in ns],
            'common': [ns for ns in KNOWN_NAMESPACES if 'common' in ns],
            'detection': [ns for ns in KNOWN_NAMESPACES if 'detection' in ns],
            
            # FIXED: Augmentation-related categories
            'augmentation': [ns for ns in KNOWN_NAMESPACES if 'augment' in ns.lower()],
            'file_operations': [ns for ns in KNOWN_NAMESPACES if any(term in ns.lower() for term in ['file', 'organizer', 'downloader'])],
            'threading': [ns for ns in KNOWN_NAMESPACES if 'thread' in ns.lower()]
        },
        'namespace_ids': list(KNOWN_NAMESPACES.values())
    }
    
    # Add augmentation-specific statistics
    augment_count = len(base_summary['categories']['augmentation'])
    base_summary['augmentation_support'] = {
        'enabled': augment_count > 0,
        'augmentation_namespaces': augment_count,
        'ui_modules': len([ns for ns in KNOWN_NAMESPACES if 'ui.dataset.augmentation' in ns]),
        'backend_modules': len([ns for ns in KNOWN_NAMESPACES if 'dataset.augmentation' in ns and 'ui' not in ns])
    }
    
    return base_summary

# Export constants
__all__ = [
    'KNOWN_NAMESPACES', 'NAMESPACE_COLORS', 'get_namespace_id', 'get_namespace_color',
    'format_log_message_html', 'register_namespace', 'get_all_namespaces', 'get_namespace_summary'
]