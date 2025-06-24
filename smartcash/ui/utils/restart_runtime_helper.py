"""
File: smartcash/ui/utils/restart_runtime_helper.py
Deskripsi: Utility untuk clear cache dan restart runtime di Google Colab
"""

import sys
import importlib
from typing import List, Optional

def clear_module_cache(module_patterns: List[str]) -> None:
    """
    🧹 Clear cache untuk modules yang match patterns
    
    Args:
        module_patterns: List pattern module names (e.g., ['smartcash.ui'])
    """
    cleared_modules = []
    
    # Get list modules to clear
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        for pattern in module_patterns:
            if module_name.startswith(pattern):
                modules_to_clear.append(module_name)
                break
    
    # Clear modules
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
            cleared_modules.append(module_name)
    
    if cleared_modules:
        print(f"🧹 Cleared {len(cleared_modules)} cached modules")
        for module in cleared_modules[:5]:  # Show first 5
            print(f"   ↳ {module}")
        if len(cleared_modules) > 5:
            print(f"   ↳ ... dan {len(cleared_modules) - 5} lainnya")
    else:
        print("ℹ️ No cached modules found to clear")

def restart_runtime_warning() -> None:
    """⚠️ Show restart runtime warning dengan instructions"""
    print("""
🚨 CACHE ISSUE DETECTED! 

Untuk mengatasi ImportError, silakan:

1️⃣ **RESTART RUNTIME:**
   • Menu: Runtime → Restart runtime
   • Shortcut: Ctrl+M + .

2️⃣ **Clear Module Cache (Alternative):**
   • Jalankan: clear_smartcash_cache()
   • Lalu re-import modules

3️⃣ **Re-run Cells:**
   • Jalankan ulang cell import
   • Gunakan: initialize_env_config_ui()

📝 **Note:** Restart runtime adalah solusi paling efektif!
""")

def clear_smartcash_cache() -> None:
    """🧹 Clear SmartCash module cache specifically"""
    patterns = [
        'smartcash.ui.setup.env_config',
        'smartcash.ui.components',
        'smartcash.ui.utils',
        'smartcash.ui.handlers'
    ]
    clear_module_cache(patterns)
    print("✅ SmartCash cache cleared. Re-import modules now!")

def diagnose_import_issues(module_name: str) -> None:
    """🔍 Diagnose import issues untuk specific module"""
    print(f"🔍 Diagnosing import issues for: {module_name}")
    
    # Check if module in cache
    if module_name in sys.modules:
        print(f"✅ Module {module_name} found in cache")
        module = sys.modules[module_name]
        
        # Check module attributes
        if hasattr(module, '__file__'):
            print(f"📁 File: {module.__file__}")
        
        # Check available attributes
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        if attrs:
            print(f"🔧 Available attributes: {', '.join(attrs[:5])}")
            if len(attrs) > 5:
                print(f"   ... dan {len(attrs) - 5} lainnya")
        else:
            print("⚠️ No public attributes found")
            
    else:
        print(f"❌ Module {module_name} not in cache")
        
        # Check parent modules
        parts = module_name.split('.')
        for i in range(len(parts)):
            parent = '.'.join(parts[:i+1])
            if parent in sys.modules:
                print(f"✅ Parent module found: {parent}")
            else:
                print(f"❌ Missing parent: {parent}")
                break

# Quick diagnostic functions
def check_ui_factory_import():
    """🔍 Check UIFactory import specifically"""
    try:
        from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
        print("✅ UIFactory import successful!")
        
        # Check UIFactory methods
        methods = [m for m in dir(UIFactory) if not m.startswith('_')]
        print(f"🔧 UIFactory methods: {', '.join(methods)}")
        
        return True
    except ImportError as e:
        print(f"❌ UIFactory import failed: {e}")
        diagnose_import_issues('smartcash.ui.setup.env_config.components.ui_factory')
        return False

def check_env_config_import():
    """🔍 Check env_config import chain"""
    try:
        from smartcash.ui.setup.env_config import initialize_env_config_ui
        print("✅ env_config import successful!")
        return True
    except ImportError as e:
        print(f"❌ env_config import failed: {e}")
        restart_runtime_warning()
        return False