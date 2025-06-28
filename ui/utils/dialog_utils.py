"""
File: smartcash/ui/utils/dialog_utils.py
Deskripsi: Utility functions untuk seamless integration glass morphism dialog dengan preprocessing workflow
"""

from typing import Dict, Any, Optional, Callable
import time

def ensure_dialog_readiness(ui_components: Dict[str, Any], timeout: float = 2.0) -> bool:
    """Ensure dialog component siap untuk digunakan dengan timeout protection
    
    Args:
        ui_components: Dictionary UI components
        timeout: Timeout dalam detik untuk wait readiness
        
    Returns:
        bool: True jika dialog ready, False jika timeout
    """
    start_time = time.time()
    
    try:
        from smartcash.ui.components.dialog.confirmation_dialog import create_confirmation_area
        
        while time.time() - start_time < timeout:
            try:
                # Ensure confirmation area exists
                if 'confirmation_area' not in ui_components:
                    create_confirmation_area(ui_components)
                
                confirmation_area = ui_components.get('confirmation_area')
                if confirmation_area and hasattr(confirmation_area, 'layout'):
                    return True
                    
                time.sleep(0.1)  # Small delay untuk prevent busy waiting
                
            except Exception:
                time.sleep(0.1)
                continue
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error ensuring dialog readiness: {str(e)}")
        return False

def safe_show_dialog(ui_components: Dict[str, Any],
                    title: str,
                    message: str,
                    on_confirm: Optional[Callable] = None,
                    on_cancel: Optional[Callable] = None,
                    confirm_text: str = "Konfirmasi",
                    cancel_text: str = "Batal",
                    danger_mode: bool = False,
                    max_retries: int = 3) -> bool:
    """Safely show dialog dengan retry mechanism
    
    Args:
        ui_components: Dictionary UI components
        title: Dialog title
        message: Dialog message
        on_confirm: Callback untuk confirm button
        on_cancel: Callback untuk cancel button
        confirm_text: Text untuk confirm button
        cancel_text: Text untuk cancel button
        danger_mode: Apakah menggunakan danger styling
        max_retries: Maximum retry attempts
        
    Returns:
        bool: True jika dialog berhasil ditampilkan
    """
    for attempt in range(max_retries):
        try:
            # Ensure dialog readiness
            if not ensure_dialog_readiness(ui_components):
                print(f"⚠️ Dialog not ready, attempt {attempt + 1}/{max_retries}")
                continue
            
            # Reset any existing dialog state
            reset_dialog_state_safe(ui_components)
            
            # Show dialog
            from smartcash.ui.components.dialog.confirmation_dialog import show_confirmation_dialog
            
            show_confirmation_dialog(
                ui_components=ui_components,
                title=title,
                message=message,
                on_confirm=on_confirm,
                on_cancel=on_cancel,
                confirm_text=confirm_text,
                cancel_text=cancel_text,
                danger_mode=danger_mode
            )
            
            return True
            
        except Exception as e:
            print(f"⚠️ Dialog show attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait sebelum retry
                continue
    
    # Fallback ke console input jika semua attempts gagal
    print(f"⚠️ Dialog failed after {max_retries} attempts, using console fallback")
    return _console_fallback(title, message, on_confirm, on_cancel)

def reset_dialog_state_safe(ui_components: Dict[str, Any]) -> None:
    """Reset dialog state dengan comprehensive cleanup
    
    Args:
        ui_components: Dictionary UI components
    """
    try:
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        
        # Clear dialog area
        clear_dialog_area(ui_components)
        
        # Clear confirmation flags yang mungkin stuck
        stuck_flags = [
            '_preprocessing_confirmed',
            '_cleanup_confirmed',
            '_dialog_pending',
            '_dialog_visible'
        ]
        
        for flag in stuck_flags:
            if flag in ui_components:
                # Only clear jika bukan True (completed state)
                if ui_components[flag] is not True:
                    ui_components.pop(flag, None)
        
        # Reset confirmation area layout jika ada
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'none'
            confirmation_area.layout.visibility = 'hidden'
        
    except Exception as e:
        print(f"⚠️ Error resetting dialog state: {str(e)}")

def _console_fallback(title: str, 
                     message: str,
                     on_confirm: Optional[Callable] = None,
                     on_cancel: Optional[Callable] = None) -> bool:
    """Console fallback untuk dialog confirmation
    
    Args:
        title: Dialog title
        message: Dialog message  
        on_confirm: Callback untuk confirm
        on_cancel: Callback untuk cancel
        
    Returns:
        bool: True jika user confirm
    """
    try:
        print(f"\n📋 {title}")
        print(f"📝 {message}")
        
        response = input("Konfirmasi? (y/N): ").lower().strip()
        
        if response in ['y', 'yes', 'ya']:
            if on_confirm:
                on_confirm()
            return True
        else:
            if on_cancel:
                on_cancel()
            return False
            
    except Exception as e:
        print(f"⚠️ Console fallback error: {str(e)}")
        return False

def check_dialog_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check current dialog state untuk debugging
    
    Args:
        ui_components: Dictionary UI components
        
    Returns:
        Dict dengan dialog state information
    """
    state_info = {
        'confirmation_area_exists': 'confirmation_area' in ui_components,
        'confirmation_area_type': type(ui_components.get('confirmation_area', None)).__name__,
        'dialog_visible': False,
        'pending_confirmations': [],
        'layout_info': {}
    }
    
    try:
        # Check dialog visibility
        from smartcash.ui.components.dialog.confirmation_dialog import is_dialog_visible
        state_info['dialog_visible'] = is_dialog_visible(ui_components)
        
        # Check pending confirmations
        confirmation_flags = [k for k in ui_components.keys() 
                            if k.endswith('_confirmed')]
        for flag in confirmation_flags:
            if ui_components.get(flag) is None:
                state_info['pending_confirmations'].append(flag)
        
        # Check layout info
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            layout = confirmation_area.layout
            state_info['layout_info'] = {
                'display': getattr(layout, 'display', 'unknown'),
                'visibility': getattr(layout, 'visibility', 'unknown'),
                'height': getattr(layout, 'height', 'unknown')
            }
    
    except Exception as e:
        state_info['error'] = str(e)
    
    return state_info

def force_dialog_cleanup(ui_components: Dict[str, Any]) -> None:
    """Force cleanup dialog state untuk recovery dari stuck state
    
    Args:
        ui_components: Dictionary UI components
    """
    try:
        print("🔧 Force cleaning dialog state...")
        
        # Clear semua dialog related keys
        keys_to_remove = []
        for key in ui_components.keys():
            if any(pattern in key for pattern in ['_confirmed', '_dialog', '_pending']):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            ui_components.pop(key, None)
            print(f"   🗑️ Removed: {key}")
        
        # Reset confirmation area
        if 'confirmation_area' in ui_components:
            try:
                confirmation_area = ui_components['confirmation_area']
                if hasattr(confirmation_area, 'layout'):
                    confirmation_area.layout.display = 'none'
                    confirmation_area.layout.visibility = 'hidden'
                
                # Clear output
                from IPython.display import clear_output
                with confirmation_area:
                    clear_output(wait=True)
                    
                print("   ✅ Confirmation area reset")
                
            except Exception as area_error:
                print(f"   ⚠️ Error resetting area: {str(area_error)}")
        
        # Re-create confirmation area
        from smartcash.ui.components.dialog.confirmation_dialog import create_confirmation_area
        ui_components['confirmation_area'] = create_confirmation_area(ui_components)
        
        print("✅ Dialog state force cleanup completed")
        
    except Exception as e:
        print(f"❌ Error in force cleanup: {str(e)}")

def validate_dialog_integration(ui_components: Dict[str, Any]) -> Dict[str, bool]:
    """Validate dialog integration untuk troubleshooting
    
    Args:
        ui_components: Dictionary UI components
        
    Returns:
        Dict dengan validation results
    """
    validation_results = {}
    
    try:
        # Test 1: Dialog component import
        try:
            from smartcash.ui.components.dialog.confirmation_dialog import (
                show_confirmation_dialog, clear_dialog_area, is_dialog_visible
            )
            validation_results['imports_available'] = True
        except ImportError as e:
            validation_results['imports_available'] = False
            validation_results['import_error'] = str(e)
        
        # Test 2: Confirmation area creation
        try:
            from smartcash.ui.components.dialog.confirmation_dialog import create_confirmation_area
            test_area = create_confirmation_area(ui_components)
            validation_results['area_creation'] = test_area is not None
        except Exception as e:
            validation_results['area_creation'] = False
            validation_results['area_error'] = str(e)
        
        # Test 3: Layout properties
        try:
            confirmation_area = ui_components.get('confirmation_area')
            if confirmation_area:
                has_layout = hasattr(confirmation_area, 'layout')
                validation_results['layout_available'] = has_layout
                if has_layout:
                    layout = confirmation_area.layout
                    validation_results['layout_properties'] = {
                        'has_display': hasattr(layout, 'display'),
                        'has_visibility': hasattr(layout, 'visibility'),
                        'has_height': hasattr(layout, 'height')
                    }
            else:
                validation_results['layout_available'] = False
        except Exception as e:
            validation_results['layout_available'] = False
            validation_results['layout_error'] = str(e)
        
        # Test 4: JavaScript environment
        try:
            from IPython.display import Javascript
            validation_results['javascript_available'] = True
        except ImportError:
            validation_results['javascript_available'] = False
        
        # Overall status
        critical_tests = ['imports_available', 'area_creation', 'layout_available']
        validation_results['overall_status'] = all(
            validation_results.get(test, False) for test in critical_tests
        )
        
    except Exception as e:
        validation_results['validation_error'] = str(e)
        validation_results['overall_status'] = False
    
    return validation_results