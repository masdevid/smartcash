#!/usr/bin/env python3
"""
Manual testing script for dependency operations.
This script provides a manual testing interface to verify all dependency operations work correctly.

Author: Claude Code
Date: 2025-07-15
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(message: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def print_section(message: str):
    """Print a formatted section."""
    print(f"\n{'-'*40}")
    print(f"  {message}")
    print(f"{'-'*40}")

def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()

def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = get_user_input(f"{message} (y/n)", "n").lower()
    return response in ['y', 'yes']

class ManualDependencyTester:
    """Manual tester for dependency operations."""
    
    def __init__(self):
        """Initialize the manual tester."""
        self.test_packages = ['requests', 'pyyaml', 'tqdm']
        self.custom_packages = ['numpy>=1.20.0', 'pandas>=1.3.0']
        
    def run_manual_tests(self):
        """Run manual tests with user interaction."""
        print_header("MANUAL DEPENDENCY OPERATIONS TEST")
        print("This script will guide you through testing all dependency operations.")
        print("You will be prompted to confirm each test operation.")
        print()
        print("⚠️  WARNING: This will perform actual package installations/uninstallations!")
        print("Make sure you're running this in a virtual environment.")
        
        if not confirm_action("Do you want to continue?"):
            print("Test cancelled by user.")
            return
        
        # Initialize the operation manager
        manager = self._initialize_operation_manager()
        if not manager:
            print("❌ Failed to initialize operation manager")
            return
        
        print("✅ Operation manager initialized successfully")
        
        # Test 1: Check current package status
        self._test_check_status(manager)
        
        # Test 2: Install packages
        self._test_install_packages(manager)
        
        # Test 3: Check status after installation
        self._test_check_status_after_install(manager)
        
        # Test 4: Update packages
        self._test_update_packages(manager)
        
        # Test 5: Install requirements.txt
        self._test_install_requirements(manager)
        
        # Test 6: Uninstall packages
        self._test_uninstall_packages(manager)
        
        # Test 7: Final status check
        self._test_final_status_check(manager)
        
        print_header("MANUAL TEST COMPLETED")
        print("All manual tests have been completed.")
        print("Please review the output above for any errors or unexpected behavior.")
    
    def _initialize_operation_manager(self):
        """Initialize the dependency operation manager."""
        try:
            from smartcash.ui.setup.dependency.operations.operation_manager import DependencyOperationManager
            
            # Create mock UI components
            ui_components = {
                'operation_container': self._create_mock_container(),
                'widgets': {
                    'package_categories': self._create_mock_widget([]),
                    'custom_packages_input': self._create_mock_widget('')
                },
                'containers': {
                    'summary': self._create_mock_container()
                }
            }
            
            config = {
                'selected_packages': [],
                'custom_packages': '',
                'use_index_url': False
            }
            
            manager = DependencyOperationManager(config, ui_components)
            manager.initialize()
            
            return manager
            
        except Exception as e:
            print(f"❌ Error initializing operation manager: {e}")
            return None
    
    def _test_check_status(self, manager):
        """Test checking package status."""
        print_section("Testing Package Status Check")
        
        if not confirm_action("Test checking package status?"):
            print("Skipping package status check test")
            return
        
        try:
            print("Checking status of test packages...")
            result = manager.execute_check_status(self.test_packages)
            
            print(f"✅ Status check result: {result}")
            
            if result.get('success'):
                print("✅ Package status check completed successfully")
            else:
                print(f"⚠️  Package status check had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during status check: {e}")
    
    def _test_install_packages(self, manager):
        """Test installing packages."""
        print_section("Testing Package Installation")
        
        packages_to_install = self.test_packages.copy()
        
        print(f"Packages to install: {packages_to_install}")
        
        if not confirm_action("Test installing packages? (This will actually install packages)"):
            print("Skipping package installation test")
            return
        
        try:
            # Update UI components with packages to install
            manager.ui_components['widgets']['package_categories'].selected = packages_to_install
            manager.ui_components['widgets']['custom_packages_input'].value = '\n'.join(self.custom_packages)
            
            print("Installing packages...")
            result = manager.execute_install(packages_to_install)
            
            print(f"✅ Installation result: {result}")
            
            if result.get('success'):
                print("✅ Package installation completed successfully")
            else:
                print(f"⚠️  Package installation had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during installation: {e}")
    
    def _test_check_status_after_install(self, manager):
        """Test checking status after installation."""
        print_section("Testing Package Status After Installation")
        
        if not confirm_action("Test checking package status after installation?"):
            print("Skipping post-installation status check")
            return
        
        try:
            print("Checking status after installation...")
            result = manager.execute_check_status(self.test_packages)
            
            print(f"✅ Post-installation status result: {result}")
            
            if result.get('success'):
                print("✅ Post-installation status check completed successfully")
            else:
                print(f"⚠️  Post-installation status check had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during post-installation status check: {e}")
    
    def _test_update_packages(self, manager):
        """Test updating packages."""
        print_section("Testing Package Update")
        
        packages_to_update = self.test_packages[:2]  # Update first 2 packages
        
        print(f"Packages to update: {packages_to_update}")
        
        if not confirm_action("Test updating packages? (This will actually update packages)"):
            print("Skipping package update test")
            return
        
        try:
            print("Updating packages...")
            result = manager.execute_update(packages_to_update)
            
            print(f"✅ Update result: {result}")
            
            if result.get('success'):
                print("✅ Package update completed successfully")
            else:
                print(f"⚠️  Package update had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during update: {e}")
    
    def _test_install_requirements(self, manager):
        """Test installing requirements.txt."""
        print_section("Testing Requirements.txt Installation")
        
        # Create a temporary requirements.txt file
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_path = os.path.join(temp_dir, 'requirements.txt')
            
            # Create requirements.txt with test packages
            with open(requirements_path, 'w') as f:
                f.write("# Test requirements file\n")
                f.write("requests>=2.25.0\n")
                f.write("pyyaml>=6.0\n")
                f.write("tqdm>=4.60.0\n")
            
            print(f"Created test requirements.txt at: {requirements_path}")
            
            with open(requirements_path, 'r') as f:
                print("Contents:")
                print(f.read())
            
            if not confirm_action("Test installing requirements.txt? (This will actually install packages)"):
                print("Skipping requirements.txt installation test")
                return
            
            try:
                print("Installing requirements.txt...")
                result = manager.install_requirements_txt(temp_dir)
                
                print(f"✅ Requirements installation result: {result}")
                
                if result.get('success'):
                    print("✅ Requirements.txt installation completed successfully")
                else:
                    print(f"⚠️  Requirements.txt installation had issues: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error during requirements.txt installation: {e}")
    
    def _test_uninstall_packages(self, manager):
        """Test uninstalling packages."""
        print_section("Testing Package Uninstallation")
        
        packages_to_uninstall = self.test_packages.copy()
        
        print(f"Packages to uninstall: {packages_to_uninstall}")
        print("⚠️  WARNING: This will actually uninstall packages!")
        
        if not confirm_action("Test uninstalling packages? (This will actually uninstall packages)"):
            print("Skipping package uninstallation test")
            return
        
        try:
            # Update UI components with packages to uninstall
            manager.ui_components['widgets']['package_categories'].selected = packages_to_uninstall
            
            print("Uninstalling packages...")
            result = manager.execute_uninstall(packages_to_uninstall)
            
            print(f"✅ Uninstallation result: {result}")
            
            if result.get('success'):
                print("✅ Package uninstallation completed successfully")
            else:
                print(f"⚠️  Package uninstallation had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during uninstallation: {e}")
    
    def _test_final_status_check(self, manager):
        """Test final status check."""
        print_section("Testing Final Package Status Check")
        
        if not confirm_action("Test final package status check?"):
            print("Skipping final status check")
            return
        
        try:
            print("Checking final status...")
            result = manager.execute_check_status(self.test_packages)
            
            print(f"✅ Final status result: {result}")
            
            if result.get('success'):
                print("✅ Final status check completed successfully")
            else:
                print(f"⚠️  Final status check had issues: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during final status check: {e}")
    
    def _create_mock_container(self):
        """Create a mock container for testing."""
        class MockContainer:
            def __init__(self):
                self.value = ""
                self.outputs = []
                
            def clear_output(self):
                self.outputs.clear()
                
            def clear_outputs(self):
                self.outputs.clear()
                
            def get_ui_components(self):
                return {}
                
        return MockContainer()
    
    def _create_mock_widget(self, value):
        """Create a mock widget for testing."""
        class MockWidget:
            def __init__(self, value):
                self.value = value
                self.selected = value if isinstance(value, list) else []
                
            def on_click(self, callback):
                pass
                
        return MockWidget(value)

def main():
    """Main function to run the manual tests."""
    tester = ManualDependencyTester()
    tester.run_manual_tests()

if __name__ == "__main__":
    main()