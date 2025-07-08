"""
Tests for smartcash.ui.setup.colab.components.setup_summary
"""

import pytest
from unittest.mock import Mock
from smartcash.ui.setup.colab.components.setup_summary import (
    create_setup_summary,
    update_setup_summary,
    update_setup_summary_with_verification,
    _get_initial_summary_content,
    _get_status_icon,
    _format_enhanced_summary_content,
    _format_verification_status,
    _format_system_status,
    _format_issues_section,
    _get_section_color,
    _get_value_status
)


class TestSetupSummary:
    """Test cases for setup summary components."""
    
    def test_create_setup_summary_default(self):
        """Test creating setup summary with default content."""
        widget = create_setup_summary()
        
        assert widget is not None
        assert hasattr(widget, 'value')
        assert 'Waiting for setup to begin' in widget.value
        assert 'Setup Environment' in widget.value
    
    def test_create_setup_summary_with_message(self):
        """Test creating setup summary with custom message."""
        custom_message = "Custom initialization message"
        widget = create_setup_summary(custom_message)
        
        assert widget is not None
        assert custom_message in widget.value
        assert 'Setup Environment' in widget.value
    
    def test_update_setup_summary_basic(self):
        """Test basic setup summary update."""
        widget = Mock()
        widget.value = ""
        
        update_setup_summary(
            widget, 
            "Setup in progress", 
            "info"
        )
        
        assert "Setup in progress" in widget.value
        assert "ℹ️" in widget.value  # Info icon
    
    def test_update_setup_summary_with_details(self):
        """Test setup summary update with details."""
        widget = Mock()
        widget.value = ""
        
        details = {
            "System": {
                "OS": "Linux",
                "RAM": "16GB",
                "GPU": True
            },
            "Setup": ["Drive mounted", "Configs synced"]
        }
        
        update_setup_summary(
            widget,
            "Setup completed successfully",
            "success",
            details
        )
        
        assert "Setup completed successfully" in widget.value
        assert "✅" in widget.value  # Success icon
        assert "Linux" in widget.value
        assert "16GB" in widget.value
        assert "Drive mounted" in widget.value
    
    def test_update_setup_summary_with_verification(self):
        """Test setup summary update with verification results."""
        widget = Mock()
        widget.value = ""
        
        verification_results = {
            'success': True,
            'issues': [],
            'verification': {
                'drive_mount': {
                    'mounted': True,
                    'write_access': True
                },
                'symlinks': {
                    'valid_count': 5,
                    'total_count': 5,
                    'all_valid': True
                },
                'folders': {
                    'existing_count': 10,
                    'total_count': 10,
                    'all_exist': True
                },
                'env_vars': {
                    'valid_count': 3,
                    'total_count': 3,
                    'all_valid': True
                }
            }
        }
        
        system_info = {
            'environment': {'type': 'colab'},
            'hardware': {
                'cpu_cores': 2,
                'total_ram_gb': 13.0,
                'gpu_info': 'Tesla T4'
            },
            'storage': {
                'free_gb': 54.8,
                'total_gb': 100.0
            }
        }
        
        update_setup_summary_with_verification(
            widget,
            verification_results,
            system_info
        )
        
        assert "Environment Setup Complete" in widget.value
        assert "🔧 Setup Components" in widget.value
        assert "💻 System Status" in widget.value
        assert "✅ 5/5" in widget.value  # Symlinks
        assert "Tesla T4" in widget.value
    
    def test_update_setup_summary_with_verification_issues(self):
        """Test setup summary update with verification issues."""
        widget = Mock()
        widget.value = ""
        
        verification_results = {
            'success': False,
            'issues': [
                'Google Drive not mounted',
                'Missing configuration files'
            ],
            'verification': {
                'drive_mount': {
                    'mounted': False,
                    'write_access': False
                },
                'symlinks': {
                    'valid_count': 3,
                    'total_count': 5,
                    'all_valid': False
                }
            }
        }
        
        update_setup_summary_with_verification(
            widget,
            verification_results
        )
        
        assert "Setup Issues Found (2 issues)" in widget.value
        assert "⚠️ Issues Found (2)" in widget.value
        assert "Google Drive not mounted" in widget.value
        assert "Missing configuration files" in widget.value
    
    def test_get_initial_summary_content_default(self):
        """Test initial summary content generation."""
        content = _get_initial_summary_content()
        
        assert "Waiting for setup to begin" in content
        assert "Setup Environment" in content
        assert "font-family" in content  # CSS styling
    
    def test_get_initial_summary_content_custom(self):
        """Test initial summary content with custom message."""
        custom_message = "Ready to configure environment"
        content = _get_initial_summary_content(custom_message)
        
        assert custom_message in content
        assert "Setup Environment" in content
    
    def test_get_status_icon(self):
        """Test status icon selection."""
        assert _get_status_icon('success') == '✅'
        assert _get_status_icon('warning') == '⚠️'
        assert _get_status_icon('error') == '❌'
        assert _get_status_icon('info') == 'ℹ️'
        assert _get_status_icon('unknown') == 'ℹ️'  # default
    
    def test_format_enhanced_summary_content_dict(self):
        """Test enhanced summary content formatting with dictionary."""
        data = {
            "System Information": {
                "OS": "Linux",
                "Python": "3.8.10",
                "Available": True,
                "Missing": False,
                "Count": 42
            }
        }
        
        content = _format_enhanced_summary_content(data)
        
        assert "System Information" in content
        assert "Linux" in content
        assert "3.8.10" in content
        assert "✅" in content  # True value
        assert "❌" in content  # False value
        assert "42" in content
    
    def test_format_enhanced_summary_content_list(self):
        """Test enhanced summary content formatting with list."""
        data = {
            "Setup Steps": [
                "Initialize environment",
                "Mount Google Drive",
                "Create symbolic links"
            ]
        }
        
        content = _format_enhanced_summary_content(data)
        
        assert "Setup Steps" in content
        assert "Initialize environment" in content
        assert "Mount Google Drive" in content
        assert "Create symbolic links" in content
    
    def test_format_enhanced_summary_content_invalid(self):
        """Test enhanced summary content formatting with invalid data."""
        content = _format_enhanced_summary_content("not a dict")
        
        assert "No summary data available" in content
    
    def test_format_verification_status(self):
        """Test verification status formatting."""
        verification = {
            'drive_mount': {
                'mounted': True,
                'write_access': True
            },
            'symlinks': {
                'valid_count': 4,
                'total_count': 5,
                'all_valid': False
            },
            'folders': {
                'existing_count': 10,
                'total_count': 10,
                'all_exist': True
            },
            'env_vars': {
                'valid_count': 2,
                'total_count': 3,
                'all_valid': False
            }
        }
        
        content = _format_verification_status(verification)
        
        assert "✅ Mounted (writable)" in content
        assert "⚠️ 4/5" in content  # Symlinks
        assert "✅ 10/10" in content  # Folders
        assert "⚠️ 2/3" in content  # Environment vars
    
    def test_format_system_status_complete(self):
        """Test system status formatting with complete info."""
        system_info = {
            'environment': {'type': 'colab'},
            'hardware': {
                'cpu_cores': 2,
                'total_ram_gb': 13.0,
                'gpu_info': 'Tesla T4'
            },
            'storage': {
                'free_gb': 54.8,
                'total_gb': 100.0
            }
        }
        
        content = _format_system_status(system_info)
        
        assert "Colab" in content
        assert "2 cores" in content
        assert "13.0GB" in content
        assert "Tesla T4" in content
        assert "54.8GB free of 100.0GB" in content
    
    def test_format_system_status_none(self):
        """Test system status formatting with no info."""
        content = _format_system_status(None)
        
        assert "System information not available" in content
    
    def test_format_issues_section_with_issues(self):
        """Test issues section formatting with issues."""
        issues = [
            "Google Drive not mounted",
            "Missing configuration file: model_config.yaml",
            "Environment variable PYTHONPATH not set"
        ]
        
        content = _format_issues_section(issues)
        
        assert "⚠️ Issues Found (3)" in content
        assert "Google Drive not mounted" in content
        assert "model_config.yaml" in content
        assert "PYTHONPATH not set" in content
        assert "#fff3cd" in content  # Warning background color
    
    def test_format_issues_section_no_issues(self):
        """Test issues section formatting with no issues."""
        content = _format_issues_section([])
        
        assert content == ""
    
    def test_get_section_color(self):
        """Test section color selection."""
        assert _get_section_color("System Information") == '#2196f3'
        assert _get_section_color("Resources") == '#4caf50'
        assert _get_section_color("Verification Status") == '#ff9800'
        assert _get_section_color("Setup Progress") == '#9c27b0'
        assert _get_section_color("Environment Variables") == '#607d8b'
        assert _get_section_color("Unknown Section") == '#333'
    
    def test_get_value_status(self):
        """Test value status determination."""
        # Boolean values
        icon, color = _get_value_status(True)
        assert icon == '✅' and color == '#4caf50'
        
        icon, color = _get_value_status(False)
        assert icon == '❌' and color == '#f44336'
        
        icon, color = _get_value_status(None)
        assert icon == '❌' and color == '#f44336'
        
        # String values
        icon, color = _get_value_status("Error occurred")
        assert icon == '❌' and color == '#f44336'
        
        icon, color = _get_value_status("Warning: partial setup")
        assert icon == '⚠️' and color == '#ff9800'
        
        icon, color = _get_value_status("Success: completed")
        assert icon == '✅' and color == '#4caf50'
        
        icon, color = _get_value_status("Normal value")
        assert icon == 'ℹ️' and color == '#333'
    
    def test_widget_layout_properties(self):
        """Test that widget has correct layout properties."""
        widget = create_setup_summary()
        
        layout = widget.layout
        assert layout.width == '100%'
        assert layout.padding == '15px'
        assert '1px solid #e0e0e0' in layout.border
        assert layout.border_radius == '6px'
        assert layout.background == '#f9f9f9'


class TestSetupSummaryEdgeCases:
    """Test edge cases for setup summary."""
    
    def test_empty_verification_results(self):
        """Test with empty verification results."""
        widget = Mock()
        widget.value = ""
        
        verification_results = {}
        
        update_setup_summary_with_verification(
            widget,
            verification_results
        )
        
        # Should handle empty results gracefully
        assert "Setup Issues Found" in widget.value
    
    def test_partial_verification_data(self):
        """Test with partial verification data."""
        widget = Mock()
        widget.value = ""
        
        verification_results = {
            'success': True,
            'verification': {
                'drive_mount': {'mounted': True}
                # Missing other verification data
            }
        }
        
        update_setup_summary_with_verification(
            widget,
            verification_results
        )
        
        # Should handle partial data without crashing
        assert widget.value is not None
        assert len(widget.value) > 0
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        # Test with non-dict data
        content = _format_enhanced_summary_content(None)
        assert "No summary data available" in content
        
        # Test with empty sections
        content = _format_enhanced_summary_content({
            "Empty Section": None,
            "Another Empty": []
        })
        # Should not include empty sections
        assert "Empty Section" not in content
        assert "Another Empty" not in content
    
    def test_long_issue_list(self):
        """Test with many issues."""
        issues = [f"Issue number {i}" for i in range(20)]
        
        content = _format_issues_section(issues)
        
        assert "⚠️ Issues Found (20)" in content
        assert "Issue number 0" in content
        assert "Issue number 19" in content
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        widget = Mock()
        widget.value = ""
        
        details = {
            "Unicode Test": {
                "Chinese": "测试",
                "Emoji": "🚀💻🔥",
                "Special": "<>&\"'"
            }
        }
        
        update_setup_summary(
            widget,
            "Testing unicode: 测试 🚀",
            "info",
            details
        )
        
        # Should handle unicode without crashing
        assert widget.value is not None
        assert "测试" in widget.value
        assert "🚀" in widget.value