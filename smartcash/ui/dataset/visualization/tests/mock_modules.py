"""
File: smartcash/ui/dataset/visualization/tests/mock_modules.py
Deskripsi: Mock untuk modul yang dibutuhkan dalam pengujian
"""

import sys
from unittest.mock import MagicMock
import ipywidgets as widgets

# Mock untuk smartcash.ui.components.header
class HeaderMock:
    @staticmethod
    def create_header(title, description=None, icon=None):
        """
        Mock untuk create_header dari ui/utils/header_utils.py
        """
        return widgets.HTML(f"<h2>{title}</h2>")

# Mock untuk smartcash.ui.components.tabs
class TabsMock:
    @staticmethod
    def create_tabs(tabs_list):
        """
        Mock untuk create_tabs
        """
        tab = widgets.Tab()
        for i, (title, content) in enumerate(tabs_list):
            tab.set_title(i, title)
        return tab

# Mock untuk smartcash.ui.utils.alert_utils
class AlertUtilsMock:
    @staticmethod
    def create_status_indicator(status_type, message):
        """
        Mock untuk create_status_indicator
        """
        return widgets.HTML(f"<div class='{status_type}'>{message}</div>")

# Mock untuk smartcash.common.config
class ConfigManagerMock:
    """Mock untuk ConfigManager"""
    
    def get_module_config(self, module_name):
        """Mock untuk get_module_config"""
        return {'dataset_path': '/dummy/path'}

def get_config_manager_mock():
    """Mock untuk get_config_manager"""
    return ConfigManagerMock()

# Setup mock modules
def setup_mock_modules():
    """Setup mock modules untuk testing"""
    # Mock untuk header
    sys.modules['smartcash.ui.components.header'] = MagicMock()
    sys.modules['smartcash.ui.components.header'].create_header = HeaderMock.create_header
    
    # Mock untuk tabs
    sys.modules['smartcash.ui.components.tabs'] = MagicMock()
    sys.modules['smartcash.ui.components.tabs'].create_tabs = TabsMock.create_tabs
    
    # Mock untuk alert_utils
    sys.modules['smartcash.ui.utils.alert_utils'] = MagicMock()
    sys.modules['smartcash.ui.utils.alert_utils'].create_status_indicator = AlertUtilsMock.create_status_indicator
    
    # Mock untuk config
    sys.modules['smartcash.common.config'] = MagicMock()
    sys.modules['smartcash.common.config'].get_config_manager = get_config_manager_mock 