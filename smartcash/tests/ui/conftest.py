"""
Test configuration for UI component tests.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the ipywidgets module
sys.modules['ipywidgets'] = MagicMock()
sys.modules['ipywidgets'].Widget = MagicMock
sys.modules['ipywidgets'].VBox = MagicMock
sys.modules['ipywidgets'].HBox = MagicMock
sys.modules['ipywidgets'].HTML = MagicMock
sys.modules['ipywidgets'].Button = MagicMock
sys.modules['ipywidgets'].Output = MagicMock

# Mock the IPython module
sys.modules['IPython'] = MagicMock()
sys.modules['IPython'].display = MagicMock()

# Mock the smartcash.ui.components module
sys.modules['smartcash.ui.components'] = MagicMock()
sys.modules['smartcash.ui.components.base_component'] = MagicMock()
sys.modules['smartcash.ui.components.log_accordion'] = MagicMock()
sys.modules['smartcash.ui.components.log_accordion.log_level'] = MagicMock()

# Mock the LogLevel enum
from enum import Enum
class MockLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

sys.modules['smartcash.ui.components.log_accordion.log_level'].LogLevel = MockLogLevel
