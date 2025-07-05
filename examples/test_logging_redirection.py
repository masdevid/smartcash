"""
Test logging redirection to log_accordion in OperationContainer.

This script demonstrates how logging is redirected to the log_accordion component
within an OperationContainer instance.
"""
import time
from datetime import datetime
from IPython.display import display
import ipywidgets as widgets

# Import the operation container
from smartcash.ui.components.operation_container import create_operation_container

# Create an operation container with all components enabled
container = create_operation_container(
    show_progress=True,
    show_dialog=True,
    show_logs=True,
    log_module_name="TestLogging"
)

# Display the container
display(container['container'])

# Test logging at different levels
container.log("This is an INFO message")
container.log("This is a WARNING message", level="WARNING")
container.log("This is an ERROR message", level="ERROR")
container.log("This is a DEBUG message", level="DEBUG")
container.log("This is a CRITICAL message", level="CRITICAL")

# Test convenience methods
container.info("Info message using convenience method")
container.warning("Warning message using convenience method")
container.error("Error message using convenience method")
container.debug("Debug message using convenience method")
container.critical("Critical message using convenience method")

# Test progress updates
container.update_progress(25, "Starting operation...")
time.sleep(1)
container.update_progress(50, "Operation in progress...")
time.sleep(1)
container.update_progress(75, "Almost done...")
time.sleep(1)
container.complete_progress("Operation completed successfully!")

# Test dialog functionality
container.show_info("Test Dialog", "This is an informational message")

# Test log clearing functionality
def clear_logs(btn):
    container.clear_logs()
    container.info("Logs cleared!")

clear_button = widgets.Button(description="Clear Logs")
clear_button.on_click(clear_logs)
display(clear_button)

# Add a test for log deduplication
container.info("This is a duplicate message")
container.info("This is a duplicate message")  # Should be deduplicated
container.info("This is a duplicate message")  # Should be deduplicated

# Test with different namespaces
container.info("[NAMESPACE1] Message from namespace 1")
container.info("[NAMESPACE2] Message from namespace 2")
container.info("[NAMESPACE1] Another message from namespace 1")
