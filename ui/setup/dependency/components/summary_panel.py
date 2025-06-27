"""Summary panel component for dependency management UI."""
from typing import Dict, Any, List, Optional, Callable
import ipywidgets as widgets

class DependencySummaryPanel:
    """A summary panel that displays selected packages and actions."""
    
    def __init__(self):
        """Initialize the summary panel."""
        self.selected_count = 0
        self.total_count = 0
        self.install_required = False
        self.update_available = False
        
        # Create summary text
        self.summary_text = widgets.HTML(
            "<div style='padding: 10px; border-radius: 4px; background: #f8f9fa;'>"
            "<b>Summary:</b> No packages selected"
            "</div>"
        )
        
        # Create action buttons
        self.install_button = widgets.Button(
            description="Install Selected",
            button_style="success",
            icon="download",
            layout=widgets.Layout(width="180px"),
            tooltip="Install selected packages",
        )
        
        self.update_button = widgets.Button(
            description="Update All",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(width="150px"),
            tooltip="Update all installed packages",
            disabled=True,
        )
        
        self.save_button = widgets.Button(
            description="Save Configuration",
            button_style="primary",
            icon="save",
            layout=widgets.Layout(width="180px"),
            tooltip="Save current package selection",
        )
        
        # Create button container
        self.button_container = widgets.HBox(
            [self.install_button, self.update_button, self.save_button],
            layout=widgets.Layout(
                margin="10px 0",
                justify_content="flex-start",
                flex_wrap="wrap",
                gap="10px",
            ),
        )
        
        # Create the main panel
        self.panel = widgets.VBox(
            [
                self.summary_text,
                self.button_container,
            ],
            layout=widgets.Layout(
                width="100%",
                padding="10px",
                border="1px solid #dee2e6",
                border_radius="4px",
                margin="10px 0",
            ),
        )
    
    def update_summary(
        self,
        selected_count: int,
        total_count: int,
        install_required: bool = False,
        update_available: bool = False,
    ) -> None:
        """Update the summary text and button states.
        
        Args:
            selected_count: Number of selected packages.
            total_count: Total number of packages.
            install_required: Whether installation is required.
            update_available: Whether updates are available.
        """
        self.selected_count = selected_count
        self.total_count = total_count
        self.install_required = install_required
        self.update_available = update_available
        
        # Update summary text
        if selected_count == 0:
            self.summary_text.value = (
                "<div style='padding: 10px; border-radius: 4px; background: #f8f9fa;'>"
                "<b>Summary:</b> No packages selected"
                "</div>"
            )
        else:
            status = []
            if install_required:
                status.append("<span style='color: #28a745;'>Installation required</span>")
            if update_available:
                status.append("<span style='color: #17a2b8;'>Updates available</span>")
            
            status_text = " | ".join(status) if status else "Ready to install"
            
            self.summary_text.value = (
                f"<div style='padding: 10px; border-radius: 4px; background: #f8f9fa;'>"
                f"<b>Summary:</b> {selected_count} of {total_count} packages selected"
                f"<div style='margin-top: 5px;'>{status_text}</div>"
                "</div>"
            )
        
        # Update button states
        self.install_button.disabled = selected_count == 0
        self.update_button.disabled = not update_available
    
    def on_install_click(self, callback: Callable) -> None:
        """Set the callback for the install button."""
        self.install_button.on_click(lambda _: callback())
    
    def on_update_click(self, callback: Callable) -> None:
        """Set the callback for the update button."""
        self.update_button.on_click(lambda _: callback())
    
    def on_save_click(self, callback: Callable) -> None:
        """Set the callback for the save button."""
        self.save_button.on_click(lambda _: callback())
    
    def show_loading(self, loading: bool = True) -> None:
        """Show or hide loading state on buttons."""
        self.install_button.disabled = loading
        self.update_button.disabled = loading or not self.update_available
        self.save_button.disabled = loading
        
        if loading:
            self.install_button.icon = "spinner fa-spin"
            self.update_button.icon = "spinner fa-spin"
            self.save_button.icon = "spinner fa-spin"
        else:
            self.install_button.icon = "download"
            self.update_button.icon = "refresh"
            self.save_button.icon = "save"
    
    def show_error(self, message: str) -> None:
        """Show an error message in the summary panel."""
        self.summary_text.value = (
            f"<div style='padding: 10px; border-radius: 4px; background: #f8d7da; color: #721c24;'>"
            f"<b>Error:</b> {message}"
            "</div>"
        )
    
    def show_success(self, message: str) -> None:
        """Show a success message in the summary panel."""
        self.summary_text.value = (
            f"<div style='padding: 10px; border-radius: 4px; background: #d4edda; color: #155724;'>"
            f"<b>Success:</b> {message}"
            "</div>"
        )
