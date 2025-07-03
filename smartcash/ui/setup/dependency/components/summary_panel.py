"""Summary panel component for dependency management UI."""
from typing import Dict, Any, List, Optional, Callable
import ipywidgets as widgets

class DependencySummaryPanel:
    """A summary panel that displays selected packages and operation summaries."""
    
    def __init__(self):
        """Initialize the summary panel."""
        self.selected_count = 0
        self.total_count = 0
        self.install_required = False
        self.update_available = False
        self.selected_packages = []
        self.custom_packages = []
        
        # Create selection summary text
        self.summary_text = widgets.HTML(
            "<div style='padding: 10px; border-radius: 4px; background: #f8f9fa;'>"
            "<b>Ringkasan:</b> Belum ada paket yang dipilih"
            "</div>"
        )
        
        # Create the main panel for package selection summary only
        # Action summaries will be displayed in the summary container
        self.container = widgets.VBox(
            [self.summary_text],
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
        """Update the selection summary text.
        
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
                "<b>Ringkasan:</b> Belum ada paket yang dipilih"
                "</div>"
            )
        else:
            status_class = "warning" if install_required else "success"
            bg_color = "#fff3cd" if install_required else "#d1e7dd"
            text_color = "#664d03" if install_required else "#0f5132"
            
            self.summary_text.value = (
                f"<div style='padding: 10px; border-radius: 4px; background: {bg_color}; color: {text_color};'>"
                f"<b>Ringkasan:</b> {selected_count} dari {total_count} paket dipilih"
                f"<div style='margin-top: 5px;'>"
                f"{self._get_status_message(install_required, update_available)}"
                f"</div>"
                "</div>"
            )
    
    def _get_status_message(self, install_required: bool, update_available: bool) -> str:
        """Get the status message based on the current state.
        
        Args:
            install_required: Whether installation is required.
            update_available: Whether updates are available.
            
        Returns:
            Status message to display.
        """
        messages = []
        
        if install_required:
            messages.append("<span style='font-weight: bold;'>✓ Siap untuk diinstall</span>")
        
        if update_available:
            messages.append("<span style='font-weight: bold;'>⟳ Update tersedia</span>")
            
        if not messages:
            return "Semua paket sudah terinstall"
            
        return " | ".join(messages)
    
    # Button callbacks removed as they're now handled by the action container
    
    # Loading state handling is now managed by the action container
    
    # Error and success messages are now handled by the summary container
    
    def update_selection(self, package_key: str, is_checked: bool) -> None:
        """Update the selected packages list based on checkbox selection.
        
        Args:
            package_key: The key of the package
            is_checked: Whether the package is checked/selected
        """
        if is_checked and package_key not in self.selected_packages:
            self.selected_packages.append(package_key)
        elif not is_checked and package_key in self.selected_packages:
            self.selected_packages.remove(package_key)
        
        # Update the summary display
        self.update_summary(
            selected_count=len(self.selected_packages) + len(self.custom_packages),
            total_count=self.total_count,
            install_required=len(self.selected_packages) > 0 or len(self.custom_packages) > 0,
            update_available=self.update_available
        )
    
    def update_custom_packages(self, packages: List[str]) -> None:
        """Update the custom packages list.
        
        Args:
            packages: List of custom package specifications
        """
        self.custom_packages = packages
        
        # Update the summary display
        self.update_summary(
            selected_count=len(self.selected_packages) + len(self.custom_packages),
            total_count=self.total_count,
            install_required=len(self.selected_packages) > 0 or len(self.custom_packages) > 0,
            update_available=self.update_available
        )
        
    # Duplicate error and success methods removed - these are now handled by the summary container
