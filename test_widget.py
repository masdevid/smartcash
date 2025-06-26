import ipywidgets as widgets

# Create a simple widget
def test_widget():
    print("Creating a simple VBox widget...")
    widget = widgets.VBox()
    print(f"Widget type: {type(widget).__name__}")
    print(f"Is instance of Widget: {isinstance(widget, widgets.Widget)}")
    print(f"Widget class: {widget.__class__.__name__}")
    print(f"Widget module: {widget.__class__.__module__}")
    return widget

# Run the test
if __name__ == "__main__":
    widget = test_widget()
    print("\nWidget attributes:")
    for attr in dir(widget):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    print("\nWidget representation:", repr(widget))
