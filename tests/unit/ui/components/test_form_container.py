"""
Unit tests for form_container.py
"""
import unittest
from unittest.mock import MagicMock, call
import ipywidgets as widgets

# Import test helpers
from tests.unit.ui.test_helpers import BaseTestCase

# Import the module under test
from smartcash.ui.components.form_container import (
    FormItem,
    LayoutType,
    create_form_container
)


class TestFormItem(unittest.TestCase):
    """Test cases for the FormItem class."""

    def test_init_with_defaults(self):
        """Test FormItem initialization with default values."""
        mock_widget = widgets.Button() # Use a real widget
        form_item = FormItem(widget=mock_widget)

        self.assertEqual(form_item.widget, mock_widget)
        self.assertIsNone(form_item.width)
        self.assertIsNone(form_item.height)
        self.assertIsNone(form_item.flex)
        self.assertIsNone(form_item.grid_area)
        self.assertIsNone(form_item.justify_content)
        self.assertEqual(form_item.align_items, 'stretch')

    def test_init_with_custom_values(self):
        """Test FormItem initialization with custom values."""
        mock_widget = widgets.Button() # Use a real widget
        form_item = FormItem(
            widget=mock_widget,
            width="200px",
            height="100px",
            flex="1",
            grid_area="header",
            justify_content="center",
            align_items="center"
        )

        self.assertEqual(form_item.width, "200px")
        self.assertEqual(form_item.height, "100px")
        self.assertEqual(form_item.flex, "1")
        self.assertEqual(form_item.grid_area, "header")
        self.assertEqual(form_item.justify_content, "center")
        self.assertEqual(form_item.align_items, "center")

    def test_validate_align_items(self):
        """Test the _validate_align_items method."""
        self.assertEqual(FormItem._validate_align_items("flex-start"), "flex-start")
        self.assertEqual(FormItem._validate_align_items("center"), "center")
        self.assertEqual(FormItem._validate_align_items("stretch"), "stretch")
        self.assertEqual(FormItem._validate_align_items("left"), "flex-start")
        self.assertEqual(FormItem._validate_align_items("right"), "flex-end")
        self.assertEqual(FormItem._validate_align_items("middle"), "center")
        self.assertEqual(FormItem._validate_align_items("invalid"), "stretch")
        self.assertEqual(FormItem._validate_align_items(None), "stretch")

    def test_layout_property(self):
        """Test the layout property returns the widget's layout."""
        mock_widget = widgets.Button() # Use a real widget
        self.assertIsNotNone(mock_widget.layout)
        form_item = FormItem(widget=mock_widget)
        self.assertEqual(form_item.layout, mock_widget.layout)


class TestFormContainer(BaseTestCase):
    """Test suite for the create_form_container function."""

    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # MockWidget is now a MagicMock that returns a real widget
        self.mock_widget1 = self.MockWidget()
        self.mock_widget2 = self.MockWidget()

    def tearDown(self):
        """Tear down the test environment."""
        # No need to reset mocks, BaseTestCase handles it
        super().tearDown()

    def test_create_column_layout(self):
        """Test creating a form container with column layout."""
        form = create_form_container(
            layout_type=LayoutType.COLUMN,
            container_padding='16px',
            gap='10px'
        )
        self.MockVBox.assert_called_once()
        container = form['get_form_container']()
        # Use the test case's VBox class for isinstance check
        self.assertIsInstance(container, self.VBox)
        layout = container.layout
        self.assertEqual(layout.padding, '16px')
        self.assertEqual(layout.gap, '10px')

    def test_create_row_layout(self):
        """Test creating a form container with row layout."""
        form = create_form_container(layout_type=LayoutType.ROW)
        self.MockHBox.assert_called_once()
        container = form['get_form_container']()
        # Use the test case's HBox class for isinstance check
        self.assertIsInstance(container, self.HBox)
        self.assertEqual(container.layout.flex_flow, 'row wrap')

    def test_create_grid_layout(self):
        """Test creating a form container with grid layout."""
        form = create_form_container(
            layout_type='GRID',
            grid_columns=3,
            grid_template_areas=['h h', 'm s'],
            grid_auto_flow='column'
        )
        self.MockGridBox.assert_called_once()
        container = form['get_form_container']()
        # Use the test case's GridBox class for isinstance check
        self.assertIsInstance(container, self.GridBox)
        layout = container.layout
        self.assertEqual(layout.grid_template_columns, 'repeat(3, 1fr)')
        self.assertEqual(layout.grid_template_areas, '"h h" "m s"')
        self.assertEqual(layout.grid_auto_flow, 'column')

    def test_add_item_to_column_layout(self):
        """Test adding an item to a column layout."""
        form = create_form_container(layout_type=LayoutType.COLUMN)
        form['add_item'](self.mock_widget1, width='100%', height='50px')
        form['add_item'](self.mock_widget2, flex='1')
        container = form['get_form_container']()
        self.assertEqual(len(container.children), 2)
        self.assertIn(self.mock_widget1, container.children)
        self.assertEqual(self.mock_widget1.layout.width, '100%')
        self.assertEqual(self.mock_widget1.layout.height, '50px')
        self.assertEqual(self.mock_widget2.layout.flex, '1')

    def test_add_item_with_form_item(self):
        """Test adding a FormItem instance directly."""
        form = create_form_container()
        form_item = FormItem(widget=self.mock_widget1, width='300px')
        form['add_item'](form_item)
        items = form['get_items']()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].width, '300px')
        container = form['get_form_container']()
        self.assertIn(self.mock_widget1, container.children)

    def test_set_layout_dynamically(self):
        """Test changing the layout type dynamically."""
        form = create_form_container(layout_type=LayoutType.COLUMN)
        original_container = form['get_form_container']()
        form['add_item'](self.mock_widget1)

        # Change layout to ROW
        form['set_layout']('ROW', gap='20px')
        new_container_row = form['get_form_container']()
        self.assertIsNot(new_container_row, original_container)
        # Use the test case's HBox class for isinstance check
        self.assertIsInstance(new_container_row, self.HBox)
        self.assertEqual(new_container_row.layout.gap, '20px')
        self.assertEqual(len(new_container_row.children), 1)
        self.assertIn(self.mock_widget1, new_container_row.children)
        self.MockHBox.assert_called_once()

        # Change layout to GRID
        form['set_layout']('GRID', grid_columns=3)
        new_container_grid = form['get_form_container']()
        self.assertIsNot(new_container_grid, new_container_row)
        # Use the test case's GridBox class for isinstance check
        self.assertIsInstance(new_container_grid, self.GridBox)
        self.assertEqual(new_container_grid.layout.grid_template_columns, 'repeat(3, 1fr)')
        self.assertEqual(len(new_container_grid.children), 1)
        self.assertIn(self.mock_widget1, new_container_grid.children)
        self.MockGridBox.assert_called_once()


if __name__ == '__main__':
    unittest.main()
