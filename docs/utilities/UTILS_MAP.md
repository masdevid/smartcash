# SmartCash Utility Functions Reference

## Common Utilities

### worker_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [get_optimal_worker_count](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:51:0-82:90) | `operation_type: Literal['io', 'cpu', 'mixed', 'io_bound', 'cpu_bound'] = 'io'` | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Get optimal worker count based on operation type and system specs |
| [get_worker_counts_for_operations](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:84:0-98:5) | - | `Dict[str, int]` | Get optimal worker counts for all standard operations |
| [get_file_operation_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:100:0-110:73) | `file_count: int` | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Calculate optimal workers for file operations based on file count |
| [get_download_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:112:0-119:49) | - | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Get optimal workers for download operations |
| [get_rename_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:121:0-131:45) | `total_files: int` | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Calculate optimal workers for file renaming operations |
| [safe_worker_count](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:133:0-143:32) | `count: int` | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Ensure worker count is within safe limits (1-8) |
| [optimal_io_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:147:0-156:41) | - | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Get optimal worker count for I/O bound operations |
| [optimal_cpu_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:158:0-167:42) | - | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Get optimal worker count for CPU bound operations |
| [optimal_mixed_workers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/common/worker_utils.py:169:0-179:44) | - | [int](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | Get optimal worker count for mixed operations |

## UI Utilities


### validator_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [create_validation_message](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:17:0-35:8) | `message: str, is_error: bool = True` | `widgets.HTML` | Create styled validation message |
| [show_validation_message](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:37:0-46:49) | `container: widgets.Box, message: str, is_error: bool = True` | `None` | Display validation message in container |
| [clear_validation_messages](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:48:0-54:24) | `container: widgets.Box` | `None` | Clear all validation messages from container |
| [validate_required](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:58:0-63:20) | `value: Any` | `ValidationResult` | Validate field is not empty |
| [validate_numeric](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:65:0-75:57) | `value: Any` | `ValidationResult` | Validate value is numeric |
| [validate_integer](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:77:0-87:57) | `value: Any` | `ValidationResult` | Validate value is integer |
| [validate_min_value](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:89:0-100:45) | `value: Any, min_value: float` | `ValidationResult` | Validate minimum value |
| [validate_max_value](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:102:0-113:45) | `value: Any, max_value: float` | `ValidationResult` | Validate maximum value |
| [validate_range](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:115:0-122:43) | `value: Any, min_value: float, max_value: float` | `ValidationResult` | Validate value is within range |
| [validate_min_length](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:124:0-131:20) | `value: str, min_length: int` | `ValidationResult` | Validate minimum string length |
| [validate_max_length](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:133:0-140:20) | `value: str, max_length: int` | `ValidationResult` | Validate maximum string length |
| [validate_regex](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:142:0-149:20) | `value: str, pattern: str, message: str = "Format tidak valid"` | `ValidationResult` | Validate string against regex pattern |
| [validate_email](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:151:0-154:67) | `value: str` | `ValidationResult` | Validate email format |
| [validate_url](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:151:0-154:67) | `value: str` | `ValidationResult` | Validate URL format |
| [validate_file_exists](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:156:0-163:57) | `value: str` | `ValidationResult` | Validate file exists |
| [validate_directory_exists](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:165:0-172:61) | `value: str` | `ValidationResult` | Validate directory exists |
| [validate_file_extension](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:174:0-187:20) | `value: str, allowed_extensions: List[str]` | `ValidationResult` | Validate file extension |
| [validate_api_key](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:189:0-201:20) | `value: str, min_length: int = 10` | `ValidationResult` | Validate API key format |
| [validate_form](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:204:0-229:18) | `form_data: Dict[str, Any], validation_rules: Dict[str, List[Callable]]` | `Dict[str, str]` | Validate form data against validation rules |
| [create_validator](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:232:0-248:20) | `validation_func: Callable, error_message: str` | `Callable` | Create custom validator function |
| [combine_validators](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:251:0-268:20) | `*validators: Callable` | `Callable` | Combine multiple validators into one |

### widget_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [display_widget](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/widget_utils.py:6:0-33:16) | `widget: Any` | `None` | Display a widget, handling both direct widget and dictionary with common container keys |
| [safe_display](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/widget_utils.py:35:0-51:17) | `widget: Any, condition: bool = True` | `Optional[Any]` | Safely display a widget if the condition is True |

## Dataset Utilities

### dataset_utils.py
| Class/Method | Parameters | Return Type | Description |
|--------------|------------|-------------|-------------|
| [DatasetUtils](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:21:0-379:23) | `config: Dict, data_dir: Optional[str] = None, logger=None, layer_config: Optional[ILayerConfigManager] = None` | - | Utility class for dataset operations |
| [get_split_path](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:53:0-69:35) | `self, split: str` | `Path` | Get path for a specific dataset split |
| [get_class_name](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:71:0-101:30) | `self, cls_id: int` | `str` | Get class name from class ID |
| [get_layer_from_class](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:103:0-127:19) | `self, cls_id: int` | `Optional[str]` | Get layer name from class ID |
| [find_image_files](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:129:0-170:89) | `self, directory: Union[str, Path], with_labels: bool = True` | `List[Path]` | Find image files in directory |
| [get_random_sample](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:172:0-191:41) | `self, items: List, sample_size: int, seed: int = DEFAULT_RANDOM_SEED` | `List` | Get random sample from list |
| [load_image](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:193:0-223:52) | `self, image_path: Path, target_size: Optional[Tuple[int, int]] = None` | `np.ndarray` | Load image from file |
| [parse_yolo_label](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:225:0-273:22) | `self, label_path: Path` | `List[Dict]` | Parse YOLO label file |
| [get_available_layers](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:275:0-293:32) | `self, label_path: Path` | `List[str]` | Get available layers in label file |
| [get_split_statistics](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:295:0-344:20) | `self, splits: List[str] = DEFAULT_SPLITS` | `Dict[str, Dict[str, int]]` | Get basic statistics for dataset splits |
| [backup_directory](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:346:0-379:23) | `self, source_dir: Union[str, Path], suffix: Optional[str] = None` | `Optional[Path]` | Create directory backup |
| [move_invalid_files](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/dataset/utils/dataset_utils.py:381:0-417:21) | `self, source_dir: Union[str, Path], target_dir: Union[str, Path], file_list: List[Path]` | `Dict[str, int]` | Move files to target directory |

## Error Handling Utilities

### error_utils.py
| Class/Function | Parameters | Return Type | Description |
|----------------|------------|-------------|-------------|
| [ErrorHandler](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:20:0-165:23) | `logger: Optional[UILogger] = None, default_component: str = "ui", ui_components: Optional[Dict[str, Any]] = None` | - | Centralized error handling for UI components |
| [handle_error](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:52:4-77:47) | `self, error: Exception, context: Optional[ErrorContext] = None, ui_components: Optional[Dict[str, Any]] = None, show_ui: bool = True, log_level: str = "error"` | `None` | Handle an error with proper logging and UI feedback |
| [wrap_async](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:148:4-156:23) | `self, func: Callable[..., T]` | `Callable[..., T]` | Decorator for async functions to handle errors |
| [wrap_sync](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:158:4-166:23) | `self, func: Callable[..., T]` | `Callable[..., T]` | Decorator for sync functions to handle errors |
| [create_error_context](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:168:0-204:16) | `component: str = "", operation: str = "", details: Optional[Dict[str, Any]] = None, ui_components: Optional[Dict[str, Any]] = None` | `ErrorContext` | Create an ErrorContext with standardized parameter handling |
| [error_handler_scope](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:206:0-281:5) | `component: str = "ui", operation: str = "unknown", logger: Optional[UILogger] = None, ui_components: Optional[Dict[str, Any]] = None, show_ui_error: bool = True, log_level: str = "error"` | `ContextManager` | Context manager for scoped error handling |
| [with_error_handling](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:283:0-353:20) | `error_handler: Optional[ErrorHandler] = None, component: str = "ui", operation: str = "unknown", show_traceback: bool = False, ui_components: Optional[Dict[str, Any]] = None, fallback_value: Any = None, log_level: str = "error"` | `Callable` | Decorator that wraps a function with error handling logic |
| [log_errors](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:387:0-423:20) | `logger: Optional[UILogger] = None, level: str = "error", component: str = "ui", operation: str = "unknown"` | `Callable` | Simple error logging decorator with UI integration |
| [safe_ui_operation](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:447:0-480:16) | `component: str = "ui", operation: str = "unknown"` | `Callable` | Decorator untuk menjalankan operasi UI dengan error handling yang aman |
| [handle_ui_error](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:434:0-434:134) | `ui_components, error_msg` | - | One-liner utility for showing UI errors |
| [log_and_ignore](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:435:0-435:88) | `logger, error, msg=""` | - | Log and ignore errors |
| [safe_execute](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/error_utils.py:436:0-436:105) | `func, fallback=None` | - | Execute function with fallback on error |

## Dialog Utilities

### dialog_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [ensure_dialog_readiness](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:8:0-41:20) | `ui_components: Dict[str, Any], timeout: float = 2.0` | `bool` | Ensure dialog component is ready with timeout protection |
| [safe_show_dialog](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:43:0-110:46) | `ui_components: Dict[str, Any], title: str, message: str, on_confirm: Optional[Callable] = None, on_cancel: Optional[Callable] = None, confirm_text: str = "Konfirmasi", cancel_text: str = "Batal", danger_mode: bool = False, max_retries: int = 3` | `bool` | Safely show dialog with retry mechanism |
| [reset_dialog_state_safe](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:112:0-145:51) | `ui_components: Dict[str, Any]` | `None` | Reset dialog state with comprehensive cleanup |
| [check_dialog_state](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:175:0-217:20) | `ui_components: Dict[str, Any]` | `Dict[str, Any]` | Check current dialog state for debugging |
| [force_dialog_cleanup](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:219:0-263:28) | `ui_components: Dict[str, Any]` | `None` | Force cleanup dialog state for recovery from stuck state |
| [validate_dialog_integration](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/dialog_utils.py:265:0-332:20) | `ui_components: Dict[str, Any]` | `Dict[str, Any]` | Validate dialog integration for troubleshooting |

## Info Display Utilities

### info_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [create_info_accordion](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/info_utils.py:10:0-66:21) | `title: str, content: Union[str, widgets.Widget], style: str = "info", icon: Optional[str] = None, open_by_default: bool = False` | `widgets.Accordion` | Create collapsible accordion with info box |
| [style_info_content](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/info_utils.py:68:0-97:25) | `content: str, style: str = "info", padding: int = 10, border_radius: int = 5` | `str` | Style HTML content for info box |
| [create_tabbed_info](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/info_utils.py:99:0-148:15) | `tabs_content: Dict[str, str], style: str = "info"` | `widgets.Tab` | Create tabbed info box with multiple content sections |

## Metric Display Utilities

### metric_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [create_metric_display](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/metric_utils.py:13:0-56:33) | `label: str, value: Union[int, float, str], unit: Optional[str] = None, is_good: Optional[bool] = None` | `widgets.HTML` | Create a consistent metric display with styling |
| [create_result_table](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/metric_utils.py:58:0-82:18) | `data: Dict[str, Any], title: str = 'Results', highlight_max: bool = True` | `None` | Display a results table with optional highlighting |
| [plot_statistics](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/metric_utils.py:84:0-106:14) | `data: pd.DataFrame, title: str, kind: str = 'bar', figsize=(10, 6), **kwargs` | `None` | Plot statistics data with customizable visualization |
| [styled_html](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/metric_utils.py:108:0-136:7) | `content: str, bg_color: str = "#f8f9fa", text_color: str = "#2c3e50", border_color: Optional[str] = None, padding: int = 10, margin: int = 10` | `widgets.HTML` | Create custom styled HTML content |

## Logging Utilities

### ui_logger.py

#### UILogger Class
| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `__init__` | `ui_components: Dict[str, Any], name: str = "ui_logger", log_to_file: bool = False, log_dir: str = "logs", log_level: int = logging.INFO, enable_buffering: bool = False` | `None` | Initialize the UI Logger with UI components and configuration |
| `debug` | `message: str, **kwargs` | `None` | Log a debug message |
| `info` | `message: str, **kwargs` | `None` | Log an info message |
| `success` | `message: str, **kwargs` | `None` | Log a success message |
| `warning` | `message: str, **kwargs` | `None` | Log a warning message |
| `error` | `message: str, **kwargs` | `None` | Log an error message |
| `critical` | `message: str, **kwargs` | `None` | Log a critical message |
| `log` | `level: str, message: str, **kwargs` | `None` | Generic log method with custom level |
| `flush_buffered_logs` | `clear_buffer: bool = True` | `None` | Flush any buffered logs to the output |
| `capture_output` | `func: Callable, *args, **kwargs` | `Any` | Context manager to capture stdout/stderr |
| `suppress_output` | `func: Callable = None` | `Any` | Decorator to suppress output from a function |

#### Module-Level Functions
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [create_ui_logger](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/ui_logger.py:531:0-553:5) | `ui_components: Dict[str, Any], name: str = "ui_logger", log_to_file: bool = False, redirect_stdout: bool = True, log_dir: str = "logs", log_level: int = logging.INFO` | `UILogger` | Create and configure a new UILogger instance |
| [get_logger](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/ui_logger.py:555:0-587:5) | `name: str = None, log_to_file: bool = False, log_dir: str = "logs", log_level: int = logging.INFO` | `UILogger` | Get or create a logger with the given name |
| [setup_global_logging](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/ui_logger.py:593:0-626:13) | `ui_components: Dict[str, Any] = None, log_level: int = logging.INFO, log_to_file: bool = False, log_dir: str = "logs"` | `None` | Setup global logging configuration |
| [get_module_logger](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/ui_logger.py:682:0-702:5) | `name: Optional[str] = None` | `UILogger` | Get or create a UILogger instance for the specified module |
| [log_to_ui](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/ui_logger.py:668:0-679:5) | `ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None` | `None` | Log a message to the UI (legacy function) |

#### Namespace Management
| Class/Method | Description |
|--------------|-------------|
| `NamespaceManager` | Manage logging namespaces with colors and prefixes |
| `get_namespace_id` | Get the namespace ID for a module |
| `get_namespace_color` | Get the color for a namespace ID |
| `register_namespace` | Register a new namespace |

#### Log Suppression
| Class/Method | Description |
|--------------|-------------|
| `LogSuppressor` | Handle log suppression and redirection |
| `setup_aggressive_log_suppression` | Setup aggressive log suppression for backend services |
| `suppress_ml_logs` | Suppress machine learning library logs |
| `suppress_viz_logs` | Suppress visualization library logs |

### validator_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| [create_validation_message](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:17:0-35:5) | `message: str, is_error: bool = True` | `widgets.HTML` | Create styled validation message |
| [show_validation_message](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:37:0-47:61) | `container: widgets.Box, message: str, is_error: bool = True` | `None` | Display validation message in container |
| [clear_validation_messages](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:49:0-56:27) | `container: widgets.Box` | `None` | Clear all validation messages from container |
| [validate_required](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:59:0-63:21) | `value: Any` | `ValidationResult` | Validate field is not empty |
| [validate_numeric](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:65:0-74:48) | `value: Any` | `ValidationResult` | Validate value is numeric |
| [validate_integer](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:76:0-85:57) | `value: Any` | `ValidationResult` | Validate value is integer |
| [validate_min_value](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:87:0-97:41) | `value: Any, min_value: float` | `ValidationResult` | Validate minimum value |
| [validate_max_value](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:99:0-109:41) | `value: Any, max_value: float` | `ValidationResult` | Validate maximum value |
| [validate_range](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:111:0-117:47) | `value: Any, min_value: float, max_value: float` | `ValidationResult` | Validate value is within range |
| [validate_min_length](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:119:0-126:21) | `value: str, min_length: int` | `ValidationResult` | Validate minimum string length |
| [validate_max_length](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:128:0-135:21) | `value: str, max_length: int` | `ValidationResult` | Validate maximum string length |
| [validate_regex](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:137:0-144:21) | `value: str, pattern: str, message: str = "Format tidak valid"` | `ValidationResult` | Validate string against regex pattern |
| [validate_email](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:146:0-149:69) | `value: str` | `ValidationResult` | Validate email format |
| [validate_url](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:151:0-154:67) | `value: str` | `ValidationResult` | Validate URL format |
| [validate_file_exists](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:156:0-163:21) | `value: str` | `ValidationResult` | Validate file exists |
| [validate_directory_exists](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:165:0-172:21) | `value: str` | `ValidationResult` | Validate directory exists |
| [validate_file_extension](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:174:0-187:21) | `value: str, allowed_extensions: List[str]` | `ValidationResult` | Validate file extension |
| [validate_api_key](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:189:0-201:21) | `value: str, min_length: int = 10` | `ValidationResult` | Validate API key format |
| [validate_form](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:204:0-229:17) | `form_data: Dict[str, Any], validation_rules: Dict[str, List[Callable]]` | `Dict[str, str]` | Validate form data against validation rules |
| [create_validator](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:232:0-248:20) | `validation_func: Callable, error_message: str` | `Callable` | Create custom validator function |
| [combine_validators](cci:1://file:///Users/masdevid/Projects/smartcash/smartcash/ui/utils/validator_utils.py:251:0-268:29) | `*validators: Callable` | `Callable` | Combine multiple validators into one |

## Widget Utilities

### widget_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| `create_button` | `description: str, on_click: Callable, style: str = "primary", tooltip: str = ""` | `widgets.Button` | Create styled button with click handler |
| `create_dropdown` | `options: List[Tuple[str, Any]], description: str, value: Any = None, disabled: bool = False` | `widgets.Dropdown` | Create dropdown with options |
| `create_text_input` | `description: str, value: str = "", placeholder: str = "", disabled: bool = False` | `widgets.Text` | Create text input field |
| `create_int_slider` | `description: str, min: int, max: int, value: int, step: int = 1, disabled: bool = False` | `widgets.IntSlider` | Create integer slider |
| `create_float_slider` | `description: str, min: float, max: float, value: float, step: float = 0.1, disabled: bool = False` | `widgets.FloatSlider` | Create float slider |
| `create_checkbox` | `description: str, value: bool = False, disabled: bool = False` | `widgets.Checkbox` | Create checkbox |
| `create_tab` | `children: List[widgets.Widget], titles: List[str]` | `widgets.Tab` | Create tabbed interface |
| `create_accordion` | `children: List[widgets.Widget], titles: List[str]` | `widgets.Accordion` | Create accordion widget |
| `create_output` | `) -> widgets.Output` | `widgets.Output` | Create output widget for displaying content |
| `display_widgets_side_by_side` | `*widgets_to_display, width: str = "auto"` | `widgets.HBox` | Display widgets side by side |
| `create_progress_bar` | `value: float = 0, min: float = 0, max: float = 100, description: str = "Progress:"` | `widgets.FloatProgress` | Create progress bar |
| `update_progress_bar` | `progress_bar: widgets.FloatProgress, value: float, description: str = None` | `None` | Update progress bar value and description |
| `create_log_output` | `max_lines: int = 100` | `widgets.Output` | Create output widget for logging |
| `log_message` | `output_widget: widgets.Output, message: str, level: str = "info"` | `None` | Log message to output widget |
| `clear_output_widget` | `output_widget: widgets.Output` | `None` | Clear output widget content |
| `create_download_button` | `data: bytes, filename: str, description: str = "Download"` | `widgets.Button` | Create download button for data |
| `create_tooltip` | `widget: widgets.Widget, description: str` | `widgets.HBox` | Add tooltip to a widget |
| `disable_widget` | `widget: widgets.Widget` | `None` | Disable widget |
| `enable_widget` | `widget: widgets.Widget` | `None` | Enable widget |
| `create_toggle_buttons` | `options: List[Tuple[str, Any]], description: str, value: Any = None, disabled: bool = False` | `widgets.ToggleButtons` | Create toggle buttons group |

## Error Handling Utilities

### error_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| `handle_ui_errors` | `error_component_title: str = "Error", log_error: bool = True` | `Callable` | Decorator for handling UI errors gracefully |
| `show_error_dialog` | `title: str, message: str, details: str = ""` | `widgets.HTML` | Display error dialog with details |
| `create_error_summary` | `errors: Dict[str, List[str]]` | `widgets.VBox` | Create error summary widget |
| `log_exception` | `logger: Any, message: str = "An error occurred"` | `Callable` | Decorator for logging exceptions |
| `retry_on_failure` | `max_retries: int = 3, delay: float = 1.0, exceptions: Tuple[Exception] = (Exception,), logger: Any = None` | `Callable` | Decorator for retrying failed operations |
| `create_error_badge` | `count: int, color: str = "red"` | `widgets.HTML` | Create error badge with count |
| `format_error_traceback` | `exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: TracebackType` | `str` | Format exception traceback for display |
| `create_error_report` | `title: str, error: Exception, context: Dict[str, Any] = None` | `Dict[str, Any]` | Create structured error report |
| `show_warning` | `message: str, title: str = "Warning"` | `widgets.HTML` | Display warning message |
| `show_success` | `message: str, title: str = "Success"` | `widgets.HTML` | Display success message |

## Dialog Utilities

### dialog_utils.py
| Function | Parameters | Return Type | Description |
|----------|------------|-------------|-------------|
| `show_confirmation_dialog` | `message: str, title: str = "Confirm"` | `bool` | Show confirmation dialog |
| `show_input_dialog` | `prompt: str, title: str = "Input", default: str = ""` | `Optional[str]` | Show input dialog |
| `show_file_dialog` | `title: str = "Select File", filter: str = "All files (*)"` | `Optional[str]` | Show file selection dialog |
| `show_directory_dialog` | `title: str = "Select Directory"` | `Optional[str]` | Show directory selection dialog |
| `show_progress_dialog` | `title: str = "Processing", message: str = "Please wait..."` | `widgets.VBox` | Show progress dialog |
| `close_dialog` | `dialog: widgets.Widget` | `None` | Close dialog |
| `create_custom_dialog` | `title: str, content: widgets.Widget, buttons: List[widgets.Widget] = None` | `widgets.VBox` | Create custom dialog |
| `show_message_dialog` | `message: str, title: str = "Message", message_type: str = "info"` | `widgets.VBox` | Show message dialog |
| `create_loading_overlay` | `message: str = "Loading..."` | `widgets.HTML` | Create loading overlay |
| `remove_loading_overlay` | `overlay: widgets.HTML` | `None` | Remove loading overlay |