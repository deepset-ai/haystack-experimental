import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from haystack import component, logging
from haystack.core.serialization import default_from_dict, default_to_dict
from quickchart import QuickChart

logger = logging.getLogger(__name__)


@component
class PandasChartJSConverter:
    """
    A Haystack component that processes a pandas DataFrame and generates
    interactive charts using the QuickChart API (Chart.js v4 compatible).

    This component transforms tabular data into shareable chart visualizations by:
    1. Converting DataFrame data into Chart.js compatible configuration
    2. The component produces chart URLs that can be:
        - Embedded in web applications as <img> tags
        - Downloaded as PNG images
        - Used in reports and dashboards

    Usage Examples:
        ```python
        df = pd.DataFrame({
            "quarter": ["Q1", "Q2", "Q3", "Q4"],
            "sales": [100, 150, 200, 180],
            "expenses": [80, 120, 160, 140]
        })

        converter = PandasChartJSConverter(
            default_chart_type="line",
            dataset_colors=["#4BC0C0", "#FF6384"],  # Teal and pink
            width=800,
            height=400
        )

        result = converter.run(
            dataframe=df,
            chart_columns_config='{"label_column": "quarter", "data_columns": ["sales", "expenses"]}'
        )
        ```

        Adding custom metadata:
        ```python
        result = converter.run(
            dataframe=df,
            chart_columns_config='{"label_column": "date", "data_columns": ["value"]}',
            meta={"source": "Sales DB", "generated_by": "Analytics Pipeline"}
        )
        ```
    """

    def __init__(
        self,
        default_chart_type: Literal["bar", "line", "pie", "doughnut", "radar", "polarArea", "scatter"] = "bar",
        responsive_chart: bool = True,
        width: int = 500,
        height: int = 300,
        background_color: str = "white",
        version: str = "4",
        dataset_colors: Optional[List[str]] = None,
    ):
        """
        Initialize the PandasChartJSConverter.

        :param default_chart_type: The type of Chart.js chart to generate
        :param responsive_chart: Whether to make the chart responsive
        :param width: Width of the chart in pixels
        :param height: Height of the chart in pixels
        :param background_color: Background color of the chart
        :param version: Chart.js version to use (default: "4")
        :param dataset_colors: List of colors for datasets (CSS color strings, hex, rgb, rgba).
            If None, Chart.js will use default colors. Colors are applied in order to datasets.
        """
        self.default_chart_type = default_chart_type
        self.responsive_chart = responsive_chart
        self.width = width
        self.height = height
        self.background_color = background_color
        self.version = version
        self.dataset_colors = dataset_colors or []
        self.qc = QuickChart()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            default_chart_type=self.default_chart_type,
            responsive_chart=self.responsive_chart,
            width=self.width,
            height=self.height,
            background_color=self.background_color,
            version=self.version,
            dataset_colors=self.dataset_colors,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PandasChartJSConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(chart_url=str, short_url=str, chart_config=Dict[str, Any], metadata=Dict[str, Any])
    def run(
        self,
        dataframe: pd.DataFrame,
        chart_columns_config: str | Dict[str, Any],
        other_chart_config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        get_long_url: bool = True,
        get_short_url: bool = True,
        default_chart_type: Optional[Literal["bar", "line", "pie", "doughnut", "radar", "polarArea", "scatter"]] = None,
        responsive_chart: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        background_color: Optional[str] = None,
        version: Optional[str] = None,
        dataset_colors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a pandas DataFrame to generate Chart.js-compatible visualization using QuickChart.

        :param dataframe: Input pandas DataFrame to process
        :param chart_columns_config: JSON string or dict with label_column and data_columns.
            Must contain: {"label_column": "col_name", "data_columns": ["col1", "col2", ...]}
            Example: '{"label_column": "month", "data_columns": ["sales", "profit"]}'
        :param other_chart_config: Additional chart configuration to merge into the output
        :param meta: Optional dictionary of custom metadata to include in the output.
            Will be merged into the metadata field of the result.
            Example: {"source": "database", "timestamp": "2024-01-01"}
        :param get_long_url: Whether to generate a long URL (default: True).
            Long URL: Full URL with entire chart configuration encoded in query parameters.
            - Self-contained, works immediately, no expiration
            - Can be very long (thousands of characters)
            - May break in some contexts (email, chat apps)
        :param get_short_url: Whether to generate a short URL (default: True).
            Short URL: Shortened fixed-length URL that aliases the chart.
            - Easy to share, always short and readable
            - Requires API call to QuickChart service
            - May have expiration depending on service policy
            - Falls back to long URL if generation fails
        :param default_chart_type: The type of Chart.js chart to generate
        :param responsive_chart: Whether to make the chart responsive
        :param width: Width of the chart in pixels
        :param height: Height of the chart in pixels
        :param background_color: Background color of the chart
        :param version: Chart.js version to use (default: "4")
        :param dataset_colors: List of colors for datasets (CSS color strings, hex, rgb, rgba).
            If None, Chart.js will use default colors. Colors are applied in order to datasets.
        """
        empty_result = {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

        _default_chart_type = default_chart_type or self.default_chart_type
        _responsive_chart = responsive_chart if responsive_chart is not None else self.responsive_chart
        _width = width or self.width
        _height = height or self.height
        _background_color = background_color or self.background_color
        _version = version or self.version
        _dataset_colors = dataset_colors or self.dataset_colors

        # Check for empty DataFrame
        if dataframe.empty:
            logger.info("DataFrame is empty. Skipping chart generation.")
            return empty_result

        # Normalize column names to lowercase
        df_lower_columns = dataframe.copy()
        df_lower_columns.columns = [col.lower() for col in dataframe.columns]

        # Validate and parse chart_columns_config
        if not chart_columns_config:
            logger.warning("Empty `chart_columns_config` provided. Returning empty chart data.")
            return empty_result

        # Parse config (handles both string and dict inputs)
        parsed_config = self._str_to_json(chart_columns_config)
        if not parsed_config:
            logger.warning("Failed to parse `chart_columns_config`.")
            return empty_result

        # Validate required keys
        if "label_column" not in parsed_config or "data_columns" not in parsed_config:
            logger.warning("label_column or data_columns not in `chart_columns_config`.")
            return empty_result

        # Extract label and data columns
        label_column, data_columns = self._extract_columns(parsed_config)
        if not label_column or label_column not in df_lower_columns.columns:
            logger.warning(f"Label column '{label_column}' not found in DataFrame.")
            return empty_result

        # Generate Chart.js config
        chart_config = self._generate_chartjs_data(
            df_lower_columns,
            label_column,
            data_columns,
            _default_chart_type,
            _responsive_chart,
            _dataset_colors,
        )

        # Merge additional chart config
        if other_chart_config:
            if "options" in other_chart_config:
                chart_config["options"].update(other_chart_config["options"])
            if "data" in other_chart_config:
                chart_config["data"].update(other_chart_config["data"])

        # Update QuickChart instance properties
        self.qc.width = _width
        self.qc.height = _height
        self.qc.background_color = _background_color
        self.qc.version = _version
        self.qc.config = chart_config

        chart_url = ""
        short_url = ""

        if get_long_url:
            chart_url = self.qc.get_url()

        if get_short_url:
            try:
                short_url = self.qc.get_short_url()
            except Exception as e:
                logger.warning("Failed to generate short URL: {error}", error=str(e))
                # Fallback to long URL if available
                if get_long_url and chart_url:
                    short_url = chart_url

        # Create metadata
        metadata = {
            "row_count": len(df_lower_columns),
            "chart_width": _width,
            "chart_height": _height,
            "chart_type": _default_chart_type,
            "label_column": label_column,
            "data_columns": data_columns
        }

        # Merge custom metadata if provided
        if meta:
            metadata.update(meta)

        return {
            "chart_url": chart_url,
            "short_url": short_url,
            "chart_config": chart_config,
            "metadata": metadata,
        }

    def _extract_columns(self, parsed_config: Dict[str, Any]) -> tuple[str, List[str]]:
        """Extract and normalize label and data columns from parsed chart columns config."""
        try:
            lowered_config = {}
            for k, v in parsed_config.items():
                key = k.lower()
                if isinstance(v, list):
                    values = [x.lower() if isinstance(x, str) else str(x).lower() for x in v]
                else:
                    values = [v.lower() if isinstance(v, str) else str(v).lower()]
                lowered_config[key] = values

            label_column_value = lowered_config.get("label_column", [""])
            label_column = label_column_value[0] if label_column_value else ""
            data_columns = lowered_config.get("data_columns", [])

            return label_column, data_columns
        except (IndexError, TypeError, AttributeError) as e:
            logger.warning("Error processing parsed chart columns config: {error}", error=str(e))
            return "", []

    def _generate_chartjs_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        data_columns: List[str],
        chart_type: str,
        responsive_chart: bool,
        dataset_colors: List[str],
    ) -> Dict[str, Any]:
        """Generate Chart.js compatible data structure from the DataFrame."""

        labels = df[label_column].to_list()

        # Create datasets
        datasets = []
        for idx, column in enumerate(data_columns):
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping this dataset.")
                continue

            data = df[column].to_list()
            dataset = {
                "label": self._snake_to_title(column),
                "data": [self._json_serializable(item) for item in data],
            }

            # Apply color if provided
            if idx < len(dataset_colors):
                color = dataset_colors[idx]
                dataset["backgroundColor"] = color
                dataset["borderColor"] = color

            datasets.append(dataset)

        if not datasets:
            logger.warning(
                "No valid data columns found in DataFrame. All specified columns {columns} are missing. "
                "Chart will be generated with no datasets.",
                columns=data_columns
            )

        chart_config = {
            "type": chart_type,
            "data": {"labels": labels, "datasets": datasets},
            "options": {"responsive": responsive_chart},
        }

        return chart_config

    @staticmethod
    def _str_to_json(text: str | Dict[str, Any]) -> Dict[str, Any]:
        """Convert a string or dict to JSON dict."""
        if not text:
            return {}

        if isinstance(text, dict):
            return text

        # Extract JSON from text
        cleaned_text = text.strip()
        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}")

        if json_start == -1 or json_end == -1:
            logger.warning("Unable to find a dictionary in the input string")
            return {}

        cleaned_text = cleaned_text[json_start : json_end + 1]

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON string: {error}", error=str(e))
            return {}

    @staticmethod
    def _json_serializable(obj: Any) -> Any:
        """Ensure that the given object is JSON serializable."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, Decimal)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.datetime64):
            return str(obj)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [PandasChartJSConverter._json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: PandasChartJSConverter._json_serializable(value) for key, value in obj.items()}

        logger.warning("Object of type {obj_type} is not JSON serializable. Converting to string.", obj_type=type(obj))
        try:
            return str(obj)
        except Exception:
            return None

    @staticmethod
    def _snake_to_title(text: str) -> str:
        """Convert snake_case text to Title Case."""
        return text.replace("_", " ").title()
