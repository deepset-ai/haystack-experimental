# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import date, datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from haystack_experimental.components.preprocessors.pandas_to_chartjs_converter import PandasChartJSConverter


class TestPandasChartJSConverter:
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Sales": [100, 150, 200],
            "Expenses": [80, 120, 160],
        })

    @pytest.fixture
    def converter(self) -> PandasChartJSConverter:
        return PandasChartJSConverter()

    def test_init_default_parameters(self):
        converter = PandasChartJSConverter()
        assert converter.default_chart_type == "bar"
        assert converter.responsive_chart is True
        assert converter.width == 500
        assert converter.height == 300
        assert converter.background_color == "white"
        assert converter.version == "4"
        assert converter.dataset_colors == []
        assert converter.qc is not None

    def test_init_custom_parameters(self):
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        converter = PandasChartJSConverter(
            default_chart_type="line",
            responsive_chart=False,
            width=1000,
            height=800,
            background_color="transparent",
            version="3",
            dataset_colors=colors,
        )
        assert converter.default_chart_type == "line"
        assert converter.responsive_chart is False
        assert converter.width == 1000
        assert converter.height == 800
        assert converter.background_color == "transparent"
        assert converter.version == "3"
        assert converter.dataset_colors == colors

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_basic_success(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        chart_config = '{"label_column": "Date", "data_columns": ["Sales", "Expenses"]}'
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=chart_config)

        assert result["chart_url"] == "http://example.com/chart.png"
        assert result["short_url"] == "http://short.url/abc"
        assert result["chart_config"]["type"] == "bar"
        assert result["chart_config"]["data"]["labels"] == ["2023-01-01", "2023-01-02", "2023-01-03"]
        assert len(result["chart_config"]["data"]["datasets"]) == 2
        assert result["metadata"]["row_count"] == 3
        assert result["metadata"]["label_column"] == "date"
        assert result["metadata"]["data_columns"] == ["sales", "expenses"]

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_with_dict_config(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        chart_config = {"label_column": "Date", "data_columns": ["Sales"]}
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=chart_config)

        assert result["chart_url"] != ""
        assert len(result["chart_config"]["data"]["datasets"]) == 1

    def test_run_empty_dataframe(self, converter):
        empty_df = pd.DataFrame()
        result = converter.run(dataframe=empty_df, chart_columns_config='{"label_column": "Date", "data_columns": ["Sales"]}')
        assert result == {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

    @pytest.mark.parametrize("config", ["", None])
    def test_run_empty_or_none_config(self, converter, sample_dataframe, config):
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config)
        assert result == {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

    def test_run_invalid_json_config(self, converter, sample_dataframe):
        result = converter.run(dataframe=sample_dataframe, chart_columns_config='{"label_column": "Date", invalid}')
        assert result == {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

    @pytest.mark.parametrize("config", [
        '{"data_columns": ["Sales"]}',  # missing label_column
        '{"label_column": "Date"}',  # missing data_columns
    ])
    def test_run_missing_required_keys(self, converter, sample_dataframe, config):
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config)
        assert result == {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

    def test_run_nonexistent_label_column(self, converter, sample_dataframe):
        result = converter.run(dataframe=sample_dataframe, chart_columns_config='{"label_column": "NonExistent", "data_columns": ["Sales"]}')
        assert result == {"chart_url": "", "short_url": "", "chart_config": {}, "metadata": {}}

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_nonexistent_data_column(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        result = converter.run(dataframe=sample_dataframe, chart_columns_config='{"label_column": "Date", "data_columns": ["Sales", "NonExistent"]}')

        assert len(result["chart_config"]["data"]["datasets"]) == 1
        assert result["chart_config"]["data"]["datasets"][0]["label"] == "Sales"

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_with_custom_metadata(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        custom_meta = {"source": "test_db", "author": "test_user"}
        result = converter.run(
            dataframe=sample_dataframe,
            chart_columns_config='{"label_column": "Date", "data_columns": ["Sales"]}',
            meta=custom_meta
        )

        assert result["metadata"]["source"] == "test_db"
        assert result["metadata"]["author"] == "test_user"
        assert result["metadata"]["row_count"] == 3

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_with_other_chart_config(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        other_config = {
            "options": {"plugins": {"title": {"display": True, "text": "Sales Chart"}}},
            "data": {"custom_field": "custom_value"}
        }
        result = converter.run(
            dataframe=sample_dataframe,
            chart_columns_config='{"label_column": "Date", "data_columns": ["Sales"]}',
            other_chart_config=other_config
        )

        assert result["chart_config"]["options"]["plugins"]["title"]["text"] == "Sales Chart"
        assert result["chart_config"]["data"]["custom_field"] == "custom_value"

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_url_generation_options(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/long"
        mock_qc_instance.get_short_url.return_value = "http://short.url"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        config = '{"label_column": "Date", "data_columns": ["Sales"]}'

        # Long URL only
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config, get_long_url=True, get_short_url=False)
        assert result["chart_url"] == "http://example.com/long"
        assert result["short_url"] == ""

        # Short URL only
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config, get_long_url=False, get_short_url=True)
        assert result["chart_url"] == ""
        assert result["short_url"] == "http://short.url"

        # No URLs
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config, get_long_url=False, get_short_url=False)
        assert result["chart_url"] == ""
        assert result["short_url"] == ""

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_short_url_error_handling(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.side_effect = Exception("Service error")
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        config = '{"label_column": "Date", "data_columns": ["Sales"]}'

        # With fallback
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config, get_long_url=True, get_short_url=True)
        assert result["short_url"] == "http://example.com/chart.png"

        # Without fallback
        result = converter.run(dataframe=sample_dataframe, chart_columns_config=config, get_long_url=False, get_short_url=True)
        assert result["short_url"] == ""

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_parameter_overrides(self, mock_quickchart, converter, sample_dataframe):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        colors = ["#FF0000", "#00FF00"]
        result = converter.run(
            dataframe=sample_dataframe,
            chart_columns_config='{"label_column": "Date", "data_columns": ["Sales", "Expenses"]}',
            default_chart_type="line",
            responsive_chart=False,
            width=1200,
            height=900,
            background_color="transparent",
            version="3",
            dataset_colors=colors
        )

        assert result["chart_config"]["type"] == "line"
        assert result["chart_config"]["options"]["responsive"] is False
        assert result["metadata"]["chart_width"] == 1200
        assert result["metadata"]["chart_height"] == 900
        assert mock_qc_instance.background_color == "transparent"
        assert mock_qc_instance.version == "3"
        assert result["chart_config"]["data"]["datasets"][0]["backgroundColor"] == "#FF0000"
        assert result["chart_config"]["data"]["datasets"][1]["backgroundColor"] == "#00FF00"

    @patch("haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.QuickChart")
    def test_run_case_insensitive_columns(self, mock_quickchart, converter):
        mock_qc_instance = Mock()
        mock_qc_instance.get_url.return_value = "http://example.com/chart.png"
        mock_qc_instance.get_short_url.return_value = "http://short.url/abc"
        mock_quickchart.return_value = mock_qc_instance
        converter.qc = mock_qc_instance

        df = pd.DataFrame({"DATE": ["2023-01-01"], "SALES": [100]})
        result = converter.run(dataframe=df, chart_columns_config='{"label_column": "date", "data_columns": ["sales"]}')

        assert result["metadata"]["label_column"] == "date"
        assert result["metadata"]["data_columns"] == ["sales"]

    def test_extract_columns_success(self, converter):
        # Normal case
        label, data_cols = converter._extract_columns({"label_column": "Date", "data_columns": ["Sales", "Expenses"]})
        assert label == "date"
        assert data_cols == ["sales", "expenses"]

        # Non-list data_columns
        label, data_cols = converter._extract_columns({"label_column": "Date", "data_columns": "Sales"})
        assert label == "date"
        assert data_cols == ["sales"]

        # Numeric values
        label, data_cols = converter._extract_columns({"label_column": 123, "data_columns": [456]})
        assert label == "123"
        assert data_cols == ["456"]

        # Empty config
        label, data_cols = converter._extract_columns({})
        assert label == ""
        assert data_cols == []

        # None values
        label, data_cols = converter._extract_columns({"label_column": None, "data_columns": None})
        assert isinstance(label, str)
        assert isinstance(data_cols, list)

    def test_str_to_json(self, converter):
        # Dict input
        assert converter._str_to_json({"key": "value"}) == {"key": "value"}

        # Valid JSON string
        assert converter._str_to_json('{"label_column": "Date"}') == {"label_column": "Date"}

        # JSON embedded in text
        assert converter._str_to_json('Text {"key": "value"} more') == {"key": "value"}

        # Invalid inputs
        assert converter._str_to_json('{"key": invalid}') == {}
        assert converter._str_to_json("no json here") == {}
        assert converter._str_to_json("") == {}
        assert converter._str_to_json(None) == {}

    def test_json_serializable_datetime_types(self):
        assert PandasChartJSConverter._json_serializable(datetime(2023, 1, 15, 10, 30, 45)) == "2023-01-15T10:30:45"
        assert PandasChartJSConverter._json_serializable(date(2023, 1, 15)) == "2023-01-15"
        result = PandasChartJSConverter._json_serializable(np.datetime64("2023-01-15T10:30:45"))
        assert "2023-01-15" in result

    def test_json_serializable_collections(self):
        assert PandasChartJSConverter._json_serializable([1, 2.5, np.int32(42)]) == [1, 2.5, 42]
        assert PandasChartJSConverter._json_serializable((1, 2, 3)) == [1, 2, 3]
        assert PandasChartJSConverter._json_serializable(np.array([1, 2, 3])) == [1, 2, 3]
        assert PandasChartJSConverter._json_serializable({"a": 1, "b": np.int32(42)}) == {"a": 1, "b": 42}

    def test_json_serializable_nested_structures(self):
        nested = {"list": [1, np.int32(3)], "dict": {"a": np.float64(1.5)}, "tuple": (4, 5)}
        result = PandasChartJSConverter._json_serializable(nested)
        assert result == {"list": [1, 3], "dict": {"a": 1.5}, "tuple": [4, 5]}

    def test_json_serializable_unknown_type(self):
        class CustomClass:
            def __str__(self):
                return "custom_string"

        assert PandasChartJSConverter._json_serializable(CustomClass()) == "custom_string"

        class BadClass:
            def __str__(self):
                raise ValueError("Cannot convert")

        assert PandasChartJSConverter._json_serializable(BadClass()) is None

    def test_snake_to_title(self):
        assert PandasChartJSConverter._snake_to_title("sales") == "Sales"
        assert PandasChartJSConverter._snake_to_title("total_sales") == "Total Sales"

    def test_generate_chartjs_data(self, converter):
        df = pd.DataFrame({"date": ["2023-01-01"], "sales": [100], "expenses": [80]})

        # Basic generation
        result = converter._generate_chartjs_data(df, "date", ["sales"], "bar", True, [])
        assert result["type"] == "bar"
        assert result["data"]["labels"] == ["2023-01-01"]
        assert len(result["data"]["datasets"]) == 1
        assert result["data"]["datasets"][0]["label"] == "Sales"
        assert result["options"]["responsive"] is True

        # With colors
        colors = ["#FF0000", "#00FF00"]
        result = converter._generate_chartjs_data(df, "date", ["sales", "expenses"], "line", False, colors)
        assert result["data"]["datasets"][0]["backgroundColor"] == "#FF0000"
        assert result["data"]["datasets"][1]["backgroundColor"] == "#00FF00"
        assert result["options"]["responsive"] is False

        # More columns than colors
        result = converter._generate_chartjs_data(df, "date", ["sales", "expenses"], "bar", True, ["#FF0000"])
        assert "backgroundColor" in result["data"]["datasets"][0]
        assert "backgroundColor" not in result["data"]["datasets"][1]

    def test_to_dict(self):
        converter = PandasChartJSConverter(
            default_chart_type="line",
            responsive_chart=False,
            width=800,
            height=600,
            background_color="transparent",
            version="3",
            dataset_colors=["#FF0000", "#00FF00"],
        )

        data = converter.to_dict()

        assert data["type"] == "haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.PandasChartJSConverter"
        assert data["init_parameters"]["default_chart_type"] == "line"
        assert data["init_parameters"]["responsive_chart"] is False
        assert data["init_parameters"]["width"] == 800
        assert data["init_parameters"]["height"] == 600
        assert data["init_parameters"]["background_color"] == "transparent"
        assert data["init_parameters"]["version"] == "3"
        assert data["init_parameters"]["dataset_colors"] == ["#FF0000", "#00FF00"]

    def test_to_dict_default_parameters(self):
        converter = PandasChartJSConverter()
        data = converter.to_dict()

        assert data["init_parameters"]["default_chart_type"] == "bar"
        assert data["init_parameters"]["responsive_chart"] is True
        assert data["init_parameters"]["width"] == 500
        assert data["init_parameters"]["height"] == 300
        assert data["init_parameters"]["background_color"] == "white"
        assert data["init_parameters"]["version"] == "4"
        assert data["init_parameters"]["dataset_colors"] == []

    def test_from_dict(self):
        data = {
            "type": "haystack_experimental.components.preprocessors.pandas_to_chartjs_converter.PandasChartJSConverter",
            "init_parameters": {
                "default_chart_type": "pie",
                "responsive_chart": False,
                "width": 1000,
                "height": 800,
                "background_color": "black",
                "version": "3",
                "dataset_colors": ["#AABBCC", "#DDEEFF"],
            },
        }

        converter = PandasChartJSConverter.from_dict(data)

        assert converter.default_chart_type == "pie"
        assert converter.responsive_chart is False
        assert converter.width == 1000
        assert converter.height == 800
        assert converter.background_color == "black"
        assert converter.version == "3"
        assert converter.dataset_colors == ["#AABBCC", "#DDEEFF"]
        assert converter.qc is not None

    def test_to_dict_from_dict_roundtrip(self):
        original = PandasChartJSConverter(
            default_chart_type="scatter",
            responsive_chart=True,
            width=1200,
            height=900,
            background_color="white",
            version="4",
            dataset_colors=["#123456"],
        )

        data = original.to_dict()
        restored = PandasChartJSConverter.from_dict(data)

        assert restored.default_chart_type == original.default_chart_type
        assert restored.responsive_chart == original.responsive_chart
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.background_color == original.background_color
        assert restored.version == original.version
        assert restored.dataset_colors == original.dataset_colors
