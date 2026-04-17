
import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'city': ['Delhi', 'Mumbai'],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'aqi': [150.5, 120.3],
        'pm25': [80.0, 60.0],
        'pm10': [120.0, 100.0],
        'no2': [50.0, 40.0],
        'o3': [30.0, 25.0],
        'so2': [20.0, 15.0],
        'co': [1.5, 1.2]
    })


def create_temp_api_key():
    """Create a temporary API key file."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    f.write('{"username": "test", "key": "test_key"}')
    f.close()
    return f.name


class TestCpcbDataIngestionInit:
    """Tests for CpcbDataIngestion initialization."""

    def test_init_with_missing_api_key(self):
        """Test initialization fails when API key file doesn't exist."""
        from src.data_ingestion.cpcb_ingestion import CpcbDataIngestion

        with pytest.raises(FileNotFoundError):
            CpcbDataIngestion(
                api_key_path='/nonexistent/path/credentials.json',
                dataset_name='test/dataset'
            )


class TestSchemaValidation:
    """Tests for schema validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key_path = create_temp_api_key()

        self.patcher = patch('kaggle.api.kaggle_api_extended.KaggleApi')
        self.mock_api_class = self.patcher.start()
        self.mock_instance = MagicMock()
        self.mock_api_class.return_value = self.mock_instance

        from src.data_ingestion.cpcb_ingestion import CpcbDataIngestion
        self.CpcbDataIngestion = CpcbDataIngestion

        self.ingestion = CpcbDataIngestion(
            api_key_path=self.api_key_path,
            dataset_name='test/dataset'
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        os.unlink(self.api_key_path)

    def test_validate_schema_valid_data(self):
        """Test schema validation with valid data."""
        df = create_sample_dataframe()
        assert self.ingestion._validate_schema(df) is True

    def test_validate_schema_empty_dataframe(self):
        """Test schema validation fails for empty DataFrame."""
        df = pd.DataFrame()
        assert self.ingestion._validate_schema(df) is False

    def test_validate_schema_missing_required_columns(self):
        """Test schema validation fails when required columns are missing."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'value': [100]
        })
        assert self.ingestion._validate_schema(df) is False

    def test_validate_schema_invalid_aqi_range(self):
        """Test schema validation with out-of-range AQI values."""
        df = pd.DataFrame({
            'city': ['Delhi', 'Mumbai'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'aqi': [150.5, 600.0],
            'pm25': [80.0, 60.0],
            'pm10': [120.0, 100.0],
            'no2': [50.0, 40.0],
            'o3': [30.0, 25.0],
            'so2': [20.0, 15.0],
            'co': [1.5, 1.2]
        })
        assert self.ingestion._validate_schema(df) is True

    def test_validate_schema_invalid_timestamps(self):
        """Test schema validation with invalid timestamps."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': ['invalid-date'],
            'aqi': [150.5],
            'pm25': [80.0],
            'pm10': [120.0],
            'no2': [50.0],
            'o3': [30.0],
            'so2': [20.0],
            'co': [1.5]
        })
        assert self.ingestion._validate_schema(df) is True

    def test_validate_schema_null_values(self):
        """Test schema validation with null values in critical fields."""
        df = pd.DataFrame({
            'city': ['Delhi', None],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'aqi': [150.5, 120.3],
            'pm25': [80.0, 60.0],
            'pm10': [120.0, 100.0],
            'no2': [50.0, 40.0],
            'o3': [30.0, 25.0],
            'so2': [20.0, 15.0],
            'co': [1.5, 1.2]
        })
        assert self.ingestion._validate_schema(df) is True


class TestRetryLogic:
    """Tests for exponential backoff retry logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key_path = create_temp_api_key()

        self.patcher = patch('kaggle.api.kaggle_api_extended.KaggleApi')
        self.mock_api_class = self.patcher.start()
        self.mock_instance = MagicMock()
        self.mock_api_class.return_value = self.mock_instance

        from src.data_ingestion.cpcb_ingestion import CpcbDataIngestion
        self.ingestion = CpcbDataIngestion(
            api_key_path=self.api_key_path,
            dataset_name='test/dataset'
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        os.unlink(self.api_key_path)

    def test_retry_with_backoff_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        result = self.ingestion._retry_with_backoff(
            func=mock_func, max_retries=3, initial_delay=1
        )
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_with_backoff_success_after_failures(self):
        """Test retry succeeds after initial failures."""
        mock_func = Mock(side_effect=[
            Exception("Error 1"), Exception("Error 2"), "success"
        ])
        with patch('time.sleep'):
            result = self.ingestion._retry_with_backoff(
                func=mock_func, max_retries=3, initial_delay=1
            )
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_all_failures(self):
        """Test retry returns None after all attempts fail."""
        mock_func = Mock(side_effect=Exception("Error"))
        with patch('time.sleep'):
            result = self.ingestion._retry_with_backoff(
                func=mock_func, max_retries=3, initial_delay=1
            )
        assert result is None
        assert mock_func.call_count == 3

    def test_retry_exponential_backoff_delays(self):
        """Test exponential backoff calculates correct delays."""
        mock_func = Mock(side_effect=Exception("Error"))
        sleep_times = []

        def mock_sleep(duration):
            sleep_times.append(duration)

        with patch('time.sleep', side_effect=mock_sleep):
            self.ingestion._retry_with_backoff(
                func=mock_func, max_retries=4, initial_delay=30
            )
        assert sleep_times == [30, 60, 120]


class TestErrorHandling:
    """Tests for error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key_path = create_temp_api_key()

        self.patcher = patch('kaggle.api.kaggle_api_extended.KaggleApi')
        self.mock_api_class = self.patcher.start()
        self.mock_instance = MagicMock()
        self.mock_api_class.return_value = self.mock_instance

        from src.data_ingestion.cpcb_ingestion import (
            CpcbDataIngestion,
            CpcbConnectionError
        )
        self.CpcbDataIngestion = CpcbDataIngestion
        self.CpcbConnectionError = CpcbConnectionError

        self.ingestion = CpcbDataIngestion(
            api_key_path=self.api_key_path,
            dataset_name='test/dataset'
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        os.unlink(self.api_key_path)

    def test_cpcb_connection_error_message(self):
        """Test CpcbConnectionError has proper message."""
        error = self.CpcbConnectionError("Test error message")
        assert str(error) == "Test error message"

    def test_download_dataset_no_csv_files(self):
        """Test download fails when no CSV files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'data.txt'), 'w') as f:
                f.write('test')
            with patch.object(self.ingestion.api, 'dataset_download_files'):
                with pytest.raises(FileNotFoundError):
                    self.ingestion._download_dataset(temp_dir)

    def test_download_dataset_api_error(self):
        """Test download handles API errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.ingestion.api.dataset_download_files.side_effect = (
                Exception("API Error")
            )
            with pytest.raises(self.CpcbConnectionError):
                self.ingestion._download_dataset(temp_dir)
