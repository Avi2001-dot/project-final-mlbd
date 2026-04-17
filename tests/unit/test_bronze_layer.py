
import os
import tempfile
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.etl_pipeline.bronze_layer import BronzeLayer, BronzeLayerError


@pytest.fixture
def spark_session():
    """Create a mock Spark session for testing."""
    spark = MagicMock()
    spark.createDataFrame = MagicMock(return_value=MagicMock())
    spark.read.parquet = MagicMock(return_value=MagicMock())
    return spark


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def bronze_layer(spark_session, temp_storage_dir):
    """Create a BronzeLayer instance for testing."""
    return BronzeLayer(temp_storage_dir, spark_session)


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'city': ['Delhi', 'Mumbai', 'Bangalore'],
        'timestamp': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 12:00:00'
        ]),
        'aqi': [150.5, 120.3, 95.0],
        'pm25': [80.0, 60.0, 45.0],
        'pm10': [120.0, 100.0, 80.0],
        'no2': [50.0, 40.0, 30.0],
        'o3': [30.0, 25.0, 20.0],
        'so2': [20.0, 15.0, 10.0],
        'co': [1.5, 1.2, 0.9]
    })


class TestBronzeLayerInit:
    """Tests for BronzeLayer initialization."""

    def test_init_creates_storage_directory(self, spark_session):
        """Test initialization creates storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, 'bronze')
            bronze = BronzeLayer(storage_path, spark_session)

            assert os.path.exists(storage_path)
            assert bronze.storage_path == storage_path
            assert bronze.spark == spark_session

    def test_init_with_existing_directory(self, spark_session, temp_storage_dir):
        """Test initialization with existing directory."""
        bronze = BronzeLayer(temp_storage_dir, spark_session)

        assert os.path.exists(temp_storage_dir)
        assert bronze.storage_path == temp_storage_dir

    def test_init_logs_initialization(self, spark_session, temp_storage_dir):
        """Test initialization logs message."""
        with patch('src.etl_pipeline.bronze_layer.get_logger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            bronze = BronzeLayer(temp_storage_dir, spark_session)

            mock_logger_instance.info.assert_called()


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_schema_valid_data(self, bronze_layer):
        """Test schema validation with valid data."""
        df = create_sample_dataframe()
        assert bronze_layer._validate_schema(df) is True

    def test_validate_schema_empty_dataframe(self, bronze_layer):
        """Test schema validation fails for empty DataFrame."""
        df = pd.DataFrame()
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_missing_city_column(self, bronze_layer):
        """Test schema validation fails when city column is missing."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
            'aqi': [150.5]
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_missing_timestamp_column(self, bronze_layer):
        """Test schema validation fails when timestamp column is missing."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'aqi': [150.5]
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_missing_aqi_column(self, bronze_layer):
        """Test schema validation fails when aqi column is missing."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00'])
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_invalid_timestamp_type(self, bronze_layer):
        """Test schema validation fails with invalid timestamp type."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': ['not-a-date'],
            'aqi': [150.5]
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_invalid_aqi_type(self, bronze_layer):
        """Test schema validation fails with invalid aqi type."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
            'aqi': ['not-a-number']
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_invalid_city_type(self, bronze_layer):
        """Test schema validation fails with invalid city type."""
        df = pd.DataFrame({
            'city': [123],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
            'aqi': [150.5]
        })
        assert bronze_layer._validate_schema(df) is False

    def test_validate_schema_aqi_out_of_range(self, bronze_layer):
        """Test schema validation with out-of-range AQI values."""
        df = pd.DataFrame({
            'city': ['Delhi', 'Mumbai'],
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 11:00:00'
            ]),
            'aqi': [150.5, 600.0]
        })
        # Should still return True but log warning
        assert bronze_layer._validate_schema(df) is True

    def test_validate_schema_negative_aqi(self, bronze_layer):
        """Test schema validation with negative AQI values."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
            'aqi': [-10.0]
        })
        # Should still return True but log warning
        assert bronze_layer._validate_schema(df) is True


class TestMetadataAddition:
    """Tests for metadata addition."""

    def test_add_metadata_columns(self, bronze_layer):
        """Test metadata columns are added correctly."""
        df = create_sample_dataframe()
        df_with_metadata = bronze_layer._add_metadata(df, 'kaggle')

        assert 'source' in df_with_metadata.columns
        assert 'ingestion_timestamp' in df_with_metadata.columns
        assert 'retrieval_time' in df_with_metadata.columns

    def test_add_metadata_source_value(self, bronze_layer):
        """Test source metadata is set correctly."""
        df = create_sample_dataframe()
        df_with_metadata = bronze_layer._add_metadata(df, 'iqair')

        assert (df_with_metadata['source'] == 'iqair').all()

    def test_add_metadata_ingestion_timestamp(self, bronze_layer):
        """Test ingestion timestamp is set to current time."""
        df = create_sample_dataframe()
        before_time = datetime.now()
        df_with_metadata = bronze_layer._add_metadata(df, 'kaggle')
        after_time = datetime.now()

        assert (df_with_metadata['ingestion_timestamp'] >= before_time).all()
        assert (df_with_metadata['ingestion_timestamp'] <= after_time).all()

    def test_add_metadata_retrieval_time(self, bronze_layer):
        """Test retrieval time is set to 0.0."""
        df = create_sample_dataframe()
        df_with_metadata = bronze_layer._add_metadata(df, 'kaggle')

        assert (df_with_metadata['retrieval_time'] == 0.0).all()

    def test_add_metadata_preserves_original_columns(self, bronze_layer):
        """Test original columns are preserved."""
        df = create_sample_dataframe()
        original_columns = set(df.columns)
        df_with_metadata = bronze_layer._add_metadata(df, 'kaggle')

        for col in original_columns:
            assert col in df_with_metadata.columns

    def test_add_metadata_does_not_modify_original(self, bronze_layer):
        """Test original DataFrame is not modified."""
        df = create_sample_dataframe()
        original_columns = set(df.columns)
        df_with_metadata = bronze_layer._add_metadata(df, 'kaggle')

        assert set(df.columns) == original_columns
        assert 'source' not in df.columns


class TestDataStorage:
    """Tests for data storage functionality."""

    def test_store_data_with_invalid_schema(self, bronze_layer):
        """Test store_data raises error with invalid schema."""
        df = pd.DataFrame({'invalid': [1, 2, 3]})

        with pytest.raises(ValueError):
            bronze_layer.store_data(df, 'kaggle')

    def test_store_data_with_empty_dataframe(self, bronze_layer):
        """Test store_data raises error with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError):
            bronze_layer.store_data(df, 'kaggle')


class TestDataRetrieval:
    """Tests for data retrieval functionality."""

    def test_read_data_nonexistent_source(self, bronze_layer):
        """Test read_data with nonexistent source returns empty DataFrame."""
        result = bronze_layer.read_data(source='nonexistent')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_read_data_without_source_filter(self, bronze_layer):
        """Test read_data without source filter."""
        # Mock the Spark read operation
        mock_sdf = MagicMock()
        mock_sdf.toPandas.return_value = pd.DataFrame()
        bronze_layer.spark.read.parquet.return_value = mock_sdf
        
        result = bronze_layer.read_data()

        assert isinstance(result, pd.DataFrame)


class TestErrorHandling:
    """Tests for error handling."""

    def test_store_data_invalid_schema_raises_error(self, bronze_layer):
        """Test store_data raises ValueError for invalid schema."""
        df = pd.DataFrame({'invalid': [1, 2, 3]})

        with pytest.raises(ValueError):
            bronze_layer.store_data(df, 'kaggle')

    def test_store_data_empty_dataframe_raises_error(self, bronze_layer):
        """Test store_data raises ValueError for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError):
            bronze_layer.store_data(df, 'kaggle')

    def test_read_data_handles_missing_path(self, bronze_layer):
        """Test read_data handles missing storage path gracefully."""
        result = bronze_layer.read_data(source='nonexistent')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_bronze_layer_error_message(self):
        """Test BronzeLayerError has proper message."""
        error = BronzeLayerError("Test error message")
        assert str(error) == "Test error message"
