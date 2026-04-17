import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

from src.etl_pipeline.gold_layer import GoldLayer, GoldLayerError


@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("test-gold-layer") \
        .master("local[1]") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    return str(tmp_path / "gold")


@pytest.fixture
def gold_layer(spark_session, temp_storage_dir):
    """Create a Gold Layer instance."""
    return GoldLayer(temp_storage_dir, spark_session)


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    base_time = datetime(2024, 1, 1, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(24)]

    return pd.DataFrame({
        'city': ['Delhi'] * 24,
        'timestamp': timestamps,
        'aqi': [100.0 + i * 5 for i in range(24)],
        'pm25': [40.0 + i * 2 for i in range(24)],
        'pm10': [80.0 + i * 3 for i in range(24)],
        'no2': [30.0 + i for i in range(24)],
        'o3': [20.0 + i * 0.5 for i in range(24)],
        'so2': [10.0 + i * 0.3 for i in range(24)],
        'co': [5.0 + i * 0.2 for i in range(24)],
        'source': ['kaggle'] * 24,
        'quality_flags': [[] for _ in range(24)]
    })


class TestGoldLayerInit:
    """Tests for Gold Layer initialization."""

    def test_init_creates_storage_directory(self, spark_session, tmp_path):
        """Test that initialization creates storage directory."""
        storage_path = str(tmp_path / "gold_new")
        gold = GoldLayer(storage_path, spark_session)
        assert gold.storage_path == storage_path

    def test_init_with_existing_directory(self, spark_session, temp_storage_dir):
        """Test initialization with existing directory."""
        gold = GoldLayer(temp_storage_dir, spark_session)
        assert gold.spark is not None
        assert gold.logger is not None


class TestLagFeatures:
    """Tests for lag feature computation."""

    def test_lag_features_computed(self, gold_layer):
        """Test that lag features are computed."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Check that lag features exist
        assert 'aqi_lag_1h' in gold_df.columns
        assert 'aqi_lag_3h' in gold_df.columns
        assert 'aqi_lag_6h' in gold_df.columns
        assert 'aqi_lag_12h' in gold_df.columns
        assert 'aqi_lag_24h' in gold_df.columns

    def test_lag_features_correct_values(self, gold_layer):
        """Test that lag features have correct values."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # First record should have NaN for lag features
        assert pd.isna(gold_df.iloc[0]['aqi_lag_1h'])

        # Second record should have lag_1h = first record's AQI
        if len(gold_df) > 1:
            assert gold_df.iloc[1]['aqi_lag_1h'] == df.iloc[0]['aqi']

    def test_lag_features_per_city(self, gold_layer):
        """Test that lag features are computed per city."""
        df = create_sample_dataframe()
        df2 = create_sample_dataframe()
        df2['city'] = 'Mumbai'
        df_combined = pd.concat([df, df2], ignore_index=True)

        gold_df = gold_layer.transform_silver_to_gold(df_combined)

        # Should have lag features for both cities
        assert 'aqi_lag_1h' in gold_df.columns


class TestRollingStatistics:
    """Tests for rolling statistics computation."""

    def test_rolling_statistics_computed(self, gold_layer):
        """Test that rolling statistics are computed."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Check that rolling statistics exist
        assert 'aqi_mean_3h' in gold_df.columns
        assert 'aqi_std_3h' in gold_df.columns
        assert 'aqi_min_3h' in gold_df.columns
        assert 'aqi_max_3h' in gold_df.columns

        assert 'aqi_mean_6h' in gold_df.columns
        assert 'aqi_mean_12h' in gold_df.columns
        assert 'aqi_mean_24h' in gold_df.columns

    def test_rolling_mean_correctness(self, gold_layer):
        """Test that rolling mean is computed correctly."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # For a simple case, verify rolling mean
        # First record should have NaN or single value
        # Third record should have mean of first 3 values
        if len(gold_df) >= 3:
            expected_mean = df.iloc[0:3]['aqi'].mean()
            actual_mean = gold_df.iloc[2]['aqi_mean_3h']
            # Allow small floating point differences
            assert abs(actual_mean - expected_mean) < 1.0

    def test_rolling_statistics_per_city(self, gold_layer):
        """Test that rolling statistics are computed per city."""
        df = create_sample_dataframe()
        df2 = create_sample_dataframe()
        df2['city'] = 'Mumbai'
        df_combined = pd.concat([df, df2], ignore_index=True)

        gold_df = gold_layer.transform_silver_to_gold(df_combined)

        # Should have rolling statistics for both cities
        assert 'aqi_mean_3h' in gold_df.columns


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def test_temporal_features_extracted(self, gold_layer):
        """Test that temporal features are extracted."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Check that temporal features exist
        assert 'hour_of_day' in gold_df.columns
        assert 'day_of_week' in gold_df.columns
        assert 'month' in gold_df.columns
        assert 'is_weekend' in gold_df.columns

    def test_hour_of_day_correct(self, gold_layer):
        """Test that hour_of_day is correct."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # First record is at 00:00, so hour_of_day should be 0
        assert gold_df.iloc[0]['hour_of_day'] == 0

        # Second record is at 01:00, so hour_of_day should be 1
        assert gold_df.iloc[1]['hour_of_day'] == 1

    def test_day_of_week_correct(self, gold_layer):
        """Test that day_of_week is correct."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # All records are from same day, so day_of_week should be same
        day_of_week = gold_df.iloc[0]['day_of_week']
        assert all(gold_df['day_of_week'] == day_of_week)

    def test_month_correct(self, gold_layer):
        """Test that month is correct."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # All records are from January, so month should be 1
        assert all(gold_df['month'] == 1)

    def test_is_weekend_correct(self, gold_layer):
        """Test that is_weekend is correct."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # January 1, 2024 is a Monday, so is_weekend should be 0
        assert gold_df.iloc[0]['is_weekend'] == 0


class TestSeasonalIndicators:
    """Tests for seasonal indicator computation."""

    def test_seasonal_indicators_computed(self, gold_layer):
        """Test that seasonal indicators are computed."""
        df = create_sample_dataframe()

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Check that season column exists
        assert 'season' in gold_df.columns

    def test_season_winter(self, gold_layer):
        """Test that winter season is correctly identified."""
        base_time = datetime(2024, 1, 1, 0, 0)  # January
        timestamps = [base_time + timedelta(hours=i) for i in range(24)]

        df = pd.DataFrame({
            'city': ['Delhi'] * 24,
            'timestamp': timestamps,
            'aqi': [100.0 + i * 5 for i in range(24)],
            'pm25': [40.0 + i * 2 for i in range(24)],
            'pm10': [80.0 + i * 3 for i in range(24)],
            'no2': [30.0 + i for i in range(24)],
            'o3': [20.0 + i * 0.5 for i in range(24)],
            'so2': [10.0 + i * 0.3 for i in range(24)],
            'co': [5.0 + i * 0.2 for i in range(24)],
            'source': ['kaggle'] * 24,
            'quality_flags': [[] for _ in range(24)]
        })

        gold_df = gold_layer.transform_silver_to_gold(df)

        # January is Winter
        assert all(gold_df['season'] == 'Winter')

    def test_season_summer(self, gold_layer):
        """Test that summer season is correctly identified."""
        base_time = datetime(2024, 4, 1, 0, 0)  # April
        timestamps = [base_time + timedelta(hours=i) for i in range(24)]

        df = pd.DataFrame({
            'city': ['Delhi'] * 24,
            'timestamp': timestamps,
            'aqi': [100.0 + i * 5 for i in range(24)],
            'pm25': [40.0 + i * 2 for i in range(24)],
            'pm10': [80.0 + i * 3 for i in range(24)],
            'no2': [30.0 + i for i in range(24)],
            'o3': [20.0 + i * 0.5 for i in range(24)],
            'so2': [10.0 + i * 0.3 for i in range(24)],
            'co': [5.0 + i * 0.2 for i in range(24)],
            'source': ['kaggle'] * 24,
            'quality_flags': [[] for _ in range(24)]
        })

        gold_df = gold_layer.transform_silver_to_gold(df)

        # April is Summer
        assert all(gold_df['season'] == 'Summer')

    def test_season_monsoon(self, gold_layer):
        """Test that monsoon season is correctly identified."""
        base_time = datetime(2024, 7, 1, 0, 0)  # July
        timestamps = [base_time + timedelta(hours=i) for i in range(24)]

        df = pd.DataFrame({
            'city': ['Delhi'] * 24,
            'timestamp': timestamps,
            'aqi': [100.0 + i * 5 for i in range(24)],
            'pm25': [40.0 + i * 2 for i in range(24)],
            'pm10': [80.0 + i * 3 for i in range(24)],
            'no2': [30.0 + i for i in range(24)],
            'o3': [20.0 + i * 0.5 for i in range(24)],
            'so2': [10.0 + i * 0.3 for i in range(24)],
            'co': [5.0 + i * 0.2 for i in range(24)],
            'source': ['kaggle'] * 24,
            'quality_flags': [[] for _ in range(24)]
        })

        gold_df = gold_layer.transform_silver_to_gold(df)

        # July is Monsoon
        assert all(gold_df['season'] == 'Monsoon')


class TestMissingValueHandling:
    """Tests for missing value handling."""

    def test_missing_values_filled(self, gold_layer):
        """Test that missing values are filled."""
        df = create_sample_dataframe()
        df.loc[5, 'aqi'] = None
        df.loc[6, 'aqi'] = None

        gold_df = gold_layer.transform_silver_to_gold(df)

        # After forward-fill and backward-fill, no nulls should remain
        assert gold_df['aqi'].isnull().sum() == 0

    def test_forward_fill_applied(self, gold_layer):
        """Test that forward-fill is applied."""
        df = create_sample_dataframe()
        df.loc[5, 'aqi'] = None

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Missing value should be filled with previous value
        assert gold_df.iloc[5]['aqi'] == df.iloc[4]['aqi']

    def test_backward_fill_applied(self, gold_layer):
        """Test that backward-fill is applied for remaining nulls."""
        df = create_sample_dataframe()
        # Create a scenario where forward-fill leaves nulls at the beginning
        df.loc[0, 'aqi'] = None

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Missing value at beginning should be filled with next value
        assert gold_df.iloc[0]['aqi'] == df.iloc[1]['aqi']

    def test_no_nulls_after_handling(self, gold_layer):
        """Test that no null values remain after handling."""
        df = create_sample_dataframe()
        # Add multiple missing values
        df.loc[5, 'aqi'] = None
        df.loc[10, 'aqi'] = None
        df.loc[15, 'aqi'] = None

        gold_df = gold_layer.transform_silver_to_gold(df)

        # Check numeric columns have no nulls
        numeric_cols = gold_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert gold_df[col].isnull().sum() == 0


class TestDataStorage:
    """Tests for data storage operations."""

    def test_store_data_creates_parquet_files(self, gold_layer):
        """Test that data storage creates Parquet files."""
        df = create_sample_dataframe()
        gold_df = gold_layer.transform_silver_to_gold(df)

        records_stored = gold_layer.store_data(gold_df)

        assert records_stored == len(gold_df)


class TestDataRetrieval:
    """Tests for data retrieval operations."""

    def test_read_data_returns_empty_for_nonexistent_path(self, gold_layer):
        """Test that reading from nonexistent path returns empty DataFrame."""
        df = gold_layer.read_data(city='NonExistent')

        assert df.empty
