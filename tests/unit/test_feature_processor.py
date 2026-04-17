import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

from src.feature_engineering.feature_processor import (
    FeatureProcessor,
    FeatureProcessorError
)


@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("test_feature_processor") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with time-series data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    data = {
        'city': ['Delhi'] * 50 + ['Mumbai'] * 50,
        'timestamp': list(dates) + list(dates),
        'aqi': np.random.uniform(50, 200, 100).tolist() * 2,
        'pm25': np.random.uniform(20, 100, 100).tolist() * 2,
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@pytest.fixture
def feature_processor(spark_session):
    """Create a FeatureProcessor instance."""
    return FeatureProcessor(
        spark=spark_session,
        lag_offsets=[1, 3, 6],
        rolling_windows=[3, 6]
    )


class TestFeatureProcessorInit:
    """Tests for FeatureProcessor initialization."""
    
    def test_init_with_valid_spark_session(self, spark_session):
        """Test initialization with valid Spark session."""
        processor = FeatureProcessor(spark=spark_session)
        assert processor.spark is not None
        assert processor.lag_offsets == [1, 3, 6, 12, 24]
        assert processor.rolling_windows == [3, 6, 12, 24]
    
    def test_init_with_custom_offsets(self, spark_session):
        """Test initialization with custom lag offsets."""
        custom_lags = [1, 2, 4]
        processor = FeatureProcessor(
            spark=spark_session,
            lag_offsets=custom_lags
        )
        assert processor.lag_offsets == custom_lags
    
    def test_init_with_custom_windows(self, spark_session):
        """Test initialization with custom rolling windows."""
        custom_windows = [2, 4, 8]
        processor = FeatureProcessor(
            spark=spark_session,
            rolling_windows=custom_windows
        )
        assert processor.rolling_windows == custom_windows
    
    def test_init_with_none_spark_raises_error(self):
        """Test that None Spark session raises error."""
        with pytest.raises(FeatureProcessorError):
            FeatureProcessor(spark=None)


class TestLagFeatures:
    """Tests for lag feature computation."""
    
    def test_lag_features_computed(self, feature_processor, sample_dataframe):
        """Test that lag features are computed."""
        result = feature_processor.process(sample_dataframe)
        
        for lag in feature_processor.lag_offsets:
            assert f'aqi_lag_{lag}h' in result.columns
    
    def test_lag_features_correct_values(self, feature_processor):
        """Test that lag features have correct values."""
        # Create simple data with known values
        df = pd.DataFrame({
            'city': ['Delhi'] * 10,
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'aqi': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = feature_processor.process(df)
        
        # Check lag_1h
        assert pd.isna(result.iloc[0]['aqi_lag_1h'])
        assert result.iloc[1]['aqi_lag_1h'] == 10
        assert result.iloc[2]['aqi_lag_1h'] == 20
    
    def test_lag_features_per_city(self, feature_processor):
        """Test that lag features are computed per city."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 5 + ['Mumbai'] * 5,
            'timestamp': list(pd.date_range('2024-01-01', periods=5, freq='H')) * 2,
            'aqi': [10, 20, 30, 40, 50, 100, 110, 120, 130, 140]
        })
        
        result = feature_processor.process(df)
        
        # Delhi lag_1h at index 1 should be 10
        assert result.iloc[1]['aqi_lag_1h'] == 10
        
        # Mumbai lag_1h at index 6 should be 100 (not 50 from Delhi)
        assert result.iloc[6]['aqi_lag_1h'] == 100


class TestRollingStatistics:
    """Tests for rolling statistics computation."""
    
    def test_rolling_statistics_computed(self, feature_processor, sample_dataframe):
        """Test that rolling statistics are computed."""
        result = feature_processor.process(sample_dataframe)
        
        for window in feature_processor.rolling_windows:
            assert f'aqi_mean_{window}h' in result.columns
            assert f'aqi_std_{window}h' in result.columns
            assert f'aqi_min_{window}h' in result.columns
            assert f'aqi_max_{window}h' in result.columns
    
    def test_rolling_mean_correctness(self, feature_processor):
        """Test that rolling mean is computed correctly."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 10,
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'aqi': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = feature_processor.process(df)
        
        # Check rolling mean for window=3
        # At index 2: mean of [10, 20, 30] = 20
        assert result.iloc[2]['aqi_mean_3h'] == 20.0
        
        # At index 3: mean of [20, 30, 40] = 30
        assert result.iloc[3]['aqi_mean_3h'] == 30.0
    
    def test_rolling_statistics_per_city(self, feature_processor):
        """Test that rolling statistics are computed per city."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 5 + ['Mumbai'] * 5,
            'timestamp': list(pd.date_range('2024-01-01', periods=5, freq='H')) * 2,
            'aqi': [10, 20, 30, 40, 50, 100, 110, 120, 130, 140]
        })
        
        result = feature_processor.process(df)
        
        # Delhi mean at index 2 should be mean of [10, 20, 30]
        delhi_mean = result.iloc[2]['aqi_mean_3h']
        assert delhi_mean == 20.0
        
        # Mumbai mean at index 7 should be mean of [100, 110, 120]
        mumbai_mean = result.iloc[7]['aqi_mean_3h']
        assert mumbai_mean == 110.0


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""
    
    def test_temporal_features_extracted(self, feature_processor, sample_dataframe):
        """Test that temporal features are extracted."""
        result = feature_processor.process(sample_dataframe)
        
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'is_weekend' in result.columns
    
    def test_hour_of_day_correct(self, feature_processor):
        """Test that hour_of_day is extracted correctly."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 24,
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
            'aqi': np.random.uniform(50, 200, 24)
        })
        
        result = feature_processor.process(df)
        
        # Check hours 0-23
        for i in range(24):
            assert result.iloc[i]['hour_of_day'] == i
    
    def test_day_of_week_correct(self, feature_processor):
        """Test that day_of_week is extracted correctly."""
        # 2024-01-01 is a Monday (day_of_week=0)
        df = pd.DataFrame({
            'city': ['Delhi'] * 7,
            'timestamp': pd.date_range('2024-01-01', periods=7, freq='D'),
            'aqi': np.random.uniform(50, 200, 7)
        })
        
        result = feature_processor.process(df)
        
        # Check days of week
        expected_days = [0, 1, 2, 3, 4, 5, 6]  # Mon-Sun
        for i, expected_day in enumerate(expected_days):
            assert result.iloc[i]['day_of_week'] == expected_day
    
    def test_month_correct(self, feature_processor):
        """Test that month is extracted correctly."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 12,
            'timestamp': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'aqi': np.random.uniform(50, 200, 12)
        })
        
        result = feature_processor.process(df)
        
        # Check months 1-12
        for i in range(12):
            assert result.iloc[i]['month'] == i + 1
    
    def test_is_weekend_correct(self, feature_processor):
        """Test that is_weekend is computed correctly."""
        # 2024-01-06 is Saturday, 2024-01-07 is Sunday
        df = pd.DataFrame({
            'city': ['Delhi'] * 7,
            'timestamp': pd.date_range('2024-01-01', periods=7, freq='D'),
            'aqi': np.random.uniform(50, 200, 7)
        })
        
        result = feature_processor.process(df)
        
        # Monday-Friday should be 0, Saturday-Sunday should be 1
        expected_weekend = [0, 0, 0, 0, 0, 1, 1]
        for i, expected in enumerate(expected_weekend):
            assert result.iloc[i]['is_weekend'] == expected


class TestSeasonalIndicators:
    """Tests for seasonal indicator computation."""
    
    def test_seasonal_indicators_computed(self, feature_processor, sample_dataframe):
        """Test that seasonal indicators are computed."""
        result = feature_processor.process(sample_dataframe)
        assert 'season' in result.columns
    
    def test_season_winter(self, feature_processor):
        """Test that winter season is assigned correctly."""
        # December, January, February
        df = pd.DataFrame({
            'city': ['Delhi'] * 3,
            'timestamp': [
                pd.Timestamp('2024-12-01'),
                pd.Timestamp('2024-01-01'),
                pd.Timestamp('2024-02-01')
            ],
            'aqi': [100, 100, 100]
        })
        
        result = feature_processor.process(df)
        
        assert result.iloc[0]['season'] == 'Winter'
        assert result.iloc[1]['season'] == 'Winter'
        assert result.iloc[2]['season'] == 'Winter'
    
    def test_season_summer(self, feature_processor):
        """Test that summer season is assigned correctly."""
        # March, April, May
        df = pd.DataFrame({
            'city': ['Delhi'] * 3,
            'timestamp': [
                pd.Timestamp('2024-03-01'),
                pd.Timestamp('2024-04-01'),
                pd.Timestamp('2024-05-01')
            ],
            'aqi': [100, 100, 100]
        })
        
        result = feature_processor.process(df)
        
        assert result.iloc[0]['season'] == 'Summer'
        assert result.iloc[1]['season'] == 'Summer'
        assert result.iloc[2]['season'] == 'Summer'
    
    def test_season_monsoon(self, feature_processor):
        """Test that monsoon season is assigned correctly."""
        # June, July, August, September
        df = pd.DataFrame({
            'city': ['Delhi'] * 4,
            'timestamp': [
                pd.Timestamp('2024-06-01'),
                pd.Timestamp('2024-07-01'),
                pd.Timestamp('2024-08-01'),
                pd.Timestamp('2024-09-01')
            ],
            'aqi': [100, 100, 100, 100]
        })
        
        result = feature_processor.process(df)
        
        for i in range(4):
            assert result.iloc[i]['season'] == 'Monsoon'
    
    def test_season_post_monsoon(self, feature_processor):
        """Test that post-monsoon season is assigned correctly."""
        # October, November
        df = pd.DataFrame({
            'city': ['Delhi'] * 2,
            'timestamp': [
                pd.Timestamp('2024-10-01'),
                pd.Timestamp('2024-11-01')
            ],
            'aqi': [100, 100]
        })
        
        result = feature_processor.process(df)
        
        assert result.iloc[0]['season'] == 'Post-Monsoon'
        assert result.iloc[1]['season'] == 'Post-Monsoon'


class TestFeatureProcessorIntegration:
    """Integration tests for FeatureProcessor."""
    
    def test_process_complete_pipeline(self, feature_processor, sample_dataframe):
        """Test complete feature processing pipeline."""
        result = feature_processor.process(sample_dataframe)
        
        # Check that all feature types are present
        assert len(result) == len(sample_dataframe)
        
        # Check lag features
        for lag in feature_processor.lag_offsets:
            assert f'aqi_lag_{lag}h' in result.columns
        
        # Check rolling statistics
        for window in feature_processor.rolling_windows:
            assert f'aqi_mean_{window}h' in result.columns
            assert f'aqi_std_{window}h' in result.columns
            assert f'aqi_min_{window}h' in result.columns
            assert f'aqi_max_{window}h' in result.columns
        
        # Check temporal features
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'is_weekend' in result.columns
        
        # Check seasonal
        assert 'season' in result.columns
    
    def test_get_feature_columns(self, feature_processor):
        """Test get_feature_columns method."""
        features = feature_processor.get_feature_columns()
        
        # Check lag features
        for lag in feature_processor.lag_offsets:
            assert f'aqi_lag_{lag}h' in features
        
        # Check rolling statistics
        for window in feature_processor.rolling_windows:
            assert f'aqi_mean_{window}h' in features
            assert f'aqi_std_{window}h' in features
            assert f'aqi_min_{window}h' in features
            assert f'aqi_max_{window}h' in features
        
        # Check temporal features
        assert 'hour_of_day' in features
        assert 'day_of_week' in features
        assert 'month' in features
        assert 'is_weekend' in features
        
        # Check seasonal
        assert 'season' in features
    
    def test_process_with_missing_values(self, feature_processor):
        """Test processing with missing values."""
        df = pd.DataFrame({
            'city': ['Delhi'] * 10,
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'aqi': [10, 20, np.nan, 40, 50, np.nan, 70, 80, 90, 100]
        })
        
        result = feature_processor.process(df)
        
        # Should still process without errors
        assert len(result) == len(df)
        assert 'aqi_lag_1h' in result.columns
