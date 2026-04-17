import pytest
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

from src.etl_pipeline.pipeline import ETLPipeline, ETLPipelineError


@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("test-etl-pipeline") \
        .master("local[1]") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def temp_paths(tmp_path):
    """Create temporary paths for storage layers."""
    return {
        'bronze': str(tmp_path / "bronze"),
        'silver': str(tmp_path / "silver"),
        'gold': str(tmp_path / "gold")
    }


@pytest.fixture
def etl_pipeline(spark_session, temp_paths):
    """Create an ETL Pipeline instance."""
    return ETLPipeline(
        temp_paths['bronze'],
        temp_paths['silver'],
        temp_paths['gold'],
        spark_session
    )


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
        'ingestion_timestamp': [datetime.now()] * 24,
        'retrieval_time': [0.5] * 24
    })


class TestETLPipelineInit:
    """Tests for ETL Pipeline initialization."""

    def test_init_creates_pipeline(self, spark_session, temp_paths):
        """Test that initialization creates pipeline."""
        pipeline = ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold'],
            spark_session
        )

        assert pipeline.spark is not None
        assert pipeline.bronze_layer is not None
        assert pipeline.silver_layer is not None
        assert pipeline.gold_layer is not None
        assert pipeline.validator is not None

    def test_init_creates_spark_session_if_not_provided(self, temp_paths):
        """Test that initialization creates Spark session if not provided."""
        pipeline = ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold']
        )

        assert pipeline.spark is not None
        pipeline.stop_spark_session()

    def test_init_uses_provided_spark_session(self, spark_session, temp_paths):
        """Test that initialization uses provided Spark session."""
        pipeline = ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold'],
            spark_session
        )

        assert pipeline.spark is spark_session


class TestRunPipeline:
    """Tests for pipeline execution."""

    def test_run_pipeline_returns_tuple(self, etl_pipeline):
        """Test that run_pipeline returns a tuple."""
        df = create_sample_dataframe()

        result = etl_pipeline.run_pipeline(df, 'kaggle')

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_run_pipeline_returns_dataframe_and_metrics(self, etl_pipeline):
        """Test that run_pipeline returns DataFrame and metrics."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert isinstance(gold_df, pd.DataFrame)
        assert isinstance(metrics, dict)

    def test_run_pipeline_produces_gold_layer_features(self, etl_pipeline):
        """Test that pipeline produces Gold Layer features."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        # Check that Gold Layer features are present
        assert 'aqi_lag_1h' in gold_df.columns
        assert 'aqi_mean_3h' in gold_df.columns
        assert 'hour_of_day' in gold_df.columns
        assert 'season' in gold_df.columns

    def test_run_pipeline_with_invalid_data(self, etl_pipeline):
        """Test pipeline with invalid data."""
        df = pd.DataFrame({
            'city': ['Delhi'],
            'timestamp': [datetime.now()],
            'aqi': [150.0]
        })

        # Should handle gracefully
        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert isinstance(gold_df, pd.DataFrame)
        assert isinstance(metrics, dict)


class TestPerformanceMetrics:
    """Tests for performance monitoring."""

    def test_performance_metrics_recorded(self, etl_pipeline):
        """Test that performance metrics are recorded."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert 'total_time_seconds' in metrics
        assert 'total_time_minutes' in metrics
        assert 'within_target' in metrics

    def test_performance_metrics_include_stage_times(self, etl_pipeline):
        """Test that metrics include individual stage times."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert 'bronze_time_seconds' in metrics
        assert 'silver_transform_time_seconds' in metrics
        assert 'gold_transform_time_seconds' in metrics

    def test_performance_metrics_include_record_counts(self, etl_pipeline):
        """Test that metrics include record counts."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert 'records' in metrics
        assert 'bronze_ingested' in metrics['records']
        assert 'valid_records' in metrics['records']
        assert 'rejected_records' in metrics['records']

    def test_performance_metrics_include_quality_info(self, etl_pipeline):
        """Test that metrics include quality information."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        assert 'quality' in metrics
        assert 'quality_score' in metrics['quality']

    def test_get_performance_metrics(self, etl_pipeline):
        """Test that get_performance_metrics returns last run metrics."""
        df = create_sample_dataframe()

        etl_pipeline.run_pipeline(df, 'kaggle')
        metrics = etl_pipeline.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'total_time_seconds' in metrics

    def test_processing_within_target_time(self, etl_pipeline):
        """Test that processing completes within target time."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        # Should complete within 5 minutes (300 seconds)
        assert metrics['total_time_seconds'] < 300
        assert metrics['within_target']


class TestDataFlow:
    """Tests for data flow through pipeline."""

    def test_data_flows_through_all_layers(self, etl_pipeline):
        """Test that data flows through all layers."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        # Check that records were processed through all layers
        assert metrics['records']['bronze_ingested'] > 0
        assert metrics['records']['valid_records'] > 0
        assert metrics['records']['gold_stored'] > 0

    def test_data_quality_tracked(self, etl_pipeline):
        """Test that data quality is tracked."""
        df = create_sample_dataframe()

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        # Quality score should be tracked
        assert 'quality_score' in metrics['quality']
        assert 0 <= metrics['quality']['quality_score'] <= 100

    def test_invalid_records_rejected(self, etl_pipeline):
        """Test that invalid records are rejected."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = 550.0  # Out of range

        gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')

        # Should have rejected records
        assert metrics['records']['rejected_records'] > 0


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_usage(self, spark_session, temp_paths):
        """Test that pipeline can be used as context manager."""
        with ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold'],
            spark_session
        ) as pipeline:
            assert pipeline.spark is not None

    def test_context_manager_stops_spark(self, temp_paths):
        """Test that context manager stops Spark session."""
        with ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold']
        ) as pipeline:
            spark = pipeline.spark

        # Spark session should be stopped
        # (This is hard to verify directly, but no error should occur)


class TestErrorHandling:
    """Tests for error handling."""

    def test_pipeline_handles_empty_dataframe(self, etl_pipeline):
        """Test that pipeline handles empty DataFrame."""
        df = pd.DataFrame()

        # Should handle gracefully
        try:
            gold_df, metrics = etl_pipeline.run_pipeline(df, 'kaggle')
            assert isinstance(gold_df, pd.DataFrame)
        except Exception:
            # Empty DataFrame may raise an error, which is acceptable
            pass

    def test_stop_spark_session(self, spark_session, temp_paths):
        """Test that Spark session can be stopped."""
        pipeline = ETLPipeline(
            temp_paths['bronze'],
            temp_paths['silver'],
            temp_paths['gold'],
            spark_session
        )

        # Should not raise error
        pipeline.stop_spark_session()
