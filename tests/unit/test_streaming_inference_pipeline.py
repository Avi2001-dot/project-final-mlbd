import pytest
import numpy as np
from unittest.mock import Mock
from src.streaming.streaming_inference_pipeline import StreamingInferencePipeline


class TestStreamingInferencePipeline:
    """Test cases for StreamingInferencePipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.predict.return_value = np.array([150.0])
        return model

    @pytest.fixture
    def pipeline(self, mock_model):
        """Create inference pipeline instance."""
        return StreamingInferencePipeline(
            model=mock_model,
            max_latency_ms=1000
        )

    def test_initialization(self, mock_model):
        """Test pipeline initialization."""
        pipeline = StreamingInferencePipeline(model=mock_model)
        assert pipeline.model == mock_model
        assert pipeline.max_latency_ms == 1000
        assert pipeline.events_processed == 0

    def test_initialization_no_model(self):
        """Test initialization with no model."""
        with pytest.raises(ValueError):
            StreamingInferencePipeline(model=None)

    def test_process_event_success(self, pipeline, mock_model):
        """Test successful event processing."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0
        }

        result = pipeline.process_event(event)

        assert result['city'] == 'Delhi'
        assert result['timestamp'] == 1000000.0
        assert result['current_aqi'] == 150.0
        assert result['predicted_aqi'] == 150.0
        assert result['features_computed'] is True
        assert 'latency_ms' in result

    def test_process_event_latency_tracking(self, pipeline, mock_model):
        """Test latency tracking."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0
        }

        pipeline.process_event(event)

        assert pipeline.events_processed == 1
        assert len(pipeline.latency_history) == 1

    def test_process_event_exceeding_latency(self, pipeline, mock_model):
        """Test event exceeding max latency."""
        # Mock slow prediction
        mock_model.predict.side_effect = lambda x: (
            __import__('time').sleep(0.002),
            np.array([150.0])
        )[1]

        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0
        }

        result = pipeline.process_event(event)

        # Latency should be tracked
        assert result['latency_ms'] > 0

    def test_process_event_invalid_timestamp(self, pipeline):
        """Test event with invalid timestamp."""
        event = {
            'city': 'Delhi',
            'timestamp': 'not_a_number',
            'aqi': 150.0
        }

        result = pipeline.process_event(event)

        assert result['predicted_aqi'] is None
        assert result['features_computed'] is False

    def test_process_event_invalid_aqi(self, pipeline):
        """Test event with invalid AQI."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 'not_a_number'
        }

        result = pipeline.process_event(event)

        assert result['predicted_aqi'] is None
        assert result['features_computed'] is False

    def test_extract_feature_vector_default(self, pipeline, mock_model):
        """Test feature vector extraction with default columns."""
        features = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0,
            'aqi_lag_1h': 140.0,
            'aqi_mean_3h': 145.0,
            'hour_of_day': 14,
            'season': 'Winter'
        }

        vector = pipeline._extract_feature_vector(features)

        assert vector is not None
        assert len(vector) > 0
        assert isinstance(vector, np.ndarray)

    def test_extract_feature_vector_specified_columns(self, mock_model):
        """Test feature vector extraction with specified columns."""
        pipeline = StreamingInferencePipeline(
            model=mock_model,
            feature_columns=['aqi_lag_1h', 'aqi_mean_3h', 'hour_of_day']
        )

        features = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0,
            'aqi_lag_1h': 140.0,
            'aqi_mean_3h': 145.0,
            'hour_of_day': 14,
            'season': 'Winter'
        }

        vector = pipeline._extract_feature_vector(features)

        assert len(vector) == 3
        assert vector[0] == 140.0
        assert vector[1] == 145.0
        assert vector[2] == 14

    def test_extract_feature_vector_missing_column(self, mock_model):
        """Test feature vector extraction with missing column."""
        pipeline = StreamingInferencePipeline(
            model=mock_model,
            feature_columns=['aqi_lag_1h', 'missing_column']
        )

        features = {
            'city': 'Delhi',
            'aqi_lag_1h': 140.0
        }

        vector = pipeline._extract_feature_vector(features)

        assert vector is None

    def test_get_latency_stats(self, pipeline):
        """Test latency statistics."""
        # Process multiple events
        for i in range(5):
            event = {
                'city': 'Delhi',
                'timestamp': 1000000.0 + (i * 3600),
                'aqi': 150.0 + i
            }
            pipeline.process_event(event)

        stats = pipeline.get_latency_stats()

        assert stats['events_processed'] == 5
        assert stats['min_latency_ms'] >= 0
        assert stats['max_latency_ms'] >= stats['min_latency_ms']
        assert stats['mean_latency_ms'] >= 0
        assert stats['median_latency_ms'] >= 0
        assert 'p95_latency_ms' in stats
        assert 'p99_latency_ms' in stats

    def test_get_latency_stats_empty(self, pipeline):
        """Test latency statistics with no events."""
        stats = pipeline.get_latency_stats()

        assert stats['events_processed'] == 0
        assert stats['min_latency_ms'] is None
        assert stats['max_latency_ms'] is None

    def test_reset_latency_tracking(self, pipeline):
        """Test resetting latency tracking."""
        # Process an event
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0
        }
        pipeline.process_event(event)

        assert pipeline.events_processed == 1

        # Reset
        pipeline.reset_latency_tracking()

        assert pipeline.events_processed == 0
        assert len(pipeline.latency_history) == 0

    def test_get_feature_computer(self, pipeline):
        """Test getting feature computer instance."""
        computer = pipeline.get_feature_computer()

        assert computer is not None
        assert computer == pipeline.feature_computer

    def test_process_multiple_cities(self, pipeline, mock_model):
        """Test processing events from multiple cities."""
        cities = ['Delhi', 'Mumbai', 'Bangalore']

        for city in cities:
            event = {
                'city': city,
                'timestamp': 1000000.0,
                'aqi': 150.0
            }
            result = pipeline.process_event(event)
            assert result['city'] == city

        assert pipeline.events_processed == 3

    def test_process_event_with_pollutants(self, pipeline, mock_model):
        """Test processing event with pollutant data."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0,
            'pollutants': {
                'pm25': 75.0,
                'pm10': 150.0
            }
        }

        result = pipeline.process_event(event)

        assert result['features_computed'] is True
        assert result['predicted_aqi'] is not None

    def test_latency_percentiles(self, pipeline, mock_model):
        """Test latency percentile calculations."""
        # Process events with varying latencies
        for i in range(100):
            event = {
                'city': 'Delhi',
                'timestamp': 1000000.0 + (i * 3600),
                'aqi': 150.0
            }
            pipeline.process_event(event)

        stats = pipeline.get_latency_stats()

        # Percentiles should be in order
        assert stats['p95_latency_ms'] >= stats['median_latency_ms']
        assert stats['p99_latency_ms'] >= stats['p95_latency_ms']
        assert stats['max_latency_ms'] >= stats['p99_latency_ms']
