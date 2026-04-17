import pytest
import numpy as np
from datetime import datetime
from src.streaming.streaming_feature_computer import StreamingFeatureComputer


class TestStreamingFeatureComputer:
    """Test cases for StreamingFeatureComputer."""

    @pytest.fixture
    def computer(self):
        """Create feature computer instance."""
        return StreamingFeatureComputer(window_size_hours=24)

    def test_initialization(self, computer):
        """Test feature computer initialization."""
        assert computer.window_size_hours == 24
        assert computer.window_size_seconds == 24 * 3600
        assert len(computer.city_buffers) == 0

    def test_compute_features_basic(self, computer):
        """Test basic feature computation."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0
        }

        features = computer.compute_features(event)

        assert features['city'] == 'Delhi'
        assert features['timestamp'] == 1000000.0
        assert features['aqi'] == 150.0
        assert 'hour_of_day' in features
        assert 'day_of_week' in features
        assert 'month' in features
        assert 'is_weekend' in features
        assert 'season' in features

    def test_compute_features_missing_fields(self, computer):
        """Test feature computation with missing required fields."""
        event = {'city': 'Delhi', 'timestamp': 1000000.0}  # Missing aqi

        with pytest.raises(KeyError):
            computer.compute_features(event)

    def test_compute_features_invalid_timestamp(self, computer):
        """Test feature computation with invalid timestamp."""
        event = {
            'city': 'Delhi',
            'timestamp': 'not_a_number',
            'aqi': 150.0
        }

        with pytest.raises(ValueError):
            computer.compute_features(event)

    def test_compute_features_invalid_aqi(self, computer):
        """Test feature computation with invalid AQI."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 'not_a_number'
        }

        with pytest.raises(ValueError):
            computer.compute_features(event)

    def test_lag_features(self, computer):
        """Test lag feature computation."""
        base_time = 1000000.0

        # Add events at different times
        events = [
            {'city': 'Delhi', 'timestamp': base_time, 'aqi': 100.0},
            {'city': 'Delhi', 'timestamp': base_time + 3600, 'aqi': 110.0},
            {'city': 'Delhi', 'timestamp': base_time + 7200, 'aqi': 120.0},
        ]

        for event in events:
            computer.compute_features(event)

        # Compute features for new event
        new_event = {
            'city': 'Delhi',
            'timestamp': base_time + 10800,
            'aqi': 130.0
        }

        features = computer.compute_features(new_event)

        # Check lag features
        assert features['aqi_lag_1h'] == 120.0  # 1 hour ago
        assert features['aqi_lag_3h'] == 100.0  # 3 hours ago

    def test_rolling_statistics(self, computer):
        """Test rolling statistics computation."""
        base_time = 1000000.0

        # Add events with known AQI values
        aqi_values = [100.0, 110.0, 120.0, 130.0]
        for i, aqi in enumerate(aqi_values):
            event = {
                'city': 'Delhi',
                'timestamp': base_time + (i * 3600),
                'aqi': aqi
            }
            computer.compute_features(event)

        # Compute features for new event
        new_event = {
            'city': 'Delhi',
            'timestamp': base_time + (4 * 3600),
            'aqi': 140.0
        }

        features = computer.compute_features(new_event)

        # Check rolling statistics for 3-hour window
        # Should include last 3 events: 120, 130 (within 3 hours)
        assert features['aqi_mean_3h'] == pytest.approx(125.0, rel=0.1)
        assert features['aqi_min_3h'] == 120.0
        assert features['aqi_max_3h'] == 130.0

    def test_temporal_features(self, computer):
        """Test temporal feature extraction."""
        # Use a known timestamp: 2024-01-15 10:30:00 UTC
        timestamp = 1705334400.0  # 2024-01-15 00:00:00 UTC
        timestamp += 10 * 3600 + 30 * 60  # Add 10:30

        event = {
            'city': 'Delhi',
            'timestamp': timestamp,
            'aqi': 150.0
        }

        features = computer.compute_features(event)

        assert features['hour_of_day'] == 10
        assert features['day_of_week'] == 0  # Monday
        assert features['month'] == 1

    def test_seasonal_features(self, computer):
        """Test seasonal feature extraction."""
        # Test different months
        base_timestamp = 1704067200.0  # 2024-01-01 00:00:00 UTC

        # January (Winter)
        event = {
            'city': 'Delhi',
            'timestamp': base_timestamp,
            'aqi': 150.0
        }
        features = computer.compute_features(event)
        assert features['season'] == 'Winter'

        # April (Summer)
        april_timestamp = base_timestamp + (90 * 24 * 3600)
        event = {
            'city': 'Delhi',
            'timestamp': april_timestamp,
            'aqi': 150.0
        }
        features = computer.compute_features(event)
        assert features['season'] == 'Summer'

        # July (Monsoon)
        july_timestamp = base_timestamp + (180 * 24 * 3600)
        event = {
            'city': 'Delhi',
            'timestamp': july_timestamp,
            'aqi': 150.0
        }
        features = computer.compute_features(event)
        assert features['season'] == 'Monsoon'

    def test_sliding_window_maintenance(self, computer):
        """Test sliding window buffer maintenance."""
        base_time = 1000000.0
        window_size = 24 * 3600

        # Add events within window
        for i in range(10):
            event = {
                'city': 'Delhi',
                'timestamp': base_time + (i * 3600),
                'aqi': 100.0 + i
            }
            computer.compute_features(event)

        # Check buffer size
        stats = computer.get_buffer_stats('Delhi')
        assert stats['buffer_size'] == 10

        # Add event outside window
        event = {
            'city': 'Delhi',
            'timestamp': base_time + window_size + 3600,
            'aqi': 200.0
        }
        computer.compute_features(event)

        # Old events should be removed
        stats = computer.get_buffer_stats('Delhi')
        assert stats['buffer_size'] < 10

    def test_multiple_cities(self, computer):
        """Test feature computation for multiple cities."""
        base_time = 1000000.0

        # Add events for different cities
        for city in ['Delhi', 'Mumbai', 'Bangalore']:
            event = {
                'city': city,
                'timestamp': base_time,
                'aqi': 150.0
            }
            computer.compute_features(event)

        # Check buffers for each city
        assert 'Delhi' in computer.city_buffers
        assert 'Mumbai' in computer.city_buffers
        assert 'Bangalore' in computer.city_buffers

    def test_get_buffer_stats(self, computer):
        """Test buffer statistics."""
        base_time = 1000000.0

        event = {
            'city': 'Delhi',
            'timestamp': base_time,
            'aqi': 150.0
        }
        computer.compute_features(event)

        stats = computer.get_buffer_stats('Delhi')
        assert stats['city'] == 'Delhi'
        assert stats['buffer_size'] == 1
        assert stats['oldest_timestamp'] == base_time
        assert stats['newest_timestamp'] == base_time

    def test_clear_buffer_single_city(self, computer):
        """Test clearing buffer for a single city."""
        base_time = 1000000.0

        for city in ['Delhi', 'Mumbai']:
            event = {
                'city': city,
                'timestamp': base_time,
                'aqi': 150.0
            }
            computer.compute_features(event)

        computer.clear_buffer('Delhi')

        assert computer.get_buffer_stats('Delhi')['buffer_size'] == 0
        assert computer.get_buffer_stats('Mumbai')['buffer_size'] == 1

    def test_clear_buffer_all(self, computer):
        """Test clearing all buffers."""
        base_time = 1000000.0

        for city in ['Delhi', 'Mumbai', 'Bangalore']:
            event = {
                'city': city,
                'timestamp': base_time,
                'aqi': 150.0
            }
            computer.compute_features(event)

        computer.clear_buffer()

        assert len(computer.city_buffers) == 0

    def test_get_all_buffer_stats(self, computer):
        """Test getting statistics for all buffers."""
        base_time = 1000000.0

        for city in ['Delhi', 'Mumbai']:
            event = {
                'city': city,
                'timestamp': base_time,
                'aqi': 150.0
            }
            computer.compute_features(event)

        all_stats = computer.get_all_buffer_stats()
        assert len(all_stats) == 2
        assert all_stats[0]['city'] in ['Delhi', 'Mumbai']

    def test_features_with_pollutants(self, computer):
        """Test feature computation with pollutant data."""
        event = {
            'city': 'Delhi',
            'timestamp': 1000000.0,
            'aqi': 150.0,
            'pollutants': {
                'pm25': 75.0,
                'pm10': 150.0,
                'no2': 50.0
            }
        }

        features = computer.compute_features(event)

        assert features['pm25'] == 75.0
        assert features['pm10'] == 150.0
        assert features['no2'] == 50.0
