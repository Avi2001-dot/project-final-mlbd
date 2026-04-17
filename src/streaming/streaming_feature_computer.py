import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from collections import deque

from src.utils.logger import get_logger
from src.utils.constants import (
    STREAMING_WINDOW_HOURS,
    LAG_OFFSETS,
    ROLLING_WINDOWS
)

logger = get_logger(__name__)


class StreamingFeatureComputer:
    
    def __init__(self, window_size_hours: int = STREAMING_WINDOW_HOURS):
        """
        Initialize streaming feature computer.

        Args:
            window_size_hours: Size of sliding window in hours
        """
        self.window_size_hours = window_size_hours
        self.window_size_seconds = window_size_hours * 3600
        self.city_buffers: Dict[str, deque] = {}  # Sliding window per city
        logger.info(
            f"Streaming feature computer initialized "
            f"(window={window_size_hours}h)"
        )

    def _get_or_create_buffer(self, city: str) -> deque:
        
        if city not in self.city_buffers:
            self.city_buffers[city] = deque()
        return self.city_buffers[city]

    def _add_to_buffer(
        self,
        city: str,
        timestamp: float,
        aqi: float
    ) -> None:
        
        buffer = self._get_or_create_buffer(city)

        # Remove old events outside window
        cutoff_time = timestamp - self.window_size_seconds
        while buffer and buffer[0]['timestamp'] <= cutoff_time:
            buffer.popleft()

        # Add new event
        buffer.append({
            'timestamp': timestamp,
            'aqi': aqi
        })

    def _get_lag_features(
        self,
        city: str,
        timestamp: float,
        aqi: float
    ) -> Dict[str, Optional[float]]:
        
        buffer = self._get_or_create_buffer(city)
        lag_features = {}

        # Convert buffer to list for easier indexing
        events = list(buffer)

        for lag_hours in LAG_OFFSETS:
            lag_seconds = lag_hours * 3600
            target_time = timestamp - lag_seconds

            # Find closest event before target time
            lag_value = None
            for event in reversed(events):
                if event['timestamp'] < timestamp and \
                   event['timestamp'] <= target_time:
                    lag_value = event['aqi']
                    break

            lag_features[f'aqi_lag_{lag_hours}h'] = lag_value

        return lag_features

    def _get_rolling_statistics(
        self,
        city: str,
        timestamp: float
    ) -> Dict[str, Optional[float]]:
        
        buffer = self._get_or_create_buffer(city)
        rolling_stats = {}

        # Convert buffer to list
        events = list(buffer)

        for window_hours in ROLLING_WINDOWS:
            window_seconds = window_hours * 3600
            cutoff_time = timestamp - window_seconds

            # Get AQI values within window
            window_values = [
                event['aqi']
                for event in events
                if event['timestamp'] < timestamp and \
                   event['timestamp'] > cutoff_time
            ]

            if window_values:
                rolling_stats[f'aqi_mean_{window_hours}h'] = \
                    float(np.mean(window_values))
                rolling_stats[f'aqi_std_{window_hours}h'] = \
                    float(np.std(window_values))
                rolling_stats[f'aqi_min_{window_hours}h'] = \
                    float(np.min(window_values))
                rolling_stats[f'aqi_max_{window_hours}h'] = \
                    float(np.max(window_values))
            else:
                rolling_stats[f'aqi_mean_{window_hours}h'] = None
                rolling_stats[f'aqi_std_{window_hours}h'] = None
                rolling_stats[f'aqi_min_{window_hours}h'] = None
                rolling_stats[f'aqi_max_{window_hours}h'] = None

        return rolling_stats

    def _get_temporal_features(
        self,
        timestamp: float
    ) -> Dict[str, int]:
        
        dt = datetime.fromtimestamp(timestamp)

        return {
            'hour_of_day': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }

    def _get_seasonal_feature(self, timestamp: float) -> str:
        
        dt = datetime.fromtimestamp(timestamp)
        month = dt.month

        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    def compute_features(
        self,
        event: Dict
    ) -> Dict:
        
        # Validate event
        required_fields = ['city', 'timestamp', 'aqi']
        missing_fields = [f for f in required_fields if f not in event]
        if missing_fields:
            logger.error(f"Event missing required fields: {missing_fields}")
            raise KeyError(f"Missing fields: {missing_fields}")

        city = event['city']
        timestamp = event['timestamp']
        aqi = event['aqi']

        # Validate types
        if not isinstance(timestamp, (int, float)):
            raise ValueError(f"Timestamp must be numeric, got {type(timestamp)}")
        if not isinstance(aqi, (int, float)):
            raise ValueError(f"AQI must be numeric, got {type(aqi)}")

        # Add event to buffer (before computing features)
        self._add_to_buffer(city, timestamp, aqi)

        # Compute features
        features = {
            'city': city,
            'timestamp': timestamp,
            'aqi': aqi
        }

        # Add lag features
        lag_features = self._get_lag_features(city, timestamp, aqi)
        features.update(lag_features)

        # Add rolling statistics
        rolling_stats = self._get_rolling_statistics(city, timestamp)
        features.update(rolling_stats)

        # Add temporal features
        temporal_features = self._get_temporal_features(timestamp)
        features.update(temporal_features)

        # Add seasonal feature
        features['season'] = self._get_seasonal_feature(timestamp)

        # Add pollutants if present
        if 'pollutants' in event:
            features.update(event['pollutants'])

        return features

    def get_buffer_stats(self, city: str) -> Dict:
        
        buffer = self._get_or_create_buffer(city)

        if not buffer:
            return {
                'city': city,
                'buffer_size': 0,
                'oldest_timestamp': None,
                'newest_timestamp': None,
                'time_span_hours': 0
            }

        events = list(buffer)
        oldest_ts = events[0]['timestamp']
        newest_ts = events[-1]['timestamp']
        time_span_hours = (newest_ts - oldest_ts) / 3600

        return {
            'city': city,
            'buffer_size': len(buffer),
            'oldest_timestamp': oldest_ts,
            'newest_timestamp': newest_ts,
            'time_span_hours': time_span_hours
        }

    def clear_buffer(self, city: Optional[str] = None) -> None:
        
        if city:
            if city in self.city_buffers:
                self.city_buffers[city].clear()
                logger.info(f"Cleared buffer for {city}")
        else:
            self.city_buffers.clear()
            logger.info("Cleared all buffers")

    def get_all_buffer_stats(self) -> List[Dict]:
        
        return [
            self.get_buffer_stats(city)
            for city in self.city_buffers.keys()
        ]
