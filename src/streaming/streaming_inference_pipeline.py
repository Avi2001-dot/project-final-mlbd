import time
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.constants import MAX_STREAMING_LATENCY_MS
from src.streaming.streaming_feature_computer import StreamingFeatureComputer

logger = get_logger(__name__)


class StreamingInferencePipeline:
   

    def __init__(
        self,
        model,
        feature_columns: Optional[list] = None,
        max_latency_ms: int = MAX_STREAMING_LATENCY_MS
    ):
        
        if model is None:
            raise ValueError("Model cannot be None")

        self.model = model
        self.feature_columns = feature_columns or []
        self.max_latency_ms = max_latency_ms
        self.feature_computer = StreamingFeatureComputer()

        # Latency tracking
        self.latency_history = []
        self.max_latency_observed = 0
        self.min_latency_observed = float('inf')
        self.events_processed = 0
        self.events_exceeding_latency = 0

        logger.info(
            f"Streaming inference pipeline initialized "
            f"(max_latency={max_latency_ms}ms)"
        )

    def process_event(
        self,
        event: Dict
    ) -> Dict:
       
          
    
        start_time = time.time()

        try:
            # Compute features
            features = self.feature_computer.compute_features(event)

            # Extract feature vector for model
            feature_vector = self._extract_feature_vector(features)

            if feature_vector is None:
                logger.warning(
                    f"Could not extract feature vector for {event['city']}"
                )
                return {
                    'city': event['city'],
                    'timestamp': event['timestamp'],
                    'current_aqi': event['aqi'],
                    'predicted_aqi': None,
                    'latency_ms': (time.time() - start_time) * 1000,
                    'features_computed': False,
                    'error': 'Feature extraction failed'
                }

            # Generate prediction
            prediction = self.model.predict(feature_vector.reshape(1, -1))[0]

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Track latency
            self._track_latency(latency_ms)

            # Log warning if latency exceeded
            if latency_ms > self.max_latency_ms:
                logger.warning(
                    f"Latency exceeded for {event['city']}: "
                    f"{latency_ms:.2f}ms > {self.max_latency_ms}ms"
                )
                self.events_exceeding_latency += 1

            return {
                'city': event['city'],
                'timestamp': event['timestamp'],
                'current_aqi': event['aqi'],
                'predicted_aqi': float(prediction),
                'latency_ms': latency_ms,
                'features_computed': True
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error processing event for {event.get('city', 'unknown')}: {e}"
            )
            return {
                'city': event.get('city', 'unknown'),
                'timestamp': event.get('timestamp'),
                'current_aqi': event.get('aqi'),
                'predicted_aqi': None,
                'latency_ms': latency_ms,
                'features_computed': False,
                'error': str(e)
            }

    def _extract_feature_vector(self, features: dict):
        
        try:
            exclude_keys = {'city', 'timestamp', 'aqi', 'season', 'error'}

            if not self.feature_columns:
                # Auto mode – use every numeric value that isn't metadata
                feature_values = [
                    float(v) for k, v in features.items()
                    if k not in exclude_keys
                    and isinstance(v, (int, float))
                    and v is not None
                ]
            else:
                # Explicit columns – fill missing with 0.0 so shape is always fixed
                feature_values = []
                for col in self.feature_columns:
                    val = features.get(col)
                    if val is None or not isinstance(val, (int, float)):
                        val = 0.0
                    feature_values.append(float(val))

            if not feature_values:
                logger.warning("No numeric feature values extracted")
                return None

            arr = np.array(feature_values, dtype=np.float32)

            # If the model has a known expected number of features, pad/truncate
            expected = getattr(self.model, 'n_features_in_', None)
            if expected is not None and arr.shape[0] != expected:
                if arr.shape[0] < expected:
                    arr = np.pad(arr, (0, expected - arr.shape[0]))
                else:
                    arr = arr[:expected]

            return arr

        except Exception as e:
            logger.error(f"Error extracting feature vector: {e}")
            return None

    def _track_latency(self, latency_ms: float) -> None:
        
        self.latency_history.append(latency_ms)
        self.events_processed += 1
        self.max_latency_observed = max(self.max_latency_observed, latency_ms)
        self.min_latency_observed = min(self.min_latency_observed, latency_ms)

        # Keep only last 1000 latencies to avoid memory bloat
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]

    def get_latency_stats(self) -> Dict:
        
        if not self.latency_history:
            return {
                'events_processed': 0,
                'min_latency_ms': None,
                'max_latency_ms': None,
                'mean_latency_ms': None,
                'median_latency_ms': None,
                'p95_latency_ms': None,
                'p99_latency_ms': None,
                'events_exceeding_max': 0
            }

        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)

        return {
            'events_processed': self.events_processed,
            'min_latency_ms': float(np.min(sorted_latencies)),
            'max_latency_ms': float(np.max(sorted_latencies)),
            'mean_latency_ms': float(np.mean(sorted_latencies)),
            'median_latency_ms': float(np.median(sorted_latencies)),
            'p95_latency_ms': float(np.percentile(sorted_latencies, 95)),
            'p99_latency_ms': float(np.percentile(sorted_latencies, 99)),
            'events_exceeding_max': self.events_exceeding_latency
        }

    def reset_latency_tracking(self) -> None:
        """Reset latency tracking metrics."""
        self.latency_history = []
        self.max_latency_observed = 0
        self.min_latency_observed = float('inf')
        self.events_processed = 0
        self.events_exceeding_latency = 0
        logger.info("Latency tracking reset")

    def get_feature_computer(self) -> StreamingFeatureComputer:
        
        return self.feature_computer
