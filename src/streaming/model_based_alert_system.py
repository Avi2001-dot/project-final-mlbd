
from typing import Dict, Optional
from datetime import datetime
import uuid

from src.utils.logger import get_logger
from src.utils.constants import PREDICTION_ALERT_THRESHOLD, ALERT_LEVELS

logger = get_logger(__name__)


class ModelBasedAlertSystem:
   

    # Prediction threshold for alerts (Moderately Polluted)
    PREDICTION_THRESHOLD = PREDICTION_ALERT_THRESHOLD

    def __init__(self, threshold: Optional[float] = None):
        
        if threshold is not None:
            self.PREDICTION_THRESHOLD = threshold
            logger.info(f"Model-based alert threshold set to {threshold}")
        else:
            logger.info(
                f"Model-based alert system initialized "
                f"(threshold={self.PREDICTION_THRESHOLD})"
            )

    def evaluate_prediction(
        self,
        city: str,
        predicted_aqi: float,
        current_aqi: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Optional[Dict]:
        
        if not isinstance(predicted_aqi, (int, float)):
            raise ValueError(
                f"Predicted AQI must be numeric, got {type(predicted_aqi)}"
            )

        if predicted_aqi < 0:
            logger.warning(f"Predicted AQI is negative: {predicted_aqi}")

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Only generate alert if prediction exceeds threshold
        if predicted_aqi <= self.PREDICTION_THRESHOLD:
            return None

        # Determine alert level based on predicted AQI
        alert_level = self._get_alert_level_for_aqi(predicted_aqi)

        alert = {
            'alert_id': str(uuid.uuid4()),
            'alert_type': 'model_based',
            'city': city,
            'level': alert_level,
            'current_aqi': current_aqi,
            'predicted_aqi': predicted_aqi,
            'timestamp': timestamp,
            'message': (
                f"{city}: Predicted AQI {predicted_aqi:.1f} "
                f"exceeds threshold ({self.PREDICTION_THRESHOLD})"
            ),
            'threshold': self.PREDICTION_THRESHOLD,
            'recommendation': self._get_recommendation_for_aqi(predicted_aqi)
        }

        logger.info(
            f"Model-based alert generated for {city}: "
            f"Predicted AQI={predicted_aqi:.1f}, "
            f"Current AQI={current_aqi}"
        )

        return alert

    def _get_alert_level_for_aqi(self, aqi: float) -> str:
        
        if aqi <= 100:
            return 'info'
        elif aqi <= 200:
            return 'warning'
        elif aqi <= 300:
            return 'severe'
        else:
            return 'critical'

    def _get_recommendation_for_aqi(self, aqi: float) -> str:
        
        if aqi <= 100:
            return 'Monitor air quality. Sensitive groups should limit outdoor activities.'
        elif aqi <= 200:
            return 'Sensitive groups should reduce prolonged outdoor activities.'
        elif aqi <= 300:
            return 'Everyone should reduce prolonged outdoor activities. Sensitive groups should avoid outdoor activities.'
        else:
            return 'Avoid outdoor activities. Everyone should wear N95 masks if going outside.'

    def evaluate_batch(
        self,
        predictions: list,
        timestamp: Optional[float] = None
    ) -> list:
        
        alerts = []

        for pred in predictions:
            try:
                alert = self.evaluate_prediction(
                    city=pred['city'],
                    predicted_aqi=pred['predicted_aqi'],
                    current_aqi=pred.get('current_aqi'),
                    timestamp=timestamp
                )
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(
                    f"Error evaluating prediction for {pred.get('city')}: {e}"
                )

        return alerts

    def set_threshold(self, threshold: float) -> None:
        
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"Threshold must be numeric, got {type(threshold)}"
            )

        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

        self.PREDICTION_THRESHOLD = threshold
        logger.info(f"Prediction threshold updated to {threshold}")

    def get_threshold(self) -> float:
        
        return self.PREDICTION_THRESHOLD

    def get_alert_info(self, predicted_aqi: float) -> Dict:
        
        will_trigger = predicted_aqi > self.PREDICTION_THRESHOLD
        alert_level = self._get_alert_level_for_aqi(predicted_aqi)
        recommendation = self._get_recommendation_for_aqi(predicted_aqi)

        return {
            'predicted_aqi': predicted_aqi,
            'threshold': self.PREDICTION_THRESHOLD,
            'will_trigger_alert': will_trigger,
            'alert_level': alert_level,
            'recommendation': recommendation
        }
