

from typing import Dict, Optional
from datetime import datetime
import uuid

from src.utils.logger import get_logger
from src.utils.constants import AQI_THRESHOLDS, ALERT_LEVELS

logger = get_logger(__name__)


class RuleBasedAlertSystem:
    

    # AQI thresholds from constants
    ALERT_THRESHOLDS = AQI_THRESHOLDS

    # Alert levels from constants
    ALERT_LEVELS = ALERT_LEVELS

    # Health recommendations for each category
    HEALTH_RECOMMENDATIONS = {
        'Good': 'Air quality is satisfactory. Enjoy outdoor activities.',
        'Satisfactory': 'Air quality is acceptable. Sensitive groups should limit prolonged outdoor activities.',
        'Moderately Polluted': 'Members of sensitive groups should reduce prolonged outdoor activities.',
        'Heavily Polluted': 'Everyone should reduce prolonged outdoor activities. Sensitive groups should avoid outdoor activities.',
        'Severely Polluted': 'Avoid outdoor activities. Everyone should wear N95 masks if going outside.'
    }

    def __init__(self):
        """Initialize rule-based alert system."""
        logger.info("Rule-based alert system initialized")

    def _get_aqi_category(self, aqi: float) -> Optional[str]:
        """
        Get AQI category from value.

        Args:
            aqi: AQI value

        Returns:
            Category name or None if out of range
        """
        for category, (min_aqi, max_aqi) in self.ALERT_THRESHOLDS.items():
            if min_aqi <= aqi <= max_aqi:
                return category
        return None

    def evaluate_current_aqi(
        self,
        city: str,
        aqi: float,
        timestamp: Optional[float] = None
    ) -> Optional[Dict]:
        
        if not isinstance(aqi, (int, float)):
            raise ValueError(f"AQI must be numeric, got {type(aqi)}")

        if aqi < 0 or aqi > 500:
            logger.warning(f"AQI out of valid range [0, 500]: {aqi}")

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        category = self._get_aqi_category(aqi)

        if category is None:
            logger.warning(f"Could not determine AQI category for {aqi}")
            return None

        # Only generate alerts for non-Good categories
        if category == 'Good':
            return None

        alert_level = self.ALERT_LEVELS.get(category, 'info')
        recommendation = self.HEALTH_RECOMMENDATIONS.get(
            category,
            'Check air quality status'
        )

        alert = {
            'alert_id': str(uuid.uuid4()),
            'alert_type': 'rule_based',
            'city': city,
            'category': category,
            'level': alert_level,
            'current_aqi': aqi,
            'timestamp': timestamp,
            'message': f"{city}: AQI {aqi:.1f} ({category})",
            'recommendation': recommendation,
            'threshold_min': self.ALERT_THRESHOLDS[category][0],
            'threshold_max': self.ALERT_THRESHOLDS[category][1]
        }

        logger.info(
            f"Rule-based alert generated for {city}: "
            f"AQI={aqi:.1f}, Category={category}"
        )

        return alert

    def evaluate_batch(
        self,
        city_aqi_pairs: list,
        timestamp: Optional[float] = None
    ) -> list:
       
        alerts = []

        for city, aqi in city_aqi_pairs:
            try:
                alert = self.evaluate_current_aqi(city, aqi, timestamp)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating AQI for {city}: {e}")

        return alerts

    def get_alert_threshold(self, category: str) -> Optional[tuple]:
        
        return self.ALERT_THRESHOLDS.get(category)

    def get_alert_level(self, category: str) -> Optional[str]:
        
        return self.ALERT_LEVELS.get(category)

    def get_recommendation(self, category: str) -> Optional[str]:
        
        return self.HEALTH_RECOMMENDATIONS.get(category)

    def get_all_categories(self) -> list:
        
        return list(self.ALERT_THRESHOLDS.keys())

    def get_category_info(self, category: str) -> Optional[Dict]:
        
        if category not in self.ALERT_THRESHOLDS:
            return None

        threshold = self.ALERT_THRESHOLDS[category]
        level = self.ALERT_LEVELS.get(category)
        recommendation = self.HEALTH_RECOMMENDATIONS.get(category)

        return {
            'category': category,
            'threshold_min': threshold[0],
            'threshold_max': threshold[1],
            'alert_level': level,
            'recommendation': recommendation
        }
