import time
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.constants import ALERT_DEDUP_WINDOW_HOURS

logger = get_logger(__name__)


class AlertDeduplicator:
    

    def __init__(self, dedup_window_hours: int = ALERT_DEDUP_WINDOW_HOURS):
        
        self.dedup_window_hours = dedup_window_hours
        self.dedup_window_seconds = dedup_window_hours * 3600
        self.alert_history: Dict[Tuple[str, str], float] = {}
        logger.info(
            f"Alert deduplicator initialized "
            f"(window={dedup_window_hours}h)"
        )

    def should_send_alert(
        self,
        alert: Dict,
        current_time: Optional[float] = None
    ) -> bool:
       
        if 'city' not in alert:
            raise KeyError("Alert missing 'city' field")
        if 'level' not in alert:
            raise KeyError("Alert missing 'level' field")

        if current_time is None:
            current_time = time.time()

        city = alert['city']
        level = alert['level']
        key = (city, level)

        # Check if we have a recent alert for this city/level
        last_alert_time = self.alert_history.get(key)

        if last_alert_time is None:
            # No previous alert, send this one
            self.alert_history[key] = current_time
            logger.debug(
                f"Alert sent for {city} (level={level}), "
                f"no previous alert"
            )
            return True

        # Check if enough time has passed since last alert
        time_since_last = current_time - last_alert_time

        if time_since_last >= self.dedup_window_seconds:
            # Enough time has passed, send this alert
            self.alert_history[key] = current_time
            logger.debug(
                f"Alert sent for {city} (level={level}), "
                f"dedup window expired ({time_since_last:.0f}s)"
            )
            return True
        else:
            # Still within dedup window, skip this alert
            logger.debug(
                f"Alert deduplicated for {city} (level={level}), "
                f"within dedup window ({time_since_last:.0f}s < "
                f"{self.dedup_window_seconds}s)"
            )
            return False

    def filter_alerts(
        self,
        alerts: list,
        current_time: Optional[float] = None
    ) -> list:
       
        if not isinstance(alerts, list):
            logger.error(f"Alerts must be a list, got {type(alerts)}")
            return []

        filtered_alerts = []

        for alert in alerts:
            try:
                if self.should_send_alert(alert, current_time):
                    filtered_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error filtering alert: {e}")

        logger.info(
            f"Filtered {len(alerts)} alerts, "
            f"{len(filtered_alerts)} passed deduplication"
        )

        return filtered_alerts

    def get_alert_history(self) -> Dict[Tuple[str, str], float]:
        
        return self.alert_history.copy()

    def get_last_alert_time(
        self,
        city: str,
        level: str
    ) -> Optional[float]:
       
        key = (city, level)
        return self.alert_history.get(key)

    def get_time_until_next_alert(
        self,
        city: str,
        level: str,
        current_time: Optional[float] = None
    ) -> float:
        
        if current_time is None:
            current_time = time.time()

        last_alert_time = self.get_last_alert_time(city, level)

        if last_alert_time is None:
            return 0.0

        time_since_last = current_time - last_alert_time
        time_until_next = max(0, self.dedup_window_seconds - time_since_last)

        return time_until_next

    def reset_history(self, city: Optional[str] = None) -> None:
        
        if city is None:
            self.alert_history.clear()
            logger.info("Alert history cleared")
        else:
            # Remove all entries for this city
            keys_to_remove = [
                key for key in self.alert_history.keys()
                if key[0] == city
            ]
            for key in keys_to_remove:
                del self.alert_history[key]
            logger.info(f"Alert history cleared for {city}")

    def cleanup_expired_entries(
        self,
        current_time: Optional[float] = None
    ) -> int:
        
        if current_time is None:
            current_time = time.time()

        cutoff_time = current_time - self.dedup_window_seconds
        keys_to_remove = [
            key for key, timestamp in self.alert_history.items()
            if timestamp < cutoff_time
        ]

        for key in keys_to_remove:
            del self.alert_history[key]

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} expired alert entries")

        return len(keys_to_remove)

    def set_dedup_window(self, hours: int) -> None:
        
        if not isinstance(hours, int) or hours <= 0:
            raise ValueError(f"Window must be positive integer, got {hours}")

        self.dedup_window_hours = hours
        self.dedup_window_seconds = hours * 3600
        logger.info(f"Dedup window updated to {hours} hours")

    def get_dedup_window(self) -> int:
        
        return self.dedup_window_hours

    def get_stats(self) -> Dict:
        
        return {
            'dedup_window_hours': self.dedup_window_hours,
            'total_entries': len(self.alert_history),
            'unique_cities': len(set(key[0] for key in self.alert_history.keys())),
            'unique_levels': len(set(key[1] for key in self.alert_history.keys()))
        }
