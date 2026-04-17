import pytest
import time
from src.streaming.rule_based_alert_system import RuleBasedAlertSystem
from src.streaming.model_based_alert_system import ModelBasedAlertSystem
from src.streaming.alert_deduplicator import AlertDeduplicator


class TestRuleBasedAlertSystem:
    """Test cases for RuleBasedAlertSystem."""

    @pytest.fixture
    def system(self):
        """Create rule-based alert system."""
        return RuleBasedAlertSystem()

    def test_initialization(self, system):
        """Test system initialization."""
        assert system.ALERT_THRESHOLDS is not None
        assert len(system.ALERT_THRESHOLDS) == 5

    def test_evaluate_good_aqi(self, system):
        """Test evaluation of Good AQI."""
        alert = system.evaluate_current_aqi('Delhi', 50.0)
        assert alert is None  # No alert for Good category

    def test_evaluate_satisfactory_aqi(self, system):
        """Test evaluation of Satisfactory AQI."""
        alert = system.evaluate_current_aqi('Delhi', 75.0)
        assert alert is not None
        assert alert['category'] == 'Satisfactory'
        assert alert['level'] == 'info'
        assert alert['current_aqi'] == 75.0

    def test_evaluate_moderately_polluted_aqi(self, system):
        """Test evaluation of Moderately Polluted AQI."""
        alert = system.evaluate_current_aqi('Delhi', 150.0)
        assert alert is not None
        assert alert['category'] == 'Moderately Polluted'
        assert alert['level'] == 'warning'

    def test_evaluate_heavily_polluted_aqi(self, system):
        """Test evaluation of Heavily Polluted AQI."""
        alert = system.evaluate_current_aqi('Delhi', 250.0)
        assert alert is not None
        assert alert['category'] == 'Heavily Polluted'
        assert alert['level'] == 'severe'

    def test_evaluate_severely_polluted_aqi(self, system):
        """Test evaluation of Severely Polluted AQI."""
        alert = system.evaluate_current_aqi('Delhi', 350.0)
        assert alert is not None
        assert alert['category'] == 'Severely Polluted'
        assert alert['level'] == 'critical'

    def test_evaluate_invalid_aqi(self, system):
        """Test evaluation with invalid AQI."""
        with pytest.raises(ValueError):
            system.evaluate_current_aqi('Delhi', 'not_a_number')

    def test_evaluate_batch(self, system):
        """Test batch evaluation."""
        pairs = [
            ('Delhi', 50.0),
            ('Mumbai', 150.0),
            ('Bangalore', 250.0)
        ]

        alerts = system.evaluate_batch(pairs)

        assert len(alerts) == 2  # Only 2 alerts (Good doesn't trigger)
        assert alerts[0]['city'] == 'Mumbai'
        assert alerts[1]['city'] == 'Bangalore'

    def test_get_alert_threshold(self, system):
        """Test getting alert threshold."""
        threshold = system.get_alert_threshold('Moderately Polluted')
        assert threshold == (101, 200)

    def test_get_alert_level(self, system):
        """Test getting alert level."""
        level = system.get_alert_level('Heavily Polluted')
        assert level == 'severe'

    def test_get_recommendation(self, system):
        """Test getting health recommendation."""
        rec = system.get_recommendation('Severely Polluted')
        assert 'N95' in rec or 'mask' in rec.lower()

    def test_get_all_categories(self, system):
        """Test getting all categories."""
        categories = system.get_all_categories()
        assert len(categories) == 5
        assert 'Good' in categories
        assert 'Severely Polluted' in categories

    def test_get_category_info(self, system):
        """Test getting category information."""
        info = system.get_category_info('Moderately Polluted')
        assert info['category'] == 'Moderately Polluted'
        assert info['threshold_min'] == 101
        assert info['threshold_max'] == 200
        assert info['alert_level'] == 'warning'


class TestModelBasedAlertSystem:
    """Test cases for ModelBasedAlertSystem."""

    @pytest.fixture
    def system(self):
        """Create model-based alert system."""
        return ModelBasedAlertSystem()

    def test_initialization(self, system):
        """Test system initialization."""
        assert system.PREDICTION_THRESHOLD == 150

    def test_initialization_custom_threshold(self):
        """Test initialization with custom threshold."""
        system = ModelBasedAlertSystem(threshold=200)
        assert system.PREDICTION_THRESHOLD == 200

    def test_evaluate_below_threshold(self, system):
        """Test evaluation below threshold."""
        alert = system.evaluate_prediction('Delhi', 100.0, 90.0)
        assert alert is None

    def test_evaluate_above_threshold(self, system):
        """Test evaluation above threshold."""
        alert = system.evaluate_prediction('Delhi', 200.0, 150.0)
        assert alert is not None
        assert alert['predicted_aqi'] == 200.0
        assert alert['current_aqi'] == 150.0
        assert alert['level'] == 'warning'  # 200 is in warning range (151-200)

    def test_evaluate_at_threshold(self, system):
        """Test evaluation at threshold."""
        alert = system.evaluate_prediction('Delhi', 150.0, 140.0)
        assert alert is None  # At threshold, no alert

    def test_evaluate_invalid_prediction(self, system):
        """Test evaluation with invalid prediction."""
        with pytest.raises(ValueError):
            system.evaluate_prediction('Delhi', 'not_a_number')

    def test_evaluate_batch(self, system):
        """Test batch evaluation."""
        predictions = [
            {'city': 'Delhi', 'predicted_aqi': 100.0, 'current_aqi': 90.0},
            {'city': 'Mumbai', 'predicted_aqi': 200.0, 'current_aqi': 150.0},
            {'city': 'Bangalore', 'predicted_aqi': 180.0, 'current_aqi': 160.0}
        ]

        alerts = system.evaluate_batch(predictions)

        assert len(alerts) == 2  # Only 2 above threshold
        assert alerts[0]['city'] == 'Mumbai'
        assert alerts[1]['city'] == 'Bangalore'

    def test_set_threshold(self, system):
        """Test setting custom threshold."""
        system.set_threshold(200)
        assert system.PREDICTION_THRESHOLD == 200

    def test_set_invalid_threshold(self, system):
        """Test setting invalid threshold."""
        with pytest.raises(ValueError):
            system.set_threshold('not_a_number')

    def test_get_threshold(self, system):
        """Test getting threshold."""
        threshold = system.get_threshold()
        assert threshold == 150

    def test_get_alert_info(self, system):
        """Test getting alert information."""
        info = system.get_alert_info(200.0)
        assert info['predicted_aqi'] == 200.0
        assert info['will_trigger_alert'] is True
        assert info['alert_level'] == 'warning'  # 200 is in warning range


class TestAlertDeduplicator:
    """Test cases for AlertDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        """Create alert deduplicator."""
        return AlertDeduplicator(dedup_window_hours=1)

    def test_initialization(self, deduplicator):
        """Test deduplicator initialization."""
        assert deduplicator.dedup_window_hours == 1
        assert deduplicator.dedup_window_seconds == 3600

    def test_should_send_first_alert(self, deduplicator):
        """Test first alert is sent."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        assert deduplicator.should_send_alert(alert) is True

    def test_should_deduplicate_within_window(self, deduplicator):
        """Test deduplication within window."""
        alert = {'city': 'Delhi', 'level': 'warning'}

        # First alert should be sent
        assert deduplicator.should_send_alert(alert) is True

        # Second alert within window should be deduplicated
        assert deduplicator.should_send_alert(alert) is False

    def test_should_send_after_window_expires(self, deduplicator):
        """Test alert sent after window expires."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        current_time = time.time()

        # First alert
        assert deduplicator.should_send_alert(alert, current_time) is True

        # Second alert after window expires
        future_time = current_time + 3600 + 1
        assert deduplicator.should_send_alert(alert, future_time) is True

    def test_different_levels_not_deduplicated(self, deduplicator):
        """Test different alert levels are not deduplicated."""
        alert1 = {'city': 'Delhi', 'level': 'warning'}
        alert2 = {'city': 'Delhi', 'level': 'severe'}

        assert deduplicator.should_send_alert(alert1) is True
        assert deduplicator.should_send_alert(alert2) is True

    def test_different_cities_not_deduplicated(self, deduplicator):
        """Test different cities are not deduplicated."""
        alert1 = {'city': 'Delhi', 'level': 'warning'}
        alert2 = {'city': 'Mumbai', 'level': 'warning'}

        assert deduplicator.should_send_alert(alert1) is True
        assert deduplicator.should_send_alert(alert2) is True

    def test_filter_alerts(self, deduplicator):
        """Test filtering duplicate alerts."""
        alerts = [
            {'city': 'Delhi', 'level': 'warning'},
            {'city': 'Delhi', 'level': 'warning'},
            {'city': 'Mumbai', 'level': 'warning'}
        ]

        filtered = deduplicator.filter_alerts(alerts)

        assert len(filtered) == 2  # One duplicate removed

    def test_get_alert_history(self, deduplicator):
        """Test getting alert history."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        deduplicator.should_send_alert(alert)

        history = deduplicator.get_alert_history()
        assert ('Delhi', 'warning') in history

    def test_get_last_alert_time(self, deduplicator):
        """Test getting last alert time."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        current_time = time.time()

        deduplicator.should_send_alert(alert, current_time)

        last_time = deduplicator.get_last_alert_time('Delhi', 'warning')
        assert last_time == current_time

    def test_get_time_until_next_alert(self, deduplicator):
        """Test getting time until next alert."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        current_time = time.time()

        deduplicator.should_send_alert(alert, current_time)

        time_until = deduplicator.get_time_until_next_alert(
            'Delhi', 'warning', current_time + 1800
        )
        assert 0 < time_until < 3600

    def test_reset_history_single_city(self, deduplicator):
        """Test resetting history for single city."""
        alert1 = {'city': 'Delhi', 'level': 'warning'}
        alert2 = {'city': 'Mumbai', 'level': 'warning'}

        deduplicator.should_send_alert(alert1)
        deduplicator.should_send_alert(alert2)

        deduplicator.reset_history('Delhi')

        assert deduplicator.get_last_alert_time('Delhi', 'warning') is None
        assert deduplicator.get_last_alert_time('Mumbai', 'warning') is not None

    def test_reset_history_all(self, deduplicator):
        """Test resetting all history."""
        alert1 = {'city': 'Delhi', 'level': 'warning'}
        alert2 = {'city': 'Mumbai', 'level': 'warning'}

        deduplicator.should_send_alert(alert1)
        deduplicator.should_send_alert(alert2)

        deduplicator.reset_history()

        assert len(deduplicator.get_alert_history()) == 0

    def test_cleanup_expired_entries(self, deduplicator):
        """Test cleaning up expired entries."""
        alert = {'city': 'Delhi', 'level': 'warning'}
        current_time = time.time()

        deduplicator.should_send_alert(alert, current_time)

        # Cleanup with future time beyond window
        future_time = current_time + 7200
        removed = deduplicator.cleanup_expired_entries(future_time)

        assert removed == 1

    def test_set_dedup_window(self, deduplicator):
        """Test setting deduplication window."""
        deduplicator.set_dedup_window(2)
        assert deduplicator.dedup_window_hours == 2
        assert deduplicator.dedup_window_seconds == 7200

    def test_get_stats(self, deduplicator):
        """Test getting deduplicator statistics."""
        alert1 = {'city': 'Delhi', 'level': 'warning'}
        alert2 = {'city': 'Mumbai', 'level': 'severe'}

        deduplicator.should_send_alert(alert1)
        deduplicator.should_send_alert(alert2)

        stats = deduplicator.get_stats()

        assert stats['dedup_window_hours'] == 1
        assert stats['total_entries'] == 2
        assert stats['unique_cities'] == 2
        assert stats['unique_levels'] == 2
