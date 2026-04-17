import pytest
import tempfile
import os
from datetime import datetime
from src.streaming.alert_store import AlertStore
from src.streaming.alert_service import AlertService


class TestAlertStore:
    """Test cases for AlertStore."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def store(self, temp_db):
        """Create alert store with temporary database."""
        return AlertStore(temp_db)

    def test_initialization(self, store):
        """Test store initialization."""
        assert store.db_path is not None
        assert store.conn is not None
        assert store.cursor is not None

    def test_store_alert(self, store):
        """Test storing an alert."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150.0,
            'timestamp': datetime.now().timestamp(),
            'message': 'Test alert'
        }

        alert_id = store.store_alert(alert)

        assert alert_id is not None
        assert isinstance(alert_id, str)

    def test_store_alert_missing_fields(self, store):
        """Test storing alert with missing required fields."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi'
            # Missing 'level' and 'timestamp'
        }

        with pytest.raises(KeyError):
            store.store_alert(alert)

    def test_get_alert(self, store):
        """Test retrieving an alert."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150.0,
            'timestamp': datetime.now().timestamp(),
            'message': 'Test alert'
        }

        alert_id = store.store_alert(alert)
        retrieved = store.get_alert(alert_id)

        assert retrieved is not None
        assert retrieved['city'] == 'Delhi'
        assert retrieved['level'] == 'warning'

    def test_get_nonexistent_alert(self, store):
        """Test retrieving nonexistent alert."""
        retrieved = store.get_alert('nonexistent_id')
        assert retrieved is None

    def test_store_batch(self, store):
        """Test storing batch of alerts."""
        alerts = [
            {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 1'
            },
            {
                'alert_type': 'model_based',
                'city': 'Mumbai',
                'level': 'severe',
                'predicted_aqi': 200.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 2'
            }
        ]

        alert_ids = store.store_batch(alerts)

        assert len(alert_ids) == 2
        assert all(isinstance(aid, str) for aid in alert_ids)

    def test_get_active_alerts(self, store):
        """Test retrieving active alerts."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150.0,
            'timestamp': datetime.now().timestamp(),
            'message': 'Test alert'
        }

        store.store_alert(alert)

        active = store.get_active_alerts()

        assert len(active) == 1
        assert active[0]['city'] == 'Delhi'

    def test_get_active_alerts_by_city(self, store):
        """Test retrieving active alerts by city."""
        alerts = [
            {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 1'
            },
            {
                'alert_type': 'rule_based',
                'city': 'Mumbai',
                'level': 'warning',
                'current_aqi': 160.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 2'
            }
        ]

        for alert in alerts:
            store.store_alert(alert)

        delhi_alerts = store.get_active_alerts(city='Delhi')

        assert len(delhi_alerts) == 1
        assert delhi_alerts[0]['city'] == 'Delhi'

    def test_get_alerts_by_city(self, store):
        """Test retrieving alerts by city."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150.0,
            'timestamp': datetime.now().timestamp(),
            'message': 'Test alert'
        }

        store.store_alert(alert)

        alerts = store.get_alerts_by_city('Delhi', hours=24)

        assert len(alerts) == 1
        assert alerts[0]['city'] == 'Delhi'

    def test_get_alerts_by_level(self, store):
        """Test retrieving alerts by level."""
        alerts = [
            {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 1'
            },
            {
                'alert_type': 'rule_based',
                'city': 'Mumbai',
                'level': 'severe',
                'current_aqi': 250.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 2'
            }
        ]

        for alert in alerts:
            store.store_alert(alert)

        warning_alerts = store.get_alerts_by_level('warning', hours=24)

        assert len(warning_alerts) == 1
        assert warning_alerts[0]['level'] == 'warning'

    def test_acknowledge_alert(self, store):
        """Test acknowledging an alert."""
        alert = {
            'alert_type': 'rule_based',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150.0,
            'timestamp': datetime.now().timestamp(),
            'message': 'Test alert'
        }

        alert_id = store.store_alert(alert)
        result = store.acknowledge_alert(alert_id, 'user@example.com')

        assert result is True

        # Verify alert is acknowledged
        retrieved = store.get_alert(alert_id)
        assert retrieved['acknowledged'] == 1

    def test_acknowledge_batch(self, store):
        """Test acknowledging batch of alerts."""
        alerts = [
            {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 1'
            },
            {
                'alert_type': 'rule_based',
                'city': 'Mumbai',
                'level': 'warning',
                'current_aqi': 160.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 2'
            }
        ]

        alert_ids = store.store_batch(alerts)
        count = store.acknowledge_batch(alert_ids)

        assert count == 2

    def test_get_stats(self, store):
        """Test getting alert statistics."""
        alerts = [
            {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 1'
            },
            {
                'alert_type': 'rule_based',
                'city': 'Mumbai',
                'level': 'severe',
                'current_aqi': 250.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Alert 2'
            }
        ]

        for alert in alerts:
            store.store_alert(alert)

        stats = store.get_stats(hours=24)

        assert stats['total_alerts'] == 2
        assert stats['active_alerts'] == 2
        assert 'warning' in stats['by_level']
        assert 'Delhi' in stats['by_city']

    def test_close(self, store):
        """Test closing store."""
        store.close()
        # Should not raise error

    def test_context_manager(self, temp_db):
        """Test context manager usage."""
        with AlertStore(temp_db) as store:
            alert = {
                'alert_type': 'rule_based',
                'city': 'Delhi',
                'level': 'warning',
                'current_aqi': 150.0,
                'timestamp': datetime.now().timestamp(),
                'message': 'Test alert'
            }
            store.store_alert(alert)


class TestAlertService:
    """Test cases for AlertService."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def service(self, temp_db):
        """Create alert service with temporary database."""
        return AlertService(alert_store_path=temp_db)

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.rule_based_system is not None
        assert service.model_based_system is not None
        assert service.deduplicator is not None
        assert service.alert_store is not None

    def test_process_current_aqi(self, service):
        """Test processing current AQI."""
        alerts = service.process_current_aqi('Delhi', 150.0)

        assert isinstance(alerts, list)

    def test_process_prediction(self, service):
        """Test processing prediction."""
        alerts = service.process_prediction('Delhi', 200.0, 150.0)

        assert isinstance(alerts, list)

    def test_process_inference_result(self, service):
        """Test processing inference result."""
        result = {
            'city': 'Delhi',
            'timestamp': datetime.now().timestamp(),
            'current_aqi': 150.0,
            'predicted_aqi': 200.0
        }

        alerts = service.process_inference_result(result)

        assert isinstance(alerts, list)

    def test_register_notification_handler(self, service):
        """Test registering notification handler."""
        handler_called = []

        def test_handler(alert):
            handler_called.append(alert)

        service.register_notification_handler(test_handler)

        assert len(service.notification_handlers) == 1

    def test_get_active_alerts(self, service):
        """Test getting active alerts."""
        service.process_current_aqi('Delhi', 150.0)

        alerts = service.get_active_alerts()

        assert isinstance(alerts, list)

    def test_get_alerts_by_city(self, service):
        """Test getting alerts by city."""
        service.process_current_aqi('Delhi', 150.0)

        alerts = service.get_alerts_by_city('Delhi')

        assert isinstance(alerts, list)

    def test_get_alerts_by_level(self, service):
        """Test getting alerts by level."""
        service.process_current_aqi('Delhi', 150.0)

        alerts = service.get_alerts_by_level('warning')

        assert isinstance(alerts, list)

    def test_acknowledge_alert(self, service):
        """Test acknowledging alert."""
        alerts = service.process_current_aqi('Delhi', 150.0)

        if alerts:
            alert_id = alerts[0]['alert_id']
            result = service.acknowledge_alert(alert_id)
            assert result is True

    def test_get_stats(self, service):
        """Test getting service statistics."""
        service.process_current_aqi('Delhi', 150.0)

        stats = service.get_stats()

        assert 'total_alerts' in stats
        assert 'active_alerts' in stats
        assert 'deduplicator_stats' in stats

    def test_close(self, service):
        """Test closing service."""
        service.close()
        # Should not raise error

    def test_context_manager(self, temp_db):
        """Test context manager usage."""
        with AlertService(alert_store_path=temp_db) as service:
            service.process_current_aqi('Delhi', 150.0)
