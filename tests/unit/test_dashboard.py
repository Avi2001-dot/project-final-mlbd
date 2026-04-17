import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dashboard.data_store import DataStore
from src.utils.constants import (
    CITIES,
    AQI_THRESHOLDS,
    AQI_COLORS,
    DASHBOARD_REFRESH_INTERVAL_MINUTES
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_alert_store():
    """Create mock alert store."""
    mock_store = Mock()
    mock_store.get_active_alerts.return_value = [
        {
            'alert_id': 'alert_1',
            'city': 'Delhi',
            'level': 'warning',
            'current_aqi': 150,
            'predicted_aqi': 160,
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'rule_based'
        }
    ]
    mock_store.get_alerts_by_city.return_value = []
    mock_store.get_alerts_by_level.return_value = []
    return mock_store


@pytest.fixture
def mock_model_registry():
    """Create mock model registry."""
    mock_registry = Mock()
    mock_registry.get_best_model.return_value = 'model_v1'
    mock_registry.get_model.return_value = Mock()
    return mock_registry


@pytest.fixture
def data_store(mock_alert_store, mock_model_registry):
    """Create data store with mocked dependencies."""
    with patch('src.dashboard.data_store.AlertStore', return_value=mock_alert_store):
        with patch('src.dashboard.data_store.ModelRegistry', return_value=mock_model_registry):
            store = DataStore(
                alert_db_path='test_alerts.db',
                model_registry_path='test_models',
                gold_layer_path='test_gold'
            )
            yield store
            store.close()


# ============================================================================
# TESTS: DATA STORE INITIALIZATION
# ============================================================================


def test_data_store_initialization(data_store):
    """Test DataStore initialization."""
    assert data_store is not None
    assert data_store.alert_store is not None
    assert data_store.model_registry is not None
    assert data_store.gold_layer_path == 'test_gold'
    assert len(data_store.cache) == 0


def test_data_store_cache_initialization(data_store):
    """Test cache initialization."""
    assert isinstance(data_store.cache, dict)
    assert isinstance(data_store.cache_timestamps, dict)
    assert len(data_store.cache) == 0
    assert len(data_store.cache_timestamps) == 0


# ============================================================================
# TESTS: CITY METHODS
# ============================================================================


def test_get_cities(data_store):
    """Test getting list of available cities."""
    cities = data_store.get_cities()
    assert cities is not None
    assert len(cities) > 0
    assert 'Delhi' in cities
    assert 'Mumbai' in cities
    assert all(isinstance(city, str) for city in cities)


def test_get_cities_returns_constants(data_store):
    """Test that get_cities returns CITIES constant."""
    cities = data_store.get_cities()
    assert cities == CITIES


# ============================================================================
# TESTS: CACHE MANAGEMENT
# ============================================================================


def test_cache_validity_check_empty(data_store):
    """Test cache validity check for non-existent key."""
    assert data_store._is_cache_valid('nonexistent_key') is False


def test_cache_validity_check_valid(data_store):
    """Test cache validity check for valid cache entry."""
    cache_key = 'test_key'
    data_store.cache[cache_key] = {'test': 'data'}
    data_store.cache_timestamps[cache_key] = datetime.now()
    
    assert data_store._is_cache_valid(cache_key) is True


def test_cache_validity_check_expired(data_store):
    """Test cache validity check for expired cache entry."""
    cache_key = 'test_key'
    data_store.cache[cache_key] = {'test': 'data'}
    # Set timestamp to past (older than TTL)
    data_store.cache_timestamps[cache_key] = datetime.now() - timedelta(
        seconds=DASHBOARD_REFRESH_INTERVAL_MINUTES * 60 + 1
    )
    
    assert data_store._is_cache_valid(cache_key) is False


def test_clear_cache(data_store):
    """Test cache clearing."""
    # Add some cache entries
    data_store.cache['key1'] = 'value1'
    data_store.cache['key2'] = 'value2'
    data_store.cache_timestamps['key1'] = datetime.now()
    
    assert len(data_store.cache) > 0
    
    # Clear cache
    data_store.clear_cache()
    
    assert len(data_store.cache) == 0
    assert len(data_store.cache_timestamps) == 0


# ============================================================================
# TESTS: ALERT METHODS
# ============================================================================


def test_get_active_alerts(data_store):
    """Test getting active alerts."""
    alerts = data_store.get_active_alerts()
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert 'city' in alerts[0]
    assert 'level' in alerts[0]


def test_get_active_alerts_empty(data_store):
    """Test getting active alerts when none exist."""
    data_store.alert_store.get_active_alerts.return_value = []
    alerts = data_store.get_active_alerts()
    assert isinstance(alerts, list)
    assert len(alerts) == 0


def test_get_alerts_by_city(data_store):
    """Test getting alerts by city."""
    alerts = data_store.get_alerts_by_city('Delhi')
    assert isinstance(alerts, list)
    data_store.alert_store.get_alerts_by_city.assert_called_once_with('Delhi')


def test_get_alerts_by_level(data_store):
    """Test getting alerts by level."""
    alerts = data_store.get_alerts_by_level('warning')
    assert isinstance(alerts, list)
    data_store.alert_store.get_alerts_by_level.assert_called_once_with('warning')


def test_get_alerts_error_handling(data_store):
    """Test error handling in alert retrieval."""
    data_store.alert_store.get_active_alerts.side_effect = Exception('DB Error')
    alerts = data_store.get_active_alerts()
    assert isinstance(alerts, list)
    assert len(alerts) == 0


# ============================================================================
# TESTS: LATEST AQI METHODS
# ============================================================================


def test_get_latest_aqi_returns_dict(data_store):
    """Test that get_latest_aqi returns a dictionary."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        mock_read.return_value = {
            'aqi': 75.5,
            'timestamp': datetime.now().isoformat()
        }
        
        result = data_store.get_latest_aqi('Delhi')
        
        assert isinstance(result, dict)
        assert 'aqi' in result
        assert 'timestamp' in result


def test_get_latest_aqi_caching(data_store):
    """Test that get_latest_aqi caches results."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        mock_read.return_value = {
            'aqi': 75.5,
            'timestamp': datetime.now().isoformat()
        }
        
        # First call
        result1 = data_store.get_latest_aqi('Delhi')
        call_count_1 = mock_read.call_count
        
        # Second call (should use cache)
        result2 = data_store.get_latest_aqi('Delhi')
        call_count_2 = mock_read.call_count
        
        assert result1 == result2
        assert call_count_1 == call_count_2  # No additional call


def test_get_latest_aqi_no_data(data_store):
    """Test get_latest_aqi when no data available."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        mock_read.return_value = None
        
        result = data_store.get_latest_aqi('Delhi')
        
        assert result is None


def test_get_latest_aqi_error_handling(data_store):
    """Test error handling in get_latest_aqi."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        mock_read.side_effect = Exception('Read Error')
        
        result = data_store.get_latest_aqi('Delhi')
        
        assert result is None


# ============================================================================
# TESTS: FORECAST METHODS
# ============================================================================


def test_get_forecast_returns_dataframe(data_store):
    """Test that get_forecast returns a DataFrame."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        with patch.object(data_store, '_generate_forecast') as mock_forecast:
            mock_read.return_value = {'aqi': 75.5, 'timestamp': datetime.now().isoformat()}
            mock_forecast.return_value = pd.DataFrame({
                'timestamp': [datetime.now() + timedelta(hours=i) for i in range(1, 25)],
                'predicted_aqi': [75.5] * 24,
                'upper_bound': [82.0] * 24,
                'lower_bound': [69.0] * 24
            })
            
            result = data_store.get_forecast('Delhi')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 24
            assert 'predicted_aqi' in result.columns


def test_get_forecast_caching(data_store):
    """Test that get_forecast caches results."""
    with patch.object(data_store, '_read_gold_layer_latest') as mock_read:
        with patch.object(data_store, '_generate_forecast') as mock_forecast:
            mock_read.return_value = {'aqi': 75.5, 'timestamp': datetime.now().isoformat()}
            mock_forecast.return_value = pd.DataFrame({
                'timestamp': [datetime.now() + timedelta(hours=i) for i in range(1, 25)],
                'predicted_aqi': [75.5] * 24
            })
            
            # First call
            result1 = data_store.get_forecast('Delhi')
            call_count_1 = mock_forecast.call_count
            
            # Second call (should use cache)
            result2 = data_store.get_forecast('Delhi')
            call_count_2 = mock_forecast.call_count
            
            assert call_count_1 == call_count_2  # No additional call


def test_get_forecast_no_model(data_store):
    """Test get_forecast when no model available."""
    data_store.model_registry.get_best_model.return_value = None
    
    result = data_store.get_forecast('Delhi')
    
    assert result is None


# ============================================================================
# TESTS: HISTORICAL DATA METHODS
# ============================================================================


def test_get_historical_aqi_returns_dataframe(data_store):
    """Test that get_historical_aqi returns a DataFrame."""
    with patch.object(data_store, '_read_gold_layer_historical') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(7)],
            'aqi': [75.5 + i for i in range(7)]
        })
        
        result = data_store.get_historical_aqi('Delhi', days=7)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 7
        assert 'aqi' in result.columns
        assert 'timestamp' in result.columns


def test_get_historical_aqi_caching(data_store):
    """Test that get_historical_aqi caches results."""
    with patch.object(data_store, '_read_gold_layer_historical') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(7)],
            'aqi': [75.5 + i for i in range(7)]
        })
        
        # First call
        result1 = data_store.get_historical_aqi('Delhi', days=7)
        call_count_1 = mock_read.call_count
        
        # Second call (should use cache)
        result2 = data_store.get_historical_aqi('Delhi', days=7)
        call_count_2 = mock_read.call_count
        
        assert call_count_1 == call_count_2  # No additional call


def test_get_historical_aqi_no_data(data_store):
    """Test get_historical_aqi when no data available."""
    with patch.object(data_store, '_read_gold_layer_historical') as mock_read:
        mock_read.return_value = None
        
        result = data_store.get_historical_aqi('Delhi', days=7)
        
        assert result is None


def test_get_historical_aqi_different_days(data_store):
    """Test get_historical_aqi with different day ranges."""
    with patch.object(data_store, '_read_gold_layer_historical') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(30)],
            'aqi': [75.5 + i for i in range(30)]
        })
        
        result = data_store.get_historical_aqi('Delhi', days=30)
        
        assert isinstance(result, pd.DataFrame)
        mock_read.assert_called_once_with('Delhi', 30)


# ============================================================================
# TESTS: UTILITY FUNCTIONS (Skipped if Streamlit not available)
# ============================================================================

# Check if streamlit is available
try:
    from src.dashboard.app import get_aqi_category, get_aqi_color, get_alert_emoji
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_category_good():
    """Test AQI category for Good range."""
    assert get_aqi_category(25) == 'Good'
    assert get_aqi_category(50) == 'Good'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_category_satisfactory():
    """Test AQI category for Satisfactory range."""
    assert get_aqi_category(51) == 'Satisfactory'
    assert get_aqi_category(100) == 'Satisfactory'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_category_moderately_polluted():
    """Test AQI category for Moderately Polluted range."""
    assert get_aqi_category(101) == 'Moderately Polluted'
    assert get_aqi_category(200) == 'Moderately Polluted'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_category_heavily_polluted():
    """Test AQI category for Heavily Polluted range."""
    assert get_aqi_category(201) == 'Heavily Polluted'
    assert get_aqi_category(300) == 'Heavily Polluted'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_category_severely_polluted():
    """Test AQI category for Severely Polluted range."""
    assert get_aqi_category(301) == 'Severely Polluted'
    assert get_aqi_category(500) == 'Severely Polluted'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_color_good():
    """Test AQI color for Good range."""
    assert get_aqi_color(25) == '#00E400'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_color_yellow():
    """Test AQI color for Satisfactory range."""
    assert get_aqi_color(75) == '#FFFF00'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_color_orange():
    """Test AQI color for Moderately Polluted range."""
    assert get_aqi_color(150) == '#FF7E00'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_color_red():
    """Test AQI color for Heavily Polluted range."""
    assert get_aqi_color(250) == '#FF0000'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_aqi_color_dark_red():
    """Test AQI color for Severely Polluted range."""
    assert get_aqi_color(350) == '#8F0000'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_alert_emoji_critical():
    """Test alert emoji for critical level."""
    assert get_alert_emoji('critical') == '🔴'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_alert_emoji_severe():
    """Test alert emoji for severe level."""
    assert get_alert_emoji('severe') == '🔴'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_alert_emoji_warning():
    """Test alert emoji for warning level."""
    assert get_alert_emoji('warning') == '🟡'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_alert_emoji_info():
    """Test alert emoji for info level."""
    assert get_alert_emoji('info') == '🔵'


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason='Streamlit not installed')
def test_alert_emoji_none():
    """Test alert emoji for no alert."""
    assert get_alert_emoji('none') == '🟢'


# ============================================================================
# TESTS: INTEGRATION
# ============================================================================


def test_data_store_full_workflow(data_store):
    """Test complete data store workflow."""
    # Get cities
    cities = data_store.get_cities()
    assert len(cities) > 0
    
    # Get alerts
    alerts = data_store.get_active_alerts()
    assert isinstance(alerts, list)
    
    # Clear cache
    data_store.clear_cache()
    assert len(data_store.cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
