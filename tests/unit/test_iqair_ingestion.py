import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests


def create_valid_iqair_response(city: str, aqi: float = 150.5):
    """Create a valid IQAir API response."""
    return {
        'status': 'ok',
        'data': {
            'aqi': aqi,
            'time': {
                'iso': '2024-01-01T10:00:00+05:30'
            },
            'iaqi': {
                'pm25': {'v': 80.0},
                'pm10': {'v': 120.0},
                'no2': {'v': 50.0},
                'o3': {'v': 30.0},
                'so2': {'v': 20.0},
                'co': {'v': 1.5}
            }
        }
    }


def create_iqair_response_missing_pollutants(city: str):
    """Create IQAir response with missing pollutant data."""
    return {
        'status': 'ok',
        'data': {
            'aqi': 150.5,
            'time': {
                'iso': '2024-01-01T10:00:00+05:30'
            },
            'iaqi': {
                'pm25': {'v': 80.0}
            }
        }
    }


class TestIQAirDataIngestionInit:
    """Tests for IQAirDataIngestion initialization."""

    def test_init_with_valid_api_key_and_default_cities(self):
        """Test initialization with valid API key and default cities."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        ingestion = IQAirDataIngestion(api_key='test_key_12345')
        
        assert ingestion.api_key == 'test_key_12345'
        assert len(ingestion.cities) == 10
        assert 'Delhi' in ingestion.cities

    def test_init_with_custom_cities(self):
        """Test initialization with custom city list."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        custom_cities = [
            'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai',
            'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ]
        ingestion = IQAirDataIngestion(
            api_key='test_key_12345',
            cities=custom_cities
        )
        
        assert ingestion.cities == custom_cities

    def test_init_with_empty_api_key(self):
        """Test initialization fails with empty API key."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        with pytest.raises(ValueError, match="API key cannot be empty"):
            IQAirDataIngestion(api_key='')

    def test_init_with_whitespace_api_key(self):
        """Test initialization fails with whitespace-only API key."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        with pytest.raises(ValueError, match="API key cannot be empty"):
            IQAirDataIngestion(api_key='   ')

    def test_init_with_fewer_than_10_cities(self):
        """Test initialization fails with fewer than 10 cities."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        with pytest.raises(ValueError, match="at least 10 cities"):
            IQAirDataIngestion(
                api_key='test_key',
                cities=['Delhi', 'Mumbai', 'Bangalore']
            )

    def test_init_with_exactly_10_cities(self):
        """Test initialization succeeds with exactly 10 cities."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        cities = [
            'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai',
            'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ]
        ingestion = IQAirDataIngestion(api_key='test_key', cities=cities)
        
        assert len(ingestion.cities) == 10


class TestResponseParsing:
    """Tests for IQAir API response parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        self.ingestion = IQAirDataIngestion(api_key='test_key_12345')

    def test_parse_valid_response(self):
        """Test parsing valid IQAir API response."""
        response = create_valid_iqair_response('Delhi', aqi=150.5)
        
        parsed = self.ingestion._parse_iqair_response(response, 'Delhi')
        
        assert parsed['city'] == 'Delhi'
        assert parsed['aqi'] == 150.5
        assert parsed['pm25'] == 80.0
        assert parsed['pm10'] == 120.0
        assert parsed['source'] == 'iqair'

    def test_parse_response_with_missing_pollutants(self):
        """Test parsing response with missing pollutant data."""
        response = create_iqair_response_missing_pollutants('Delhi')
        
        parsed = self.ingestion._parse_iqair_response(response, 'Delhi')
        
        assert parsed['city'] == 'Delhi'
        assert parsed['aqi'] == 150.5
        assert parsed['pm25'] == 80.0
        assert parsed['pm10'] is None
        assert parsed['no2'] is None

    def test_parse_response_missing_aqi(self):
        """Test parsing fails when AQI is missing."""
        response = {
            'status': 'ok',
            'data': {
                'time': {'iso': '2024-01-01T10:00:00+05:30'},
                'iaqi': {}
            }
        }
        
        with pytest.raises(ValueError, match="Missing AQI value"):
            self.ingestion._parse_iqair_response(response, 'Delhi')

    def test_parse_response_invalid_status(self):
        """Test parsing fails with invalid API status."""
        response = {
            'status': 'error',
            'data': {}
        }
        
        with pytest.raises(ValueError, match="error status"):
            self.ingestion._parse_iqair_response(response, 'Delhi')

    def test_parse_response_missing_data_section(self):
        """Test parsing fails when data section is missing."""
        response = {'status': 'ok'}
        
        with pytest.raises(ValueError, match="No data section"):
            self.ingestion._parse_iqair_response(response, 'Delhi')

    def test_parse_response_aqi_out_of_range(self):
        """Test parsing handles out-of-range AQI values."""
        response = create_valid_iqair_response('Delhi', aqi=600.0)
        
        # Should still parse but log warning
        parsed = self.ingestion._parse_iqair_response(response, 'Delhi')
        assert parsed['aqi'] == 600.0

    def test_parse_response_invalid_aqi_type(self):
        """Test parsing fails with invalid AQI type."""
        response = {
            'status': 'ok',
            'data': {
                'aqi': 'invalid',
                'time': {'iso': '2024-01-01T10:00:00+05:30'},
                'iaqi': {}
            }
        }
        
        with pytest.raises(ValueError, match="Invalid AQI value"):
            self.ingestion._parse_iqair_response(response, 'Delhi')

    def test_parse_response_missing_timestamp(self):
        """Test parsing handles missing timestamp gracefully."""
        response = {
            'status': 'ok',
            'data': {
                'aqi': 150.5,
                'iaqi': {}
            }
        }
        
        parsed = self.ingestion._parse_iqair_response(response, 'Delhi')
        
        assert parsed['aqi'] == 150.5
        assert parsed['timestamp'] is not None


class TestFetchCurrentAQI:
    """Tests for fetching current AQI for a single city."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        self.ingestion = IQAirDataIngestion(api_key='test_key_12345')

    @patch('requests.get')
    def test_fetch_current_aqi_success(self, mock_get):
        """Test successful AQI fetch for a city."""
        mock_response = MagicMock()
        mock_response.json.return_value = create_valid_iqair_response('Delhi')
        mock_get.return_value = mock_response
        
        result = self.ingestion.fetch_current_aqi('Delhi')
        
        assert result['city'] == 'Delhi'
        assert result['aqi'] == 150.5
        assert result['source'] == 'iqair'

    @patch('requests.get')
    def test_fetch_current_aqi_with_retry(self, mock_get):
        """Test AQI fetch with retry on failure."""
        mock_response = MagicMock()
        mock_response.json.return_value = create_valid_iqair_response('Delhi')
        
        # Fail first, succeed second
        mock_get.side_effect = [
            requests.RequestException("Connection error"),
            mock_response
        ]
        
        with patch('time.sleep'):
            result = self.ingestion.fetch_current_aqi('Delhi', max_retries=3)
        
        assert result['city'] == 'Delhi'
        assert result['aqi'] == 150.5

    @patch('requests.get')
    def test_fetch_current_aqi_all_retries_fail(self, mock_get):
        """Test AQI fetch fails after all retries."""
        from src.data_ingestion.iqair_ingestion import IQAirAPIError
        
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with patch('time.sleep'):
            with pytest.raises(IQAirAPIError, match="after 3 retries"):
                self.ingestion.fetch_current_aqi('Delhi', max_retries=3)

    @patch('requests.get')
    def test_fetch_current_aqi_timeout(self, mock_get):
        """Test AQI fetch handles timeout."""
        from src.data_ingestion.iqair_ingestion import IQAirAPIError
        
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with patch('time.sleep'):
            with pytest.raises(IQAirAPIError):
                self.ingestion.fetch_current_aqi('Delhi', max_retries=1)

    @patch('requests.get')
    def test_fetch_current_aqi_http_error(self, mock_get):
        """Test AQI fetch handles HTTP errors."""
        from src.data_ingestion.iqair_ingestion import IQAirAPIError
        
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        with patch('time.sleep'):
            with pytest.raises(IQAirAPIError):
                self.ingestion.fetch_current_aqi('Delhi', max_retries=1)


class TestFetchAllCitiesAQI:
    """Tests for fetching AQI for all configured cities."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        self.ingestion = IQAirDataIngestion(api_key='test_key_12345')

    @patch('requests.get')
    def test_fetch_all_cities_success(self, mock_get):
        """Test successful fetch for all cities."""
        mock_response = MagicMock()
        
        def side_effect(*args, **kwargs):
            city = args[0].split('/')[-1].split('?')[0]
            mock_response.json.return_value = create_valid_iqair_response(city)
            return mock_response
        
        mock_get.side_effect = side_effect
        
        df = self.ingestion.fetch_all_cities_aqi()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'city' in df.columns
        assert 'aqi' in df.columns
        assert 'source' in df.columns

    @patch('requests.get')
    def test_fetch_all_cities_partial_failure(self, mock_get):
        """Test fetch handles partial failures gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = create_valid_iqair_response('Delhi')
        
        # Alternate between success and failure
        mock_get.side_effect = [
            mock_response,
            requests.RequestException("Error"),
            mock_response,
            requests.RequestException("Error"),
            mock_response,
            requests.RequestException("Error"),
            mock_response,
            requests.RequestException("Error"),
            mock_response,
            requests.RequestException("Error"),
        ]
        
        with patch('time.sleep'):
            df = self.ingestion.fetch_all_cities_aqi()
        
        # Should have at least some successful records
        assert len(df) > 0

    @patch('requests.get')
    def test_fetch_all_cities_all_fail(self, mock_get):
        """Test fetch fails when all cities fail."""
        from src.data_ingestion.iqair_ingestion import IQAirAPIError
        
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with patch('time.sleep'):
            with pytest.raises(IQAirAPIError, match="all.*cities"):
                self.ingestion.fetch_all_cities_aqi(max_retries=1)

    @patch('requests.get')
    def test_fetch_all_cities_returns_dataframe(self, mock_get):
        """Test fetch returns properly formatted DataFrame."""
        mock_response = MagicMock()
        
        def side_effect(*args, **kwargs):
            city = args[0].split('/')[-1].split('?')[0]
            mock_response.json.return_value = create_valid_iqair_response(city)
            return mock_response
        
        mock_get.side_effect = side_effect
        
        df = self.ingestion.fetch_all_cities_aqi()
        
        # Check DataFrame structure
        expected_columns = [
            'city', 'timestamp', 'aqi', 'pm25', 'pm10',
            'no2', 'o3', 'so2', 'co', 'source'
        ]
        for col in expected_columns:
            assert col in df.columns


class TestRetryLogic:
    """Tests for exponential backoff retry logic."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        self.ingestion = IQAirDataIngestion(api_key='test_key_12345')

    def test_retry_with_backoff_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        mock_func = Mock(return_value={'status': 'ok'})
        
        result = self.ingestion._retry_with_backoff(
            func=mock_func,
            max_retries=3,
            initial_delay=1
        )
        
        assert result == {'status': 'ok'}
        assert mock_func.call_count == 1

    def test_retry_with_backoff_success_after_failures(self):
        """Test retry succeeds after initial failures."""
        mock_func = Mock(
            side_effect=[
                Exception("Error 1"),
                Exception("Error 2"),
                {'status': 'ok'}
            ]
        )
        
        with patch('time.sleep'):
            result = self.ingestion._retry_with_backoff(
                func=mock_func,
                max_retries=3,
                initial_delay=1
            )
        
        assert result == {'status': 'ok'}
        assert mock_func.call_count == 3

    def test_retry_with_backoff_all_failures(self):
        """Test retry returns None after all attempts fail."""
        mock_func = Mock(side_effect=Exception("Error"))
        
        with patch('time.sleep'):
            result = self.ingestion._retry_with_backoff(
                func=mock_func,
                max_retries=3,
                initial_delay=1
            )
        
        assert result is None
        assert mock_func.call_count == 3

    def test_retry_exponential_backoff_delays(self):
        """Test exponential backoff calculates correct delays."""
        mock_func = Mock(side_effect=Exception("Error"))
        sleep_times = []
        
        def mock_sleep(duration):
            sleep_times.append(duration)
        
        with patch('time.sleep', side_effect=mock_sleep):
            self.ingestion._retry_with_backoff(
                func=mock_func,
                max_retries=4,
                initial_delay=30
            )
        
        # Should have delays: 30, 60, 120
        assert sleep_times == [30, 60, 120]


class TestErrorHandling:
    """Tests for error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        self.ingestion = IQAirDataIngestion(api_key='test_key_12345')

    def test_iqair_api_error_message(self):
        """Test IQAirAPIError has proper message."""
        from src.data_ingestion.iqair_ingestion import IQAirAPIError
        
        error = IQAirAPIError("Test error message")
        assert str(error) == "Test error message"

    @patch('requests.get')
    def test_fetch_from_api_connection_error(self, mock_get):
        """Test fetch handles connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(requests.RequestException):
            self.ingestion._fetch_from_api('Delhi')

    @patch('requests.get')
    def test_fetch_from_api_timeout(self, mock_get):
        """Test fetch handles timeout errors."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(requests.RequestException):
            self.ingestion._fetch_from_api('Delhi')

    @patch('requests.get')
    def test_fetch_from_api_http_error(self, mock_get):
        """Test fetch handles HTTP errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_response
        
        with pytest.raises(requests.RequestException):
            self.ingestion._fetch_from_api('Delhi')


class TestIntegration:
    """Integration tests for IQAir ingestion."""

    @patch('requests.get')
    def test_end_to_end_single_city_fetch(self, mock_get):
        """Test end-to-end fetch for single city."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        mock_response = MagicMock()
        mock_response.json.return_value = create_valid_iqair_response('Delhi')
        mock_get.return_value = mock_response
        
        ingestion = IQAirDataIngestion(api_key='test_key')
        result = ingestion.fetch_current_aqi('Delhi')
        
        assert result['city'] == 'Delhi'
        assert result['aqi'] == 150.5
        assert 'timestamp' in result
        assert 'source' in result

    @patch('requests.get')
    def test_end_to_end_all_cities_fetch(self, mock_get):
        """Test end-to-end fetch for all cities."""
        from src.data_ingestion.iqair_ingestion import IQAirDataIngestion
        
        mock_response = MagicMock()
        
        def side_effect(*args, **kwargs):
            city = args[0].split('/')[-1].split('?')[0]
            mock_response.json.return_value = create_valid_iqair_response(city)
            return mock_response
        
        mock_get.side_effect = side_effect
        
        ingestion = IQAirDataIngestion(api_key='test_key')
        df = ingestion.fetch_all_cities_aqi()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert all(city in df['city'].values for city in ingestion.cities)
