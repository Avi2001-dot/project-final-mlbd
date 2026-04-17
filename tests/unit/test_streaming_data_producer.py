import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.streaming.streaming_data_producer import StreamingDataProducer


class TestStreamingDataProducer:
    """Test cases for StreamingDataProducer."""

    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer."""
        with patch('src.streaming.streaming_data_producer.KafkaProducer'):
            yield

    def test_initialization(self, mock_kafka_producer):
        """Test producer initialization."""
        producer = StreamingDataProducer()
        assert producer.bootstrap_servers == 'localhost:9092'
        assert producer.topic_prefix == 'aqi_events'
        assert producer.producer is not None

    def test_get_topic_name(self, mock_kafka_producer):
        """Test topic name generation."""
        producer = StreamingDataProducer()
        assert producer._get_topic_name('Delhi') == 'aqi_events_delhi'
        assert producer._get_topic_name('Mumbai') == 'aqi_events_mumbai'

    def test_serialize_event(self, mock_kafka_producer):
        """Test event serialization."""
        producer = StreamingDataProducer()
        event = {'timestamp': 1234567890, 'aqi': 150.5}
        serialized = producer._serialize_event(event)
        assert isinstance(serialized, bytes)
        assert json.loads(serialized) == event

    def test_serialize_event_invalid(self, mock_kafka_producer):
        """Test serialization with non-serializable object."""
        producer = StreamingDataProducer()
        event = {'timestamp': 1234567890, 'obj': object()}
        with pytest.raises(TypeError):
            producer._serialize_event(event)

    def test_send_event_success(self, mock_kafka_producer):
        """Test successful event sending."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()
            future = Mock()
            future.get.return_value = Mock(partition=0, offset=100)
            producer.producer.send.return_value = future

            event = {'timestamp': 1234567890, 'aqi': 150.5}
            result = producer.send_event('Delhi', event)
            assert result is True
            producer.producer.send.assert_called_once()

    def test_send_event_missing_fields(self, mock_kafka_producer):
        """Test sending event with missing required fields."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()

            event = {'timestamp': 1234567890}  # Missing 'aqi'
            result = producer.send_event('Delhi', event)
            assert result is False

    def test_send_event_invalid_type(self, mock_kafka_producer):
        """Test sending non-dict event."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()

            result = producer.send_event('Delhi', 'not a dict')
            assert result is False

    def test_send_event_retry_logic(self, mock_kafka_producer):
        """Test retry logic on failure."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()
            producer.producer.send.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                Mock(get=Mock(return_value=Mock(partition=0, offset=100)))
            ]

            event = {'timestamp': 1234567890, 'aqi': 150.5}
            result = producer.send_event('Delhi', event, max_retries=3)
            assert result is True
            assert producer.producer.send.call_count == 3

    def test_send_batch_events(self, mock_kafka_producer):
        """Test batch event sending."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()
            future = Mock()
            future.get.return_value = Mock(partition=0, offset=100)
            producer.producer.send.return_value = future

            events = [
                {'timestamp': 1234567890, 'aqi': 150.5},
                {'timestamp': 1234567891, 'aqi': 160.0},
                {'timestamp': 1234567892, 'aqi': 155.0}
            ]

            result = producer.send_batch_events('Delhi', events)
            assert result['sent'] == 3
            assert result['failed'] == 0

    def test_send_batch_events_mixed_results(self, mock_kafka_producer):
        """Test batch sending with mixed success/failure."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()

            # First event succeeds, second fails, third succeeds
            future = Mock()
            future.get.return_value = Mock(partition=0, offset=100)
            producer.producer.send.side_effect = [
                future,
                Exception("Send failed"),
                future
            ]

            events = [
                {'timestamp': 1234567890, 'aqi': 150.5},
                {'timestamp': 1234567891, 'aqi': 160.0},
                {'timestamp': 1234567892, 'aqi': 155.0}
            ]

            result = producer.send_batch_events('Delhi', events, max_retries=1)
            assert result['sent'] == 2
            assert result['failed'] == 1

    def test_flush(self, mock_kafka_producer):
        """Test flushing pending messages."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()

            producer.flush()
            producer.producer.flush.assert_called_once()

    def test_close(self, mock_kafka_producer):
        """Test closing producer."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()

            producer.close()
            producer.producer.flush.assert_called_once()
            producer.producer.close.assert_called_once()

    def test_context_manager(self, mock_kafka_producer):
        """Test context manager usage."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            with StreamingDataProducer() as producer:
                assert producer is not None
                producer.producer = Mock()

            producer.producer.close.assert_called_once()

    def test_send_event_with_city_in_event(self, mock_kafka_producer):
        """Test that city is added to event."""
        with patch.object(StreamingDataProducer, '_initialize_producer'):
            producer = StreamingDataProducer()
            producer.producer = Mock()
            future = Mock()
            future.get.return_value = Mock(partition=0, offset=100)
            producer.producer.send.return_value = future

            event = {'timestamp': 1234567890, 'aqi': 150.5}
            producer.send_event('Delhi', event)

            # Check that city was added to event
            call_args = producer.producer.send.call_args
            sent_event = call_args[1]['value']
            assert sent_event['city'] == 'Delhi'
