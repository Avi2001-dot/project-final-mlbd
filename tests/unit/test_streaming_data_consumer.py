import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.streaming.streaming_data_consumer import StreamingDataConsumer


class TestStreamingDataConsumer:
    """Test cases for StreamingDataConsumer."""

    @pytest.fixture
    def mock_kafka_consumer(self):
        """Mock Kafka consumer."""
        with patch('src.streaming.streaming_data_consumer.KafkaConsumer'):
            yield

    def test_initialization(self, mock_kafka_consumer):
        """Test consumer initialization."""
        consumer = StreamingDataConsumer()
        assert consumer.bootstrap_servers == 'localhost:9092'
        assert consumer.group_id == 'aqi_prediction_system'
        assert consumer.consumer is not None

    def test_initialization_with_topics(self, mock_kafka_consumer):
        """Test consumer initialization with topics."""
        topics = ['aqi_events_delhi', 'aqi_events_mumbai']
        consumer = StreamingDataConsumer(topics=topics)
        assert consumer.topics == topics

    def test_deserialize_event(self, mock_kafka_consumer):
        """Test event deserialization."""
        consumer = StreamingDataConsumer()
        event_dict = {'timestamp': 1234567890, 'aqi': 150.5, 'city': 'Delhi'}
        serialized = json.dumps(event_dict).encode('utf-8')
        deserialized = consumer._deserialize_event(serialized)
        assert deserialized == event_dict

    def test_deserialize_event_invalid(self, mock_kafka_consumer):
        """Test deserialization with invalid JSON."""
        consumer = StreamingDataConsumer()
        with pytest.raises(json.JSONDecodeError):
            consumer._deserialize_event(b'invalid json')

    def test_subscribe(self, mock_kafka_consumer):
        """Test subscribing to topics."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()

            topics = ['aqi_events_delhi', 'aqi_events_mumbai']
            consumer.subscribe(topics)

            consumer.consumer.subscribe.assert_called_once_with(topics)
            assert consumer.topics == topics

    def test_consume_events_generator(self, mock_kafka_consumer):
        """Test consuming events as generator."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer(topics=['aqi_events_delhi'])
            consumer.consumer = Mock()

            # Mock message objects
            message1 = Mock(value={'timestamp': 1234567890, 'aqi': 150.5})
            message2 = Mock(value={'timestamp': 1234567891, 'aqi': 160.0})

            # Mock poll to return messages then empty
            consumer.consumer.poll.side_effect = [
                {Mock(): [message1, message2]},
                {},
                KeyboardInterrupt()
            ]

            events = []
            try:
                for event in consumer.consume_events():
                    events.append(event)
            except KeyboardInterrupt:
                pass

            assert len(events) == 2
            assert events[0]['aqi'] == 150.5
            assert events[1]['aqi'] == 160.0

    def test_consume_batch(self, mock_kafka_consumer):
        """Test consuming batch of events."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer(topics=['aqi_events_delhi'])
            consumer.consumer = Mock()

            message1 = Mock(value={'timestamp': 1234567890, 'aqi': 150.5})
            message2 = Mock(value={'timestamp': 1234567891, 'aqi': 160.0})

            consumer.consumer.poll.return_value = {
                Mock(): [message1, message2]
            }

            events = consumer.consume_batch()
            assert len(events) == 2
            assert events[0]['aqi'] == 150.5

    def test_consume_batch_empty(self, mock_kafka_consumer):
        """Test consuming batch with no messages."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer(topics=['aqi_events_delhi'])
            consumer.consumer = Mock()
            consumer.consumer.poll.return_value = {}

            events = consumer.consume_batch()
            assert len(events) == 0

    def test_seek_to_beginning(self, mock_kafka_consumer):
        """Test seeking to beginning."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()

            consumer.seek_to_beginning()
            consumer.consumer.seek_to_beginning.assert_called_once()

    def test_seek_to_end(self, mock_kafka_consumer):
        """Test seeking to end."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()

            consumer.seek_to_end()
            consumer.consumer.seek_to_end.assert_called_once()

    def test_commit(self, mock_kafka_consumer):
        """Test committing offsets."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()

            consumer.commit()
            consumer.consumer.commit.assert_called_once()

    def test_close(self, mock_kafka_consumer):
        """Test closing consumer."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()

            consumer.close()
            consumer.consumer.close.assert_called_once()

    def test_context_manager(self, mock_kafka_consumer):
        """Test context manager usage."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            with StreamingDataConsumer() as consumer:
                assert consumer is not None
                consumer.consumer = Mock()

            consumer.consumer.close.assert_called_once()

    def test_consume_events_not_subscribed(self, mock_kafka_consumer):
        """Test consuming without subscription."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()
            consumer.topics = []

            events = list(consumer.consume_events())
            assert len(events) == 0

    def test_consume_batch_not_subscribed(self, mock_kafka_consumer):
        """Test batch consume without subscription."""
        with patch.object(StreamingDataConsumer, '_initialize_consumer'):
            consumer = StreamingDataConsumer()
            consumer.consumer = Mock()
            consumer.topics = []

            events = consumer.consume_batch()
            assert len(events) == 0
