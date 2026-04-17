
import json
import time
from typing import Dict, Optional, List
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.constants import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS
)

logger = get_logger(__name__)


class StreamingDataProducer:
    
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic_prefix: str = 'aqi_events'
    ):
        
        try:
            from kafka import KafkaProducer
        except ImportError:
            logger.error(
                "kafka-python not installed. "
                "Install with: pip install kafka-python"
            )
            raise

        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.producer = None
        self._initialize_producer()

    def _initialize_producer(self) -> None:
        
        try:
            from kafka import KafkaProducer
            from kafka.errors import KafkaError

            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=self._serialize_event,
                acks='all',
                retries=MAX_RETRIES,
                max_in_flight_requests_per_connection=1
            )
            logger.info(
                f"Kafka producer initialized: {self.bootstrap_servers}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def _serialize_event(self, event: Dict) -> bytes:
        
        try:
            return json.dumps(event).encode('utf-8')
        except TypeError as e:
            logger.error(f"Failed to serialize event: {e}")
            raise

    def _get_topic_name(self, city: str) -> str:
        
        return f"{self.topic_prefix}_{city.lower()}"

    def send_event(
        self,
        city: str,
        event: Dict,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY_SECONDS
    ) -> bool:
        
        # Validate event
        if not isinstance(event, dict):
            logger.error(f"Event must be a dictionary, got {type(event)}")
            return False

        required_fields = ['timestamp', 'aqi']
        missing_fields = [f for f in required_fields if f not in event]
        if missing_fields:
            logger.error(
                f"Event missing required fields: {missing_fields}"
            )
            return False

        topic = self._get_topic_name(city)
        event_with_city = {**event, 'city': city}

        for attempt in range(max_retries):
            try:
                future = self.producer.send(topic, value=event_with_city)
                record_metadata = future.get(timeout=10)

                logger.debug(
                    f"Event sent to {topic} "
                    f"(partition={record_metadata.partition}, "
                    f"offset={record_metadata.offset})"
                )
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Failed to send event to {topic} "
                        f"(attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to send event to {topic} "
                        f"after {max_retries} attempts: {e}"
                    )
                    return False

        return False

    def send_batch_events(
        self,
        city: str,
        events: List[Dict],
        max_retries: int = MAX_RETRIES
    ) -> Dict[str, int]:
       
        if not isinstance(events, list):
            logger.error(f"Events must be a list, got {type(events)}")
            return {'sent': 0, 'failed': len(events)}

        results = {'sent': 0, 'failed': 0}

        for event in events:
            if self.send_event(city, event, max_retries=max_retries):
                results['sent'] += 1
            else:
                results['failed'] += 1

        logger.info(
            f"Batch send for {city}: {results['sent']} sent, "
            f"{results['failed']} failed"
        )
        return results

    def flush(self, timeout_ms: int = 10000) -> None:
        
        try:
            self.producer.flush(timeout_ms=timeout_ms)
            logger.debug("Producer flushed successfully")
        except Exception as e:
            logger.error(f"Failed to flush producer: {e}")
            raise

    def close(self) -> None:
        
        try:
            if self.producer:
                self.producer.flush()
                self.producer.close()
                logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing producer: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
