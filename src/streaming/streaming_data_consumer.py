

import json
import time
from typing import Dict, Optional, Iterator, List
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.constants import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP
)

logger = get_logger(__name__)


class StreamingDataConsumer:

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = KAFKA_CONSUMER_GROUP,
        topics: Optional[List[str]] = None,
        auto_offset_reset: str = 'earliest'
    ):
        
        try:
            from kafka import KafkaConsumer
        except ImportError:
            logger.error(
                "kafka-python not installed. "
                "Install with: pip install kafka-python"
            )
            raise

        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or []
        self.consumer = None
        self._initialize_consumer(auto_offset_reset)

    def _initialize_consumer(self, auto_offset_reset: str) -> None:
        
        try:
            from kafka import KafkaConsumer

            self.consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=auto_offset_reset,
                value_deserializer=self._deserialize_event,
                enable_auto_commit=True,
                max_poll_records=100,
                session_timeout_ms=30000
            )

            if self.topics:
                self.consumer.subscribe(self.topics)
                logger.info(
                    f"Kafka consumer initialized: {self.bootstrap_servers}, "
                    f"group={self.group_id}, topics={self.topics}"
                )
            else:
                logger.info(
                    f"Kafka consumer initialized: {self.bootstrap_servers}, "
                    f"group={self.group_id}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise

    def _deserialize_event(self, message: bytes) -> Dict:
        
        try:
            return json.loads(message.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize event: {e}")
            raise

    def subscribe(self, topics: List[str]) -> None:
      
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        try:
            self.consumer.subscribe(topics)
            self.topics = topics
            logger.info(f"Subscribed to topics: {topics}")
        except Exception as e:
            logger.error(f"Failed to subscribe to topics: {e}")
            raise

    def consume_events(
        self,
        timeout_ms: int = 1000,
        max_records: Optional[int] = None
    ) -> Iterator[Dict]:
        
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        if not self.topics:
            logger.error("No topics subscribed")
            return

        try:
            while True:
                messages = self.consumer.poll(
                    timeout_ms=timeout_ms,
                    max_records=max_records
                )

                if not messages:
                    continue

                for topic_partition, records in messages.items():
                    for record in records:
                        try:
                            yield record.value
                        except Exception as e:
                            logger.error(
                                f"Error processing message from "
                                f"{topic_partition}: {e}"
                            )

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Error consuming events: {e}")
            raise

    def consume_batch(
        self,
        timeout_ms: int = 1000,
        max_records: int = 100
    ) -> List[Dict]:
       
        if not self.consumer:
            logger.error("Consumer not initialized")
            return []

        if not self.topics:
            logger.error("No topics subscribed")
            return []

        try:
            messages = self.consumer.poll(
                timeout_ms=timeout_ms,
                max_records=max_records
            )

            events = []
            for topic_partition, records in messages.items():
                for record in records:
                    try:
                        events.append(record.value)
                    except Exception as e:
                        logger.error(
                            f"Error processing message from "
                            f"{topic_partition}: {e}"
                        )

            return events

        except Exception as e:
            logger.error(f"Error consuming batch: {e}")
            return []

    def seek_to_beginning(self) -> None:
        """
        Seek to the beginning of all subscribed topics.

        Useful for replaying events from the start.
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        try:
            self.consumer.seek_to_beginning()
            logger.info("Seeked to beginning of all topics")
        except Exception as e:
            logger.error(f"Failed to seek to beginning: {e}")
            raise

    def seek_to_end(self) -> None:
        
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        try:
            self.consumer.seek_to_end()
            logger.info("Seeked to end of all topics")
        except Exception as e:
            logger.error(f"Failed to seek to end: {e}")
            raise

    def commit(self) -> None:
        """
        Commit current offsets.

        Manually commits the current offset position.
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        try:
            self.consumer.commit()
            logger.debug("Offsets committed")
        except Exception as e:
            logger.error(f"Failed to commit offsets: {e}")
            raise

    def close(self) -> None:
       
        try:
            if self.consumer:
                self.consumer.close()
                logger.info("Kafka consumer closed")
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
