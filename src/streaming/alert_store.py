import sqlite3
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertStore:
    
    def __init__(self, db_path: str = 'data/alerts.db'):
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_schema()
            logger.info(f"Alert store initialized: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize alert store: {e}")
            raise

    def _create_schema(self) -> None:
        
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    city TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT,
                    current_aqi REAL,
                    predicted_aqi REAL,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT 0,
                    acknowledged_at DATETIME,
                    acknowledged_by TEXT,
                    message TEXT,
                    recommendation TEXT,
                    threshold REAL
                )
            ''')

            # Create indices for efficient queries
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_city
                ON alerts(city)
            ''')

            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_level
                ON alerts(level)
            ''')

            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON alerts(timestamp)
            ''')

            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_acknowledged
                ON alerts(acknowledged)
            ''')

            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_city_level
                ON alerts(city, level)
            ''')

            self.conn.commit()
            logger.debug("Alert store schema created")

        except sqlite3.Error as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def store_alert(self, alert: Dict) -> str:
        
        required_fields = ['alert_type', 'city', 'level', 'timestamp']
        missing_fields = [f for f in required_fields if f not in alert]
        if missing_fields:
            raise KeyError(f"Alert missing required fields: {missing_fields}")

        alert_id = alert.get('alert_id', str(uuid.uuid4()))

        try:
            self.cursor.execute('''
                INSERT INTO alerts
                (alert_id, alert_type, city, level, category, current_aqi,
                 predicted_aqi, timestamp, message, recommendation, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                alert['alert_type'],
                alert['city'],
                alert['level'],
                alert.get('category'),
                alert.get('current_aqi'),
                alert.get('predicted_aqi'),
                alert['timestamp'],
                alert.get('message'),
                alert.get('recommendation'),
                alert.get('threshold')
            ))

            self.conn.commit()
            logger.debug(f"Alert stored: {alert_id}")
            return alert_id

        except sqlite3.Error as e:
            logger.error(f"Failed to store alert: {e}")
            raise

    def store_batch(self, alerts: List[Dict]) -> List[str]:
        
        alert_ids = []

        for alert in alerts:
            try:
                alert_id = self.store_alert(alert)
                alert_ids.append(alert_id)
            except Exception as e:
                logger.error(f"Error storing alert: {e}")

        return alert_ids

    def get_alert(self, alert_id: str) -> Optional[Dict]:
        
        try:
            self.cursor.execute(
                'SELECT * FROM alerts WHERE alert_id = ?',
                (alert_id,)
            )
            row = self.cursor.fetchone()

            if row is None:
                return None

            return self._row_to_dict(row)

        except sqlite3.Error as e:
            logger.error(f"Failed to get alert: {e}")
            return None

    def get_active_alerts(
        self,
        city: Optional[str] = None,
        level: Optional[str] = None
    ) -> List[Dict]:
        
        try:
            query = 'SELECT * FROM alerts WHERE acknowledged = 0'
            params = []

            if city:
                query += ' AND city = ?'
                params.append(city)

            if level:
                query += ' AND level = ?'
                params.append(level)

            query += ' ORDER BY timestamp DESC'

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    def get_alerts_by_city(
        self,
        city: str,
        hours: int = 24,
        acknowledged: Optional[bool] = None
    ) -> List[Dict]:
        
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            query = 'SELECT * FROM alerts WHERE city = ? AND timestamp > ?'
            params = [city, cutoff_time]

            if acknowledged is not None:
                query += ' AND acknowledged = ?'
                params.append(1 if acknowledged else 0)

            query += ' ORDER BY timestamp DESC'

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get alerts by city: {e}")
            return []

    def get_alerts_by_level(
        self,
        level: str,
        hours: int = 24
    ) -> List[Dict]:
        
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            self.cursor.execute('''
                SELECT * FROM alerts
                WHERE level = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (level, cutoff_time))

            rows = self.cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get alerts by level: {e}")
            return []

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: Optional[str] = None
    ) -> bool:
        
        try:
            self.cursor.execute('''
                UPDATE alerts
                SET acknowledged = 1, acknowledged_at = CURRENT_TIMESTAMP,
                    acknowledged_by = ?
                WHERE alert_id = ?
            ''', (acknowledged_by, alert_id))

            self.conn.commit()
            logger.debug(f"Alert acknowledged: {alert_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    def acknowledge_batch(
        self,
        alert_ids: List[str],
        acknowledged_by: Optional[str] = None
    ) -> int:
        
        count = 0

        for alert_id in alert_ids:
            if self.acknowledge_alert(alert_id, acknowledged_by):
                count += 1

        return count

    def get_stats(
        self,
        hours: int = 24
    ) -> Dict:
        
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            # Total alerts
            self.cursor.execute(
                'SELECT COUNT(*) FROM alerts WHERE timestamp > ?',
                (cutoff_time,)
            )
            total = self.cursor.fetchone()[0]

            # Active alerts
            self.cursor.execute(
                'SELECT COUNT(*) FROM alerts WHERE timestamp > ? AND acknowledged = 0',
                (cutoff_time,)
            )
            active = self.cursor.fetchone()[0]

            # By level
            self.cursor.execute('''
                SELECT level, COUNT(*) FROM alerts
                WHERE timestamp > ? AND acknowledged = 0
                GROUP BY level
            ''', (cutoff_time,))
            by_level = dict(self.cursor.fetchall())

            # By city
            self.cursor.execute('''
                SELECT city, COUNT(*) FROM alerts
                WHERE timestamp > ? AND acknowledged = 0
                GROUP BY city
            ''', (cutoff_time,))
            by_city = dict(self.cursor.fetchall())

            return {
                'total_alerts': total,
                'active_alerts': active,
                'by_level': by_level,
                'by_city': by_city,
                'lookback_hours': hours
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def _row_to_dict(self, row: tuple) -> Dict:
        """
        Convert database row to dictionary.

        Args:
            row: Database row tuple

        Returns:
            Dictionary with column names as keys
        """
        columns = [
            'alert_id', 'alert_type', 'city', 'level', 'category',
            'current_aqi', 'predicted_aqi', 'timestamp', 'created_at',
            'acknowledged', 'acknowledged_at', 'acknowledged_by',
            'message', 'recommendation', 'threshold'
        ]
        return dict(zip(columns, row))

    def close(self) -> None:
        """Close database connection."""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Alert store closed")
        except sqlite3.Error as e:
            logger.error(f"Error closing alert store: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
