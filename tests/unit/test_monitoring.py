import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.utils.monitoring import (
    SystemHealthMonitor,
    ExecutionTimeTracker,
    EventMetricsCollector,
    PerformanceMonitor,
    get_performance_monitor,
    start_operation_timer,
    end_operation_timer,
    record_event_metric
)
from src.utils.logging_integration import (
    log_operation,
    log_data_processing,
    log_model_training,
    log_prediction,
    log_alert,
    StructuredLogger,
    create_structured_logger
)


class TestSystemHealthMonitor:
    """Test SystemHealthMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = SystemHealthMonitor(history_size=50)
        assert monitor.history_size == 50
        assert len(monitor.cpu_history) == 0

    def test_collect_metrics(self):
        """Test metric collection."""
        monitor = SystemHealthMonitor()
        metrics = monitor.collect_metrics()

        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_percent' in metrics
        assert 'timestamp' in metrics
        assert 0 <= metrics['cpu_percent'] <= 100
        assert 0 <= metrics['memory_percent'] <= 100
        assert 0 <= metrics['disk_percent'] <= 100

    def test_history_tracking(self):
        """Test that metrics are tracked in history."""
        monitor = SystemHealthMonitor(history_size=5)

        for _ in range(3):
            monitor.collect_metrics()

        assert len(monitor.cpu_history) == 3
        assert len(monitor.memory_history) == 3
        assert len(monitor.disk_history) == 3

    def test_history_max_size(self):
        """Test that history respects max size."""
        monitor = SystemHealthMonitor(history_size=3)

        for _ in range(5):
            monitor.collect_metrics()

        assert len(monitor.cpu_history) == 3
        assert len(monitor.memory_history) == 3

    def test_get_average_metrics(self):
        """Test average metrics calculation."""
        monitor = SystemHealthMonitor()

        for _ in range(3):
            monitor.collect_metrics()

        avg_metrics = monitor.get_average_metrics()

        assert 'avg_cpu_percent' in avg_metrics
        assert 'avg_memory_percent' in avg_metrics
        assert 'max_cpu_percent' in avg_metrics
        assert 'min_cpu_percent' in avg_metrics

    def test_is_healthy_default_thresholds(self):
        """Test health check with default thresholds."""
        monitor = SystemHealthMonitor()

        is_healthy, warnings = monitor.is_healthy()

        assert isinstance(is_healthy, bool)
        assert isinstance(warnings, list)

    def test_is_healthy_with_warnings(self):
        """Test health check generates warnings."""
        monitor = SystemHealthMonitor()

        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=90):
            with patch('psutil.virtual_memory') as mock_mem:
                with patch('psutil.disk_usage') as mock_disk:
                    mock_mem.return_value = MagicMock(percent=50)
                    mock_disk.return_value = MagicMock(percent=50)

                    is_healthy, warnings = monitor.is_healthy(cpu_threshold=80)

                    assert not is_healthy
                    assert len(warnings) > 0


class TestExecutionTimeTracker:
    """Test ExecutionTimeTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ExecutionTimeTracker()
        assert len(tracker.timings) == 0
        assert len(tracker.active_timers) == 0

    def test_start_and_end_timer(self):
        """Test starting and ending a timer."""
        tracker = ExecutionTimeTracker()

        tracker.start_timer('test_operation')
        time.sleep(0.1)
        duration = tracker.end_timer('test_operation')

        assert duration >= 0.1
        assert 'test_operation' in tracker.timings
        assert len(tracker.timings['test_operation']) == 1

    def test_end_timer_without_start(self):
        """Test ending timer that wasn't started."""
        tracker = ExecutionTimeTracker()

        with pytest.raises(ValueError):
            tracker.end_timer('nonexistent_operation')

    def test_multiple_timings(self):
        """Test tracking multiple timings for same operation."""
        tracker = ExecutionTimeTracker()

        for _ in range(3):
            tracker.start_timer('operation')
            time.sleep(0.05)
            tracker.end_timer('operation')

        assert len(tracker.timings['operation']) == 3

    def test_get_statistics(self):
        """Test getting statistics for operation."""
        tracker = ExecutionTimeTracker()

        for i in range(3):
            tracker.start_timer('operation')
            time.sleep(0.05 * (i + 1))
            tracker.end_timer('operation')

        stats = tracker.get_statistics('operation')

        assert stats['count'] == 3
        assert 'min_seconds' in stats
        assert 'max_seconds' in stats
        assert 'mean_seconds' in stats
        assert stats['min_seconds'] <= stats['mean_seconds'] <= stats['max_seconds']

    def test_get_all_statistics(self):
        """Test getting statistics for all operations."""
        tracker = ExecutionTimeTracker()

        tracker.start_timer('op1')
        time.sleep(0.05)
        tracker.end_timer('op1')

        tracker.start_timer('op2')
        time.sleep(0.05)
        tracker.end_timer('op2')

        all_stats = tracker.get_all_statistics()

        assert 'op1' in all_stats
        assert 'op2' in all_stats
        assert all_stats['op1']['count'] == 1
        assert all_stats['op2']['count'] == 1

    def test_reset(self):
        """Test resetting tracker."""
        tracker = ExecutionTimeTracker()

        tracker.start_timer('operation')
        time.sleep(0.05)
        tracker.end_timer('operation')

        assert len(tracker.timings) > 0

        tracker.reset()

        assert len(tracker.timings) == 0
        assert len(tracker.active_timers) == 0


class TestEventMetricsCollector:
    """Test EventMetricsCollector class."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = EventMetricsCollector(window_size=100)
        assert collector.window_size == 100
        assert collector.event_count == 0
        assert collector.error_count == 0

    def test_record_event(self):
        """Test recording events."""
        collector = EventMetricsCollector()

        collector.record_event(10.5, success=True)
        collector.record_event(12.3, success=True)

        assert collector.event_count == 2
        assert collector.error_count == 0

    def test_record_event_with_error(self):
        """Test recording failed events."""
        collector = EventMetricsCollector()

        collector.record_event(10.5, success=True)
        collector.record_event(15.0, success=False)

        assert collector.event_count == 2
        assert collector.error_count == 1

    def test_get_metrics(self):
        """Test getting event metrics."""
        collector = EventMetricsCollector()

        latencies = [10.0, 15.0, 12.0, 20.0, 11.0]
        for latency in latencies:
            collector.record_event(latency, success=True)

        metrics = collector.get_metrics()

        assert metrics['event_count'] == 5
        assert metrics['error_count'] == 0
        assert metrics['error_rate'] == 0.0
        assert metrics['min_latency_ms'] == 10.0
        assert metrics['max_latency_ms'] == 20.0
        assert 10.0 <= metrics['mean_latency_ms'] <= 20.0

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        collector = EventMetricsCollector()

        latencies = list(range(1, 101))  # 1 to 100
        for latency in latencies:
            collector.record_event(float(latency), success=True)

        metrics = collector.get_metrics()

        assert metrics['p95_latency_ms'] >= 95.0
        assert metrics['p99_latency_ms'] >= 99.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        collector = EventMetricsCollector()

        for _ in range(8):
            collector.record_event(10.0, success=True)

        for _ in range(2):
            collector.record_event(15.0, success=False)

        metrics = collector.get_metrics()

        assert metrics['event_count'] == 10
        assert metrics['error_count'] == 2
        assert metrics['error_rate'] == 0.2

    def test_reset(self):
        """Test resetting collector."""
        collector = EventMetricsCollector()

        collector.record_event(10.0, success=True)
        assert collector.event_count == 1

        collector.reset()

        assert collector.event_count == 0
        assert collector.error_count == 0
        assert len(collector.event_latencies) == 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(metrics_dir='test_metrics')
        assert monitor.system_monitor is not None
        assert monitor.execution_tracker is not None
        assert monitor.event_collector is not None

    def test_get_system_report(self):
        """Test getting system report."""
        monitor = PerformanceMonitor()
        report = monitor.get_system_report()

        assert 'timestamp' in report
        assert 'current_metrics' in report
        assert 'average_metrics' in report
        assert 'is_healthy' in report
        assert 'warnings' in report

    def test_get_performance_report(self):
        """Test getting performance report."""
        monitor = PerformanceMonitor()
        report = monitor.get_performance_report()

        assert 'timestamp' in report
        assert 'uptime_seconds' in report
        assert 'execution_times' in report
        assert 'event_metrics' in report

    def test_get_full_report(self):
        """Test getting full report."""
        monitor = PerformanceMonitor()
        report = monitor.get_full_report()

        assert 'timestamp' in report
        assert 'system' in report
        assert 'performance' in report

    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        monitor = PerformanceMonitor(metrics_dir=str(tmp_path))
        filepath = monitor.save_report('test_report')

        assert filepath is not None
        assert 'test_report' in filepath


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = create_structured_logger('test_module')
        assert logger is not None
        assert logger.module_name == 'test_module'

    def test_log_event(self):
        """Test logging an event."""
        logger = create_structured_logger('test_module')
        # Should not raise
        logger.log_event('TEST_EVENT', 'test_operation', level='INFO')

    def test_log_performance(self):
        """Test logging performance metrics."""
        logger = create_structured_logger('test_module')
        # Should not raise
        logger.log_performance(
            'test_operation',
            duration_seconds=1.5,
            records_processed=100
        )

    def test_log_data_quality(self):
        """Test logging data quality metrics."""
        logger = create_structured_logger('test_module')
        # Should not raise
        logger.log_data_quality(
            'test_stage',
            total_records=1000,
            valid_records=950,
            rejected_records=50,
            quality_score=95.0
        )

    def test_log_error(self):
        """Test logging an error."""
        logger = create_structured_logger('test_module')
        # Should not raise
        logger.log_error(
            'test_operation',
            'Test error message',
            error_type='TestError'
        )


class TestLoggingDecorators:
    """Test logging decorators."""

    def test_log_operation_decorator(self):
        """Test log_operation decorator."""
        @log_operation('test_operation', track_performance=False)
        def test_function():
            return 'success'

        result = test_function()
        assert result == 'success'

    def test_log_operation_with_exception(self):
        """Test log_operation decorator with exception."""
        @log_operation('test_operation', track_performance=False)
        def test_function():
            raise ValueError('Test error')

        with pytest.raises(ValueError):
            test_function()

    def test_log_data_processing(self):
        """Test log_data_processing function."""
        # Should not raise
        log_data_processing(
            'test_stage',
            input_records=1000,
            output_records=950,
            rejected_records=50,
            quality_score=95.0
        )

    def test_log_model_training(self):
        """Test log_model_training function."""
        metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
        # Should not raise
        log_model_training(
            'test_model',
            training_samples=1000,
            test_samples=200,
            metrics=metrics,
            duration_seconds=30.5
        )

    def test_log_prediction(self):
        """Test log_prediction function."""
        # Should not raise
        log_prediction(
            'Delhi',
            current_aqi=150.5,
            predicted_aqi=160.2,
            latency_ms=0.85,
            alert_triggered=True
        )

    def test_log_alert(self):
        """Test log_alert function."""
        # Should not raise
        log_alert(
            'rule-based',
            'Delhi',
            aqi_value=250.0,
            alert_level='Heavily Polluted',
            message='AQI exceeds safe threshold'
        )


class TestGlobalMonitoringFunctions:
    """Test global monitoring functions."""

    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

    def test_start_and_end_operation_timer(self):
        """Test global timer functions."""
        start_operation_timer('test_op')
        time.sleep(0.05)
        duration = end_operation_timer('test_op')

        assert duration >= 0.05

    def test_record_event_metric(self):
        """Test recording event metric."""
        monitor = get_performance_monitor()
        initial_count = monitor.event_collector.event_count

        record_event_metric(10.5, success=True)

        assert monitor.event_collector.event_count == initial_count + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
