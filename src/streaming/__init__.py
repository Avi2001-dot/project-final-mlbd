

def __getattr__(name):
    _map = {
        'StreamingDataProducer':    ('src.streaming.streaming_data_producer',    'StreamingDataProducer'),
        'StreamingDataConsumer':    ('src.streaming.streaming_data_consumer',    'StreamingDataConsumer'),
        'StreamingFeatureComputer': ('src.streaming.streaming_feature_computer', 'StreamingFeatureComputer'),
        'StreamingInferencePipeline': ('src.streaming.streaming_inference_pipeline', 'StreamingInferencePipeline'),
        'RuleBasedAlertSystem':     ('src.streaming.rule_based_alert_system',    'RuleBasedAlertSystem'),
        'ModelBasedAlertSystem':    ('src.streaming.model_based_alert_system',   'ModelBasedAlertSystem'),
        'AlertDeduplicator':        ('src.streaming.alert_deduplicator',         'AlertDeduplicator'),
        'AlertStore':               ('src.streaming.alert_store',                'AlertStore'),
        'AlertService':             ('src.streaming.alert_service',              'AlertService'),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'src.streaming' has no attribute {name!r}")


__all__ = [
    'StreamingDataProducer', 'StreamingDataConsumer',
    'StreamingFeatureComputer', 'StreamingInferencePipeline',
    'RuleBasedAlertSystem', 'ModelBasedAlertSystem',
    'AlertDeduplicator', 'AlertStore', 'AlertService',
]
