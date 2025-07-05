import pytest
import json
from datetime import datetime, timedelta
from src.logging_schema import IterationMetrics, IterationStatus, ALPLoggingManager

def test_iteration_metrics_initialization():
    """Test basic initialization of IterationMetrics."""
    metrics = IterationMetrics()
    
    assert metrics.iteration_id is not None
    assert isinstance(metrics.timestamp, datetime)
    assert metrics.status == IterationStatus.STARTED
    assert metrics.duration_ms == 0.0

def test_iteration_metrics_update_status():
    """Test updating iteration status."""
    metrics = IterationMetrics()
    metrics.update_status(IterationStatus.IN_PROGRESS)
    
    assert metrics.status == IterationStatus.IN_PROGRESS

def test_iteration_metrics_record_performance():
    """Test recording performance metrics."""
    metrics = IterationMetrics()
    performance_data = {
        'cpu_usage': 0.75,
        'memory_usage': 0.6
    }
    metrics.record_performance(duration=123.45, resources=performance_data)
    
    assert metrics.duration_ms == 123.45
    assert metrics.computational_resources == performance_data

def test_iteration_metrics_record_learning_metrics():
    """Test recording learning metrics."""
    metrics = IterationMetrics()
    metrics.record_learning_metrics(
        learning_rate=0.01,
        loss=0.5,
        accuracy=0.95
    )
    
    assert metrics.learning_rate == 0.01
    assert metrics.loss == 0.5
    assert metrics.accuracy == 0.95

def test_iteration_metrics_to_dict():
    """Test conversion of metrics to dictionary."""
    metrics = IterationMetrics()
    metrics_dict = metrics.to_dict()
    
    assert 'iteration_id' in metrics_dict
    assert 'timestamp' in metrics_dict
    assert 'status' in metrics_dict
    assert metrics_dict['status'] == 'STARTED'

def test_iteration_metrics_to_json():
    """Test JSON serialization of metrics."""
    metrics = IterationMetrics()
    metrics_json = metrics.to_json()
    
    parsed_json = json.loads(metrics_json)
    assert 'iteration_id' in parsed_json
    assert 'timestamp' in parsed_json
    assert 'status' in parsed_json

def test_alp_logging_manager_initialization():
    """Test ALPLoggingManager initialization."""
    manager = ALPLoggingManager(max_iterations=5)
    
    assert manager.max_iterations == 5
    assert len(manager.iterations) == 0

def test_alp_logging_manager_start_iteration():
    """Test starting iterations in the logging manager."""
    manager = ALPLoggingManager(max_iterations=3)
    
    # Start 3 iterations
    iteration1 = manager.start_iteration()
    iteration2 = manager.start_iteration()
    iteration3 = manager.start_iteration()
    
    assert len(manager.iterations) == 3
    assert manager.get_latest_iteration() == iteration3

def test_alp_logging_manager_max_iterations_limit():
    """Test that max iterations limit is respected."""
    manager = ALPLoggingManager(max_iterations=3)
    
    # Start 4 iterations
    manager.start_iteration()
    manager.start_iteration()
    manager.start_iteration()
    manager.start_iteration()  # This should remove the first iteration
    
    assert len(manager.iterations) == 3