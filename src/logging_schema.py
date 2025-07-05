from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum, auto
import json
import uuid

class IterationStatus(Enum):
    """Enumeration of possible iteration statuses."""
    STARTED = auto()
    IN_PROGRESS = auto()
    SUCCESS = auto()
    FAILED = auto()
    TERMINATED = auto()

@dataclass
class IterationMetrics:
    """
    Comprehensive metrics tracking for each ALP loop iteration.
    
    Captures detailed performance and state information during learning iterations.
    """
    iteration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: IterationStatus = IterationStatus.STARTED
    
    # Performance metrics
    duration_ms: float = 0.0
    computational_resources: Dict[str, float] = field(default_factory=dict)
    
    # Learning process details
    learning_rate: Optional[float] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Errors and exceptions
    error_details: Optional[Dict[str, Any]] = None
    
    # Additional configuration and context
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, status: IterationStatus) -> None:
        """Update the current iteration status."""
        self.status = status
    
    def record_performance(self, duration: float, resources: Dict[str, float]) -> None:
        """Record performance metrics for the iteration."""
        self.duration_ms = duration
        self.computational_resources = resources
    
    def record_learning_metrics(
        self, 
        learning_rate: Optional[float] = None, 
        loss: Optional[float] = None, 
        accuracy: Optional[float] = None
    ) -> None:
        """Record key learning metrics."""
        self.learning_rate = learning_rate
        self.loss = loss
        self.accuracy = accuracy
    
    def record_error(self, error_details: Dict[str, Any]) -> None:
        """Record error details if an exception occurs."""
        self.status = IterationStatus.FAILED
        self.error_details = error_details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert iteration metrics to a dictionary representation."""
        return {
            'iteration_id': self.iteration_id,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.name,
            'duration_ms': self.duration_ms,
            'computational_resources': self.computational_resources,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'error_details': self.error_details,
            'configuration': self.configuration
        }
    
    def to_json(self) -> str:
        """Serialize iteration metrics to a JSON string."""
        return json.dumps(self.to_dict(), default=str)

class ALPLoggingManager:
    """
    Manages logging and tracking of ALP loop iterations.
    
    Provides comprehensive tracking and persistence of iteration metrics.
    """
    def __init__(self, max_iterations: int = 100):
        """
        Initialize the logging manager.
        
        Args:
            max_iterations: Maximum number of iterations to track
        """
        self.max_iterations: int = max_iterations
        self.iterations: List[IterationMetrics] = []
    
    def start_iteration(self, configuration: Optional[Dict[str, Any]] = None) -> IterationMetrics:
        """
        Start a new iteration and return the metrics tracker.
        
        Args:
            configuration: Optional configuration for the iteration
        
        Returns:
            IterationMetrics instance
        """
        if len(self.iterations) >= self.max_iterations:
            self.iterations.pop(0)
        
        iteration = IterationMetrics()
        if configuration:
            iteration.configuration = configuration
        
        self.iterations.append(iteration)
        return iteration
    
    def get_latest_iteration(self) -> Optional[IterationMetrics]:
        """
        Retrieve the most recent iteration metrics.
        
        Returns:
            The latest IterationMetrics or None if no iterations exist
        """
        return self.iterations[-1] if self.iterations else None
    
    def get_all_iterations(self) -> List[IterationMetrics]:
        """
        Retrieve all tracked iterations.
        
        Returns:
            List of all IterationMetrics
        """
        return self.iterations.copy()