"""
Observability package for Kairos — production monitoring, error tracking,
usage analytics, and health insights.

Provides:
  - ErrorClassifier — sliding-window error aggregation with root-cause analysis
  - UsageTracker — token/cost/latency tracking with time-bucketed storage
  - AgentInsights — combined health reports, efficiency scoring, anomaly detection
"""

from kairos.observability.error_classifier import ErrorClassifier
from kairos.observability.usage_tracker import UsageTracker
from kairos.observability.insights import AgentInsights

__all__ = ["ErrorClassifier", "UsageTracker", "AgentInsights"]
