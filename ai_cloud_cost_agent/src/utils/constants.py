"""
Application constants and enumerations
"""
from enum import Enum

class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

class ServiceType(str, Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORKING = "networking"
    SERVERLESS = "serverless"

class OptimizationAction(str, Enum):
    RIGHT_SIZE = "right_size"
    AUTO_SCALE = "auto_scale"
    INSTANCE_TYPE_CHANGE = "instance_type_change"
    RESERVED_INSTANCE = "reserved_instance"
    SPOT_INSTANCE = "spot_instance"
    STORAGE_CLASS_CHANGE = "storage_class_change"

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class OptimizationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

# AWS Service mappings
AWS_SERVICES = {
    "AmazonEC2": "compute",
    "AmazonRDS": "database",
    "AmazonS3": "storage",
    "AmazonCloudFront": "networking",
    "AWSLambda": "serverless",
    "ElasticLoadBalancing": "networking"
}

# Azure Service mappings
AZURE_SERVICES = {
    "Microsoft.Compute": "compute",
    "Microsoft.Storage": "storage",
    "Microsoft.Sql": "database",
    "Microsoft.Network": "networking",
    "Microsoft.Web": "serverless"
}

# GCP Service mappings
GCP_SERVICES = {
    "Compute Engine": "compute",
    "Cloud Storage": "storage",
    "Cloud SQL": "database",
    "Cloud Load Balancing": "networking",
    "Cloud Functions": "serverless"
}

# Cost thresholds for anomaly detection
COST_THRESHOLDS = {
    "daily_increase_percent": 50.0,
    "weekly_increase_percent": 25.0,
    "monthly_budget_percent": 80.0
}

# ML Model parameters
ML_PARAMS = {
    "lstm_sequence_length": 30,
    "lstm_units": 50,
    "prediction_confidence_threshold": 0.8,
    "anomaly_score_threshold": 0.7
}
