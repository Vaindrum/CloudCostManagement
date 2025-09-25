# Create core application files - starting with configuration
print("🏗️ Creating core application files...")

# src/config/settings.py
settings_content = '''"""
Configuration settings for AI Cloud Cost Management Agent
"""
import os
from typing import Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "AI Cloud Cost Management Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/ai_cost_agent"
    redis_url: str = "redis://localhost:6379/0"
    
    # Cloud Credentials
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: str = "us-east-1"
    
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    
    gcp_project_id: Optional[str] = None
    google_application_credentials: Optional[str] = None
    
    # API Configuration
    api_secret_key: str = "your-super-secret-key"
    api_access_token_expire_minutes: int = 30
    api_rate_limit_requests: int = 1000
    api_rate_limit_period: int = 3600
    
    # ML Configuration
    ml_model_path: str = "data/models/"
    ml_prediction_horizon_days: int = 30
    ml_retrain_interval_hours: int = 24
    ml_anomaly_threshold: float = 0.8
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_cost_events_topic: str = "cost_events"
    kafka_optimization_topic: str = "optimization_actions"
    
    # Monitoring
    prometheus_port: int = 8000
    
    @validator('api_secret_key')
    def secret_key_validator(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
'''

with open('ai_cloud_cost_agent/src/config/settings.py', 'w') as f:
    f.write(settings_content)

print("✅ Created settings.py")

# src/config/database.py
database_content = '''"""
Database configuration and connection management
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings
import redis

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Redis connection
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis():
    """Get Redis client"""
    return redis_client
'''

with open('ai_cloud_cost_agent/src/config/database.py', 'w') as f:
    f.write(database_content)

print("✅ Created database.py")

# src/utils/logger.py
logger_content = '''"""
Structured logging configuration
"""
import structlog
import logging
import sys
from .settings import settings

def configure_logging():
    """Configure structured logging"""
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper())
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """Get structured logger instance"""
    return structlog.get_logger(name)

# Configure logging on import
configure_logging()
'''

with open('ai_cloud_cost_agent/src/utils/logger.py', 'w') as f:
    f.write(logger_content)

print("✅ Created logger.py")

# src/utils/constants.py
constants_content = '''"""
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
'''

with open('ai_cloud_cost_agent/src/utils/constants.py', 'w') as f:
    f.write(constants_content)

print("✅ Created constants.py")

# src/utils/helpers.py
helpers_content = '''"""
Utility helper functions
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import json
from decimal import Decimal

def generate_hash(data: str) -> str:
    """Generate SHA-256 hash of string data"""
    return hashlib.sha256(data.encode()).hexdigest()

def parse_cost_amount(amount: str) -> float:
    """Parse cost amount from string to float"""
    if isinstance(amount, (int, float)):
        return float(amount)
    
    # Remove currency symbols and spaces
    cleaned = amount.replace('$', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    
    return ((new_value - old_value) / old_value) * 100

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string"""
    return f"${amount:,.2f} {currency}"

def get_date_range(days: int) -> List[str]:
    """Get list of dates for the past N days"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    return date_list

def normalize_service_name(provider: str, service: str) -> str:
    """Normalize service names across different cloud providers"""
    normalization_map = {
        "aws": {
            "AmazonEC2": "EC2",
            "Amazon Elastic Compute Cloud": "EC2",
            "AmazonRDS": "RDS",
            "Amazon Relational Database Service": "RDS",
            "AmazonS3": "S3",
            "Amazon Simple Storage Service": "S3"
        },
        "azure": {
            "Microsoft.Compute/virtualMachines": "Virtual Machines",
            "Microsoft.Storage/storageAccounts": "Storage Accounts",
            "Microsoft.Sql/servers": "SQL Database"
        },
        "gcp": {
            "compute.googleapis.com": "Compute Engine",
            "storage.googleapis.com": "Cloud Storage",
            "sql.googleapis.com": "Cloud SQL"
        }
    }
    
    provider_map = normalization_map.get(provider.lower(), {})
    return provider_map.get(service, service)

def validate_cloud_credentials(provider: str, credentials: Dict[str, Any]) -> bool:
    """Validate cloud provider credentials format"""
    required_fields = {
        "aws": ["access_key_id", "secret_access_key"],
        "azure": ["client_id", "client_secret", "tenant_id"],
        "gcp": ["project_id", "service_account_key"]
    }
    
    if provider not in required_fields:
        return False
    
    return all(field in credentials for field in required_fields[provider])

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    return numerator / denominator if denominator != 0 else default

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def serialize_json(data: Any) -> str:
    """Serialize data to JSON with custom encoder"""
    return json.dumps(data, cls=DecimalEncoder, indent=2)
'''

with open('ai_cloud_cost_agent/src/utils/helpers.py', 'w') as f:
    f.write(helpers_content)

print("✅ Created helpers.py")

print("\n🔧 Created core utility and configuration files!")
print("Next: Creating data ingestion layer...")