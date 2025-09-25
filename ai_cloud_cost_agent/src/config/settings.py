"""
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
