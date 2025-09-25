# Initialize the AI Cloud Cost Management Agent project structure
import os
import json
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

# Create project directory structure
project_structure = {
    "ai_cloud_cost_agent/": {
        "README.md": "",
        "requirements.txt": "",
        "setup.py": "",
        "docker-compose.yml": "",
        "Dockerfile": "",
        ".env.example": "",
        "src/": {
            "__init__.py": "",
            "main.py": "",
            "config/": {
                "__init__.py": "",
                "settings.py": "",
                "database.py": ""
            },
            "data_ingestion/": {
                "__init__.py": "",
                "cloud_connectors.py": "",
                "kafka_consumer.py": "",
                "data_validator.py": ""
            },
            "ml_engine/": {
                "__init__.py": "",
                "cost_predictor.py": "",
                "anomaly_detector.py": "",
                "optimization_engine.py": ""
            },
            "decision_engine/": {
                "__init__.py": "",
                "policy_engine.py": "",
                "decision_maker.py": "",
                "approval_workflow.py": ""
            },
            "action_executor/": {
                "__init__.py": "",
                "cloud_actions.py": "",
                "resource_optimizer.py": "",
                "safety_controller.py": ""
            },
            "api/": {
                "__init__.py": "",
                "app.py": "",
                "routes.py": "",
                "models.py": "",
                "auth.py": ""
            },
            "utils/": {
                "__init__.py": "",
                "logger.py": "",
                "helpers.py": "",
                "constants.py": ""
            }
        },
        "tests/": {
            "__init__.py": "",
            "test_data_ingestion.py": "",
            "test_ml_engine.py": "",
            "test_api.py": ""
        },
        "data/": {
            "sample_data/": {},
            "models/": {},
            "logs/": {}
        },
        "scripts/": {
            "setup_db.py": "",
            "deploy.sh": "",
            "run_tests.sh": ""
        },
        "docs/": {
            "API.md": "",
            "ARCHITECTURE.md": "",
            "DEPLOYMENT.md": ""
        },
        "frontend/": {
            "public/": {},
            "src/": {
                "components/": {},
                "pages/": {},
                "services/": {}
            },
            "package.json": "",
            "index.html": ""
        }
    }
}

def create_directory_structure(base_path, structure):
    """Recursively create directory structure"""
    for name, content in structure.items():
        full_path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(full_path, exist_ok=True)
            create_directory_structure(full_path, content)
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

# Create the project structure
base_dir = "ai_cloud_cost_agent"
create_directory_structure(".", project_structure)
print(f"✅ Project structure created: {base_dir}/")

# Generate sample cost data for testing
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-09-25', freq='D')
sample_data = []

for date in dates:
    # Simulate multiple services and resources
    services = ['EC2', 'RDS', 'S3', 'Lambda', 'CloudFront', 'ELB']
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    
    for service in services:
        for region in regions:
            # Generate realistic cost patterns
            base_cost = random.uniform(100, 1000)
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekly_factor = 1 + 0.05 * np.sin(2 * np.pi * date.weekday() / 7)
            noise = np.random.normal(1, 0.1)
            
            daily_cost = base_cost * seasonal_factor * weekly_factor * noise
            
            sample_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'service': service,
                'region': region,
                'daily_cost': round(daily_cost, 2),
                'resource_count': random.randint(1, 50),
                'provider': 'AWS'
            })

# Create sample data DataFrame
df = pd.DataFrame(sample_data)
df.to_csv(f'{base_dir}/data/sample_data/aws_cost_data.csv', index=False)
print(f"✅ Generated sample cost data: {len(df)} records")

# Generate anomaly data (sudden cost spikes)
anomaly_dates = np.random.choice(dates, size=20, replace=False)
anomaly_data = []

for date in anomaly_dates:
    anomaly_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'service': random.choice(['EC2', 'RDS', 'Lambda']),
        'region': random.choice(['us-east-1', 'us-west-2']),
        'anomaly_score': round(random.uniform(0.7, 0.95), 3),
        'cost_spike': round(random.uniform(1000, 5000), 2),
        'detected': True
    })

anomaly_df = pd.DataFrame(anomaly_data)
anomaly_df.to_csv(f'{base_dir}/data/sample_data/anomaly_data.csv', index=False)
print(f"✅ Generated anomaly data: {len(anomaly_df)} records")

print("\n📊 Data Summary:")
print(f"Total cost records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Services covered: {df['service'].nunique()}")
print(f"Regions covered: {df['region'].nunique()}")
print(f"Total cost simulated: ${df['daily_cost'].sum():,.2f}")
print(f"Anomalies detected: {len(anomaly_df)}")