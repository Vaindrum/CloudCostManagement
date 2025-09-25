"""
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
