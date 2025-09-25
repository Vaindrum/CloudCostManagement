"""
Data validation and quality checks for cost data
"""
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re
from ..utils.logger import get_logger
from ..utils.constants import CloudProvider
from ..utils.helpers import parse_cost_amount

logger = get_logger(__name__)


class CostDataValidator:
    """Validates cost data quality and format"""
    
    def __init__(self):
        self.required_fields = [
            'date', 'provider', 'service', 'region', 'cost', 'currency'
        ]
        self.validation_rules = {
            'date': self._validate_date,
            'provider': self._validate_provider,
            'service': self._validate_service,
            'region': self._validate_region,
            'cost': self._validate_cost,
            'currency': self._validate_currency
        }
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single cost record"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        # Validate field values
        for field, validator in self.validation_rules.items():
            if field in record:
                is_valid, error_msg = validator(record[field])
                if not is_valid:
                    errors.append(f"Invalid {field}: {error_msg}")
        
        return len(errors) == 0, errors
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate a batch of cost records"""
        valid_records = []
        invalid_records = []
        
        for i, record in enumerate(records):
            is_valid, errors = self.validate_record(record)
            
            if is_valid:
                valid_records.append(record)
            else:
                invalid_record = record.copy()
                invalid_record['_validation_errors'] = errors
                invalid_record['_record_index'] = i
                invalid_records.append(invalid_record)
        
        logger.info(
            f"Validation complete: {len(valid_records)} valid, {len(invalid_records)} invalid records"
        )
        
        return valid_records, invalid_records
    
    def _validate_date(self, date_str: Any) -> Tuple[bool, str]:
        """Validate date format"""
        if not isinstance(date_str, str):
            return False, "Date must be a string"
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True, ""
        except ValueError:
            return False, "Date must be in YYYY-MM-DD format"
    
    def _validate_provider(self, provider: Any) -> Tuple[bool, str]:
        """Validate cloud provider"""
        if not isinstance(provider, str):
            return False, "Provider must be a string"
        
        valid_providers = [p.value for p in CloudProvider]
        if provider.lower() not in valid_providers:
            return False, f"Provider must be one of: {valid_providers}"
        
        return True, ""
    
    def _validate_service(self, service: Any) -> Tuple[bool, str]:
        """Validate service name"""
        if not isinstance(service, str):
            return False, "Service must be a string"
        
        if len(service.strip()) == 0:
            return False, "Service cannot be empty"
        
        return True, ""
    
    def _validate_region(self, region: Any) -> Tuple[bool, str]:
        """Validate region"""
        if not isinstance(region, str):
            return False, "Region must be a string"
        
        # Basic region format validation
        if len(region.strip()) == 0:
            return False, "Region cannot be empty"
        
        return True, ""
    
    def _validate_cost(self, cost: Any) -> Tuple[bool, str]:
        """Validate cost amount"""
        try:
            cost_float = parse_cost_amount(str(cost))
            if cost_float < 0:
                return False, "Cost cannot be negative"
            return True, ""
        except (ValueError, TypeError):
            return False, "Cost must be a valid numeric value"
    
    def _validate_currency(self, currency: Any) -> Tuple[bool, str]:
        """Validate currency code"""
        if not isinstance(currency, str):
            return False, "Currency must be a string"
        
        # Basic currency code validation (3-letter codes)
        if not re.match(r'^[A-Z]{3}$', currency.upper()):
            return False, "Currency must be a 3-letter code (e.g., USD, EUR)"
        
        return True, ""
    
    def generate_quality_report(self, valid_records: List[Dict], invalid_records: List[Dict]) -> Dict[str, Any]:
        """Generate data quality report"""
        total_records = len(valid_records) + len(invalid_records)
        
        if total_records == 0:
            return {
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'quality_score': 0.0,
                'common_errors': []
            }
        
        # Count error types
        error_counts = {}
        for record in invalid_records:
            for error in record.get('_validation_errors', []):
                error_type = error.split(':')[0]  # Get error type before colon
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Sort errors by frequency
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        quality_score = (len(valid_records) / total_records) * 100
        
        return {
            'total_records': total_records,
            'valid_records': len(valid_records),
            'invalid_records': len(invalid_records),
            'quality_score': round(quality_score, 2),
            'common_errors': common_errors[:5],  # Top 5 errors
            'validation_timestamp': datetime.utcnow().isoformat()
        }


class CostDataEnricher:
    """Enriches cost data with additional metadata"""
    
    def __init__(self):
        self.service_categories = {
            'compute': ['EC2', 'Virtual Machines', 'Compute Engine'],
            'storage': ['S3', 'Storage Accounts', 'Cloud Storage'],
            'database': ['RDS', 'SQL Database', 'Cloud SQL'],
            'networking': ['CloudFront', 'Load Balancer', 'Cloud CDN'],
            'serverless': ['Lambda', 'Functions', 'Cloud Functions']
        }
    
    def enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single cost record with metadata"""
        enriched = record.copy()
        
        # Add service category
        enriched['service_category'] = self._categorize_service(record.get('service', ''))
        
        # Add cost per unit if usage quantity available
        cost = record.get('cost', 0)
        usage = record.get('usage_quantity', 0)
        if usage > 0:
            enriched['cost_per_unit'] = round(cost / usage, 4)
        else:
            enriched['cost_per_unit'] = 0
        
        # Add timestamp
        enriched['processed_at'] = datetime.utcnow().isoformat()
        
        return enriched
    
    def enrich_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich a batch of cost records"""
        return [self.enrich_record(record) for record in records]
    
    def _categorize_service(self, service_name: str) -> str:
        """Categorize service by type"""
        service_lower = service_name.lower()
        
        for category, services in self.service_categories.items():
            for service in services:
                if service.lower() in service_lower:
                    return category
        
        return 'other'
