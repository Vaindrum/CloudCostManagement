"""
Kafka consumer for real-time cost data streaming
"""
import asyncio
import json
from typing import Dict, List, Any, Callable
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from ..utils.logger import get_logger
from ..config.settings import settings

logger = get_logger(__name__)


class CostDataKafkaConsumer:
    """Kafka consumer for cost events"""
    
    def __init__(self, group_id: str = "ai_cost_agent"):
        self.group_id = group_id
        self.consumer = None
        self.producer = None
        self.is_running = False
        self.message_handlers = {}
    
    def start_consumer(self):
        """Initialize and start Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                settings.kafka_cost_events_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id=self.group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            logger.info(f"Kafka consumer started for topic: {settings.kafka_cost_events_topic}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {str(e)}")
            raise
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def consume_messages(self):
        """Main message consumption loop"""
        if not self.consumer:
            self.start_consumer()
        
        self.is_running = True
        logger.info("Started consuming cost events...")
        
        try:
            while self.is_running:
                # Poll for messages with timeout
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                if message_pack:
                    for topic_partition, messages in message_pack.items():
                        for message in messages:
                            await self._process_message(message)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in message consumption loop: {str(e)}")
        finally:
            self.stop_consumer()
    
    async def _process_message(self, message):
        """Process individual Kafka message"""
        try:
            data = message.value
            message_type = data.get('type', 'unknown')
            
            logger.info(f"Processing message type: {message_type}")
            
            # Route to appropriate handler
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(data)
            else:
                logger.warning(f"No handler registered for message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def stop_consumer(self):
        """Stop the Kafka consumer"""
        self.is_running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        logger.info("Kafka consumer stopped")
    
    def publish_message(self, topic: str, message: Dict[str, Any]):
        """Publish message to Kafka topic"""
        try:
            if not self.producer:
                self.start_consumer()
            
            future = self.producer.send(topic, message)
            record_metadata = future.get(timeout=10)
            
            logger.info(
                f"Message published to topic {topic}: "
                f"partition={record_metadata.partition}, offset={record_metadata.offset}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to publish message to {topic}: {str(e)}")
            raise


class CostEventProcessor:
    """Processes different types of cost events from Kafka"""
    
    def __init__(self):
        self.consumer = CostDataKafkaConsumer()
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for different event types"""
        self.consumer.register_handler('cost_data', self.handle_cost_data)
        self.consumer.register_handler('anomaly_detected', self.handle_anomaly)
        self.consumer.register_handler('optimization_trigger', self.handle_optimization)
    
    async def handle_cost_data(self, data: Dict[str, Any]):
        """Handle new cost data events"""
        logger.info(f"Processing cost data for {data.get('provider', 'unknown')} - "
                   f"{data.get('service', 'unknown')}")
        
        # Store in database, trigger ML analysis, etc.
        # This would integrate with your ML pipeline
        
        cost_amount = data.get('cost', 0)
        if cost_amount > 1000:  # Example threshold
            logger.warning(f"High cost detected: ${cost_amount}")
    
    async def handle_anomaly(self, data: Dict[str, Any]):
        """Handle anomaly detection events"""
        logger.warning(f"Cost anomaly detected: {data.get('description', 'Unknown')}")
        
        # Trigger alerts, notifications, etc.
        anomaly_score = data.get('anomaly_score', 0)
        if anomaly_score > 0.8:
            # Trigger immediate alert
            self.consumer.publish_message(
                'urgent_alerts',
                {
                    'type': 'critical_anomaly',
                    'data': data,
                    'timestamp': data.get('timestamp')
                }
            )
    
    async def handle_optimization(self, data: Dict[str, Any]):
        """Handle optimization trigger events"""
        logger.info(f"Optimization triggered for {data.get('resource_id', 'unknown')}")
        
        # Queue optimization actions
        optimization_type = data.get('optimization_type', 'unknown')
        logger.info(f"Queuing {optimization_type} optimization")
    
    async def start(self):
        """Start the event processor"""
        await self.consumer.consume_messages()
    
    def stop(self):
        """Stop the event processor"""
        self.consumer.stop_consumer()


# Sample function to simulate publishing cost events
def simulate_cost_events():
    """Simulate publishing cost events for testing"""
    import random
    from datetime import datetime
    
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    
    # Simulate different types of events
    events = [
        {
            'type': 'cost_data',
            'provider': 'aws',
            'service': 'EC2',
            'region': 'us-east-1',
            'cost': random.uniform(100, 2000),
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'type': 'anomaly_detected',
            'provider': 'azure',
            'service': 'Virtual Machines',
            'anomaly_score': 0.85,
            'description': 'Unexpected cost spike detected',
            'timestamp': datetime.utcnow().isoformat()
        }
    ]
    
    for event in events:
        producer.send(settings.kafka_cost_events_topic, event)
        print(f"Published event: {event['type']}")
    
    producer.close()
