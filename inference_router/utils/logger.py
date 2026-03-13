"""
Usage Logger - Tracks requests for analytics and online learning
"""
import json
from datetime import datetime
from typing import Optional


class UsageLogger:
    """Logs usage data for analytics and model improvement"""
    
    def __init__(self, log_file: str = "usage_logs.jsonl"):
        self.log_file = log_file
        self.buffer = []
        self.buffer_size = 100
    
    def log_request(
        self,
        user_id: str,
        prompt: str,
        complexity: str,
        model_used: str,
        cost: float,
        latency: float,
        rating: Optional[int] = None
    ):
        """Log a single request"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "prompt": prompt[:200],  # Truncate for privacy
            "complexity": complexity,
            "model_used": model_used,
            "cost": cost,
            "latency_ms": latency,
            "rating": rating
        }
        
        self.buffer.append(log_entry)
        
        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered logs to file"""
        if not self.buffer:
            return
        
        try:
            with open(self.log_file, "a") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")
            self.buffer.clear()
        except Exception as e:
            print(f"Error writing logs: {e}")
    
    def __del__(self):
        """Ensure logs are flushed on cleanup"""
        self.flush()
