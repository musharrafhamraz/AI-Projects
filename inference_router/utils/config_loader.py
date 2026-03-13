"""
Configuration Loader - Loads settings from YAML
"""
import yaml
from typing import Dict, Any


class ConfigLoader:
    """Loads and validates configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "models": {
                "llama-8b-finetune": {
                    "endpoint": "http://localhost:8000/v1/completions",
                    "cost_per_1m_tokens": 0.01,
                    "type": "vllm"
                }
            },
            "user_tiers": {
                "free": {"daily_limit_usd": 1.0},
                "basic": {"daily_limit_usd": 10.0},
                "enterprise": {"daily_limit_usd": 1000.0}
            },
            "caching": {
                "enabled": True,
                "redis_url": "redis://localhost:6379",
                "ttl_seconds": 3600
            }
        }
    
    def get_models(self) -> Dict[str, Any]:
        """Get models configuration"""
        return self.config.get("models", {})
    
    def get_user_tiers(self) -> Dict[str, float]:
        """Get user tier limits"""
        tiers = self.config.get("user_tiers", {})
        return {
            tier: config["daily_limit_usd"]
            for tier, config in tiers.items()
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        return self.config.get("caching", {})
