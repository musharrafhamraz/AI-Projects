"""
Prompt Caching - Reuses KV cache for repeated prompt prefixes
"""
import hashlib
from typing import Optional
import redis


class PromptCache:
    """Caches responses for repeated prompt prefixes"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = None
        self.ttl = ttl
        self.min_prefix_length = 100
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis not available, caching disabled: {e}")
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt prefix"""
        prefix = prompt[:self.min_prefix_length]
        return hashlib.sha256(prefix.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, model: str) -> Optional[str]:
        """Check if we have a cached response"""
        if not self.redis_client or len(prompt) < self.min_prefix_length:
            return None
        
        try:
            cache_key = f"cache:{model}:{self._hash_prompt(prompt)}"
            return self.redis_client.get(cache_key)
        except Exception:
            return None
    
    def cache_response(self, prompt: str, model: str, response: str):
        """Cache the response with TTL"""
        if not self.redis_client or len(prompt) < self.min_prefix_length:
            return
        
        try:
            cache_key = f"cache:{model}:{self._hash_prompt(prompt)}"
            self.redis_client.setex(cache_key, self.ttl, response)
        except Exception:
            pass
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.redis_client:
            return {"enabled": False}
        
        try:
            return {
                "enabled": True,
                "cache_size": self.redis_client.dbsize(),
                "memory_usage": self.redis_client.info("memory")["used_memory_human"]
            }
        except Exception:
            return {"enabled": False}
